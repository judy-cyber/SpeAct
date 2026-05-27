import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from awq import AutoAWQForCausalLM
from datasets import Dataset, load_dataset
import pandas as pd
import time
import re
import json
import requests
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, TimeoutError

# =================================================================
# HotpotQA evaluation helper functions
# =================================================================
ANSWER_NORMALIZATION_RE = re.compile(r"\b(a|an|the)\b")
MULTISPACE_RE = re.compile(r"\s+")
FINAL_ANSWER_RE_LIST = [
    re.compile(r"Final Answer\s*:\s*(.+)", flags=re.IGNORECASE),
    re.compile(r"Answer\s*:\s*(.+)", flags=re.IGNORECASE),
]
SPECIAL_EOT_TOKENS = ["<|eot_id|>", "<|im_end|>", "<|endoftext|>"]
SEARCH_WIKIPEDIA_PREFIX = "SearchWikipedia["
ACTION_MARKER = "Action:"
ACTION_PREFIX_TEXT = "Action: SearchWikipedia["
PUNCT_TABLE = str.maketrans("", "", r"!\"#$%&'()*+,./:;<=>?@[\\]^_`{|}~")


def normalize_answer(text):
    if text is None:
        return ""
    text = str(text).lower().strip()
    text = text.translate(PUNCT_TABLE)
    text = ANSWER_NORMALIZATION_RE.sub(" ", text)
    text = MULTISPACE_RE.sub(" ", text)
    return text.strip()


def clean_special_tokens(text: str) -> str:
    if not isinstance(text, str):
        return ""
    for tok in SPECIAL_EOT_TOKENS:
        text = text.replace(tok, "")
    return text.strip()



def extract_final_answer(text):
    if not isinstance(text, str):
        return ""
    for pattern in FINAL_ANSWER_RE_LIST:
        matches = pattern.findall(text)
        if matches:
            answer = matches[-1].strip()
            answer = clean_special_tokens(answer)
            if answer.endswith(self_eos := "</s>"):
                answer = answer[: -len(self_eos)].rstrip()
            return answer
    return ""


def exact_match_score(prediction, ground_truth):
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction, ground_truth):
    """Compute the F1 score using the ReWoo/ReAct evaluation convention."""
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    
    if not pred_tokens and not gt_tokens:
        return 1.0
    if not pred_tokens or not gt_tokens:
        return 0.0
    
    common = set(pred_tokens) & set(gt_tokens)
    num_same = len(common)
    
    if num_same == 0:
        return 0.0
    
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def get_first_available(entry: Dict[str, Any], candidates: List[str], default: Optional[Any]):
    for key in candidates:
        value = entry.get(key, None)
        if value is None:
            continue
        if isinstance(value, str):
            value = value.strip()
            if value != "":
                return value
            continue
        if isinstance(value, list):
            if len(value) > 0:
                return value
            continue
        if isinstance(value, dict):
            if len(value) > 0:
                return value
            continue
        return value
    return default


class ReActSpeculativeEngine:
    def __init__(
        self,
        big_path,
        small_path,
        wiki_data_path,
        threshold,
        lookahead,
        enable_ste,
        ste_max_workers,
        ste_min_query_len,
        ste_wait_timeout,
        ste_ttl_sec,
        observation_topk_docs,
        observation_topk_paragraphs,
        observation_max_words,
        observation_max_chars,
        enable_shadow_kv,
        enable_hidden_probe,
        hidden_probe_steps,
    ):
        torch.cuda.empty_cache()
        self.threshold = threshold
        self.lookahead = lookahead
        self.wiki_data_path = wiki_data_path
        self.enable_ste = enable_ste
        self.ste_min_query_len = ste_min_query_len
        self.ste_wait_timeout = ste_wait_timeout
        self.ste_ttl_sec = ste_ttl_sec
        self.observation_topk_docs = max(1, int(observation_topk_docs))
        self.observation_topk_paragraphs = max(1, int(observation_topk_paragraphs))
        self.observation_max_words = max(64, int(observation_max_words))
        self.observation_max_chars = max(256, int(observation_max_chars))
        self.enable_shadow_kv = enable_shadow_kv
        self.enable_hidden_probe = enable_hidden_probe and enable_ste
        self.hidden_probe_steps = max(1, int(hidden_probe_steps))
        self.thought_plan_token_cache = {}
        self.encode_cache = {}
        self.decode_cache = {}
        self.http_session = requests.Session()
        self.http_headers = {
            'User-Agent': 'HotpotQA-SpeculativeEngine/1.0 requests-library'
        }
        self.request_timeout = (1.5, 6.0)

        print(f">>> Loading large model (70B AWQ)...")
        self.big_model_wrapper = AutoAWQForCausalLM.from_quantized(
            big_path, fuse_layers=True, trust_remote_code=True, device_map="auto"
        )
        self.big_model = self.big_model_wrapper.model

        print(f">>> Loading small model (3B)...")
        self.small_model = AutoModelForCausalLM.from_pretrained(
            small_path, device_map="auto", torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
        )

        self.tokenizer = AutoTokenizer.from_pretrained(big_path, trust_remote_code=True)
        self.small_tokenizer = AutoTokenizer.from_pretrained(small_path, trust_remote_code=True)
        if len(self.tokenizer) != len(self.small_tokenizer):
            raise ValueError(
                f"Tokenizer vocab mismatch: big={len(self.tokenizer)} small={len(self.small_tokenizer)}"
            )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"

        if getattr(self.big_model.config, "pad_token_id", None) is None:
            self.big_model.config.pad_token_id = self.tokenizer.pad_token_id
        if getattr(self.small_model.config, "pad_token_id", None) is None:
            self.small_model.config.pad_token_id = self.tokenizer.pad_token_id

        # =================================================================
        # State machine initialization (recognize chat model end tokens)
        # =================================================================
        self.nl_id = self._encode_text("\n")[-1]
        self.eos_id = self.tokenizer.eos_token_id
        self.eot_id = self._resolve_eot_id()
 
        self.action_ids = {self._encode_text(w)[-1] for w in ["Action", " Action", "\nAction"]}
        self.obs_ids = {self._encode_text(w)[-1] for w in ["Observation", " Observation", "\nObservation"]}
        self.thought_ids = {self._encode_text(w)[-1] for w in ["Thought", " Thought", "\nThought"]}
        self.base_action_id = self._encode_text("Action")[-1]
        self.base_obs_id = self._encode_text("Observation")[-1]
        self.base_thought_id = self._encode_text("Thought")[-1]

        self.log_csv = "qwen_20_final_integrated_results_hotpotqa.csv"

        self.error_count = 0
        self.query_history = []
        self.query_signature_history = []
        self.repeat_action_count = 0

        self.ste_executor = ThreadPoolExecutor(max_workers=ste_max_workers) if self.enable_ste else None
        self.pending_tool_prefetch = None
        self.prefetch_stats = {}
        self.last_run_stats = {}
        self.total_tool_time_sec = 0.0
        self.action_prefix_ids = self._encode_text(ACTION_PREFIX_TEXT)
        self.action_suffix_ids = self._encode_text("]")
        self._reset_prefetch_state()
        self._reset_trace_state()
        self.committed_output_chunks = []

        # Switched to online Wikipedia API retrieval instead of a local large corpus
        self.wiki_corpus = self._load_wiki_corpus()

    def _reset_prefetch_state(self):
        self.pending_tool_prefetch = None
        self.hidden_probe_query = None
        self.total_tool_time_sec = 0.0
        self.prefetch_stats = {
            "submitted": 0,
            "hit": 0,
            "miss": 0,
            "stale": 0,
            "timeout": 0,
            "cancelled": 0,
            "reused": 0,
            "skipped_duplicate": 0,
            "saved_wait_ms": 0.0,
            "sync_tool_calls": 0,
            "prefetch_tool_calls": 0,
            "compute_submitted": 0,
            "compute_hit": 0,
            "compute_reject_discard": 0,
            "compute_sync_fallback": 0,
            "shadow_kv_committed": 0,
            "hidden_probe_seen": 0,
            "hidden_probe_ready": 0,
            "hidden_probe_prefetch_triggered": 0,
            "hidden_probe_prefetch_duplicate": 0,
            "action_seen": 0,
            "action_closed": 0,
            "prefetch_opportunities": 0,
            "prefetch_effective_hits": 0,
            "prefetch_wait_time_ms": 0.0,
            "prefetch_latency_ms": 0.0,
        }

    def _reset_trace_state(self):
        self.generation_trace = []
        self.tool_usage_trace = []
        self.committed_output_chunks = []
        self.trace_stats = {
            "big_direct_tokens": 0,
            "small_drafted_tokens": 0,
            "accepted_draft_tokens": 0,
            "big_resample_tokens": 0,
        }

    def _encode_text(self, text: str) -> List[int]:
        cached = self.encode_cache.get(text)
        if cached is None:
            cached = self.tokenizer.encode(text, add_special_tokens=False)
            self.encode_cache[text] = cached
        return cached

    def _decode_token(self, token_id: int) -> str:
        token_id = int(token_id)
        cached = self.decode_cache.get(token_id)
        if cached is None:
            cached = self.tokenizer.decode([token_id], skip_special_tokens=False)
            self.decode_cache[token_id] = cached
        return cached

    def _resolve_eot_id(self):
        candidate_tokens = ["<|im_end|>", "<|eot_id|>", "<|endoftext|>"]
        for tok in candidate_tokens:
            tok_id = self.tokenizer.convert_tokens_to_ids(tok)
            if tok_id is not None and tok_id != self.tokenizer.unk_token_id:
                return tok_id

        additional_special_tokens = getattr(self.tokenizer, "additional_special_tokens", []) or []
        for tok in additional_special_tokens:
            lowered = tok.lower()
            if "end" in lowered or "eot" in lowered:
                tok_id = self.tokenizer.convert_tokens_to_ids(tok)
                if tok_id is not None and tok_id != self.tokenizer.unk_token_id:
                    return tok_id

        return self.tokenizer.eos_token_id

    def _build_prompt_text(self, system_content: str, user_content: str) -> str:
        if hasattr(self.tokenizer, "apply_chat_template") and getattr(self.tokenizer, "chat_template", None):
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ]
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return f"System: {system_content}\n\nUser: {user_content}\nAssistant: "

    def _record_trace(self, token_id, source, accepted):
        piece = self._decode_token(token_id)
        self.generation_trace.append({
            "token_id": int(token_id),
            "text": piece,
            "source": source,
            "accepted": accepted,
        })
        if source == "big_direct":
            self.trace_stats["big_direct_tokens"] += 1
        elif source == "small_draft":
            self.trace_stats["small_drafted_tokens"] += 1
            if accepted:
                self.trace_stats["accepted_draft_tokens"] += 1
        elif source == "big_resample":
            self.trace_stats["big_resample_tokens"] += 1

    def _render_trace_summary(self, max_chars):
        if not self.generation_trace:
            return "[Generation Trace]\n(no generated tokens)\n"

        header = (
            "\n" + "=" * 24 + " Generation Trace " + "=" * 24 + "\n"
            "[S-ACC] = small-model draft accepted by large model | "
            "[B-DIR] = token generated directly by large model | "
            "[B-RES] = large-model rewrite after rejecting small-model draft\n"
            + "-" * 68 + "\n"
        )
        parts = []
        for item in self.generation_trace:
            if item["source"] == "small_draft":
                tag = "[S-ACC]"
            elif item["source"] == "big_direct":
                tag = "[B-DIR]"
            else:
                tag = "[B-RES]"
            text_piece = item["text"].replace("\n", "\\n")
            parts.append(f"{tag}{text_piece}")
        body = "".join(parts)
        if len(body) > max_chars:
            body = body[:max_chars] + "... [TRACE TRUNCATED]"
        footer = "\n" + "=" * 68 + "\n"
        return header + body + footer

    def _load_wiki_corpus(self):
        print(f">>> Online Wikipedia API retrieval is enabled.")
        if self.enable_ste:
            print(f">>> STE is enabled: tool results are prefetched asynchronously when the small model drafts a complete Action.")
        return None

    def close(self):
        if getattr(self, "http_session", None) is not None:
            try:
                self.http_session.close()
            except Exception:
                pass
            self.http_session = None
        if getattr(self, "ste_executor", None) is not None:
            try:
                self.ste_executor.shutdown(wait=False, cancel_futures=True)
            except TypeError:
                self.ste_executor.shutdown(wait=False)
            except Exception:
                pass
            self.ste_executor = None

    def _update_phase(self, token_val, current_phase):
        if token_val in self.action_ids:
            return 1
        elif token_val in self.obs_ids:
            return 2
        elif token_val in self.thought_ids:
            return 0
        return current_phase

    def _phase_after_tool_injection(self, observation_text: str) -> int:
        if not observation_text:
            return 0
        if "\nThought:" in observation_text or observation_text.rstrip().endswith("Thought:"):
            return 0
        if observation_text.startswith("Observation:"):
            return 2
        return 0

    def _crop_cache(self, cache: DynamicCache, length: int):
        if cache is None:
            return cache
        if hasattr(cache, "crop"):
            try:
                cache.crop(length)
                return cache
            except Exception:
                pass
        k_attr = "key_cache" if hasattr(cache, "key_cache") else "_key_cache"
        v_attr = "value_cache" if hasattr(cache, "value_cache") else "_value_cache"
        keys, values = getattr(cache, k_attr, []), getattr(cache, v_attr, [])
        for i in range(len(keys)):
            keys[i] = keys[i][:, :, :length, :]
            values[i] = values[i][:, :, :length, :]
        for attr in ["_seen_tokens", "seen_tokens", "last_seen_seq_assign"]:
            if hasattr(cache, attr):
                setattr(cache, attr, length)
        return cache

    def _extract_search_query(self, text):
        # Add quote tolerance so SearchWikipedia["Query"] can still be parsed correctly
        match = re.search(r"SearchWikipedia\[\s*\"?(.*?)\"?\s*\]", text, re.DOTALL)
        if match:
            return match.group(1).strip(), True
        if SEARCH_WIKIPEDIA_PREFIX in text:
            return None, False
        return None, False

    def _normalize_query(self, query: Optional[str]) -> str:
        if not query:
            return ""
        query = str(query).strip().strip('"').strip("'")
        return MULTISPACE_RE.sub(" ", query.lower()).strip()

    def _trim_query_to_safe_prefix(self, query: Optional[str]) -> str:
        normalized = self._normalize_query(query)
        if not normalized:
            return ""
        safe = normalized.rstrip(" ,;:([{-_/\\")
        parts = safe.split()
        if len(parts) <= 1:
            return safe
        last_token = parts[-1]
        if last_token.isalpha() and len(last_token) <= 3:
            parts = parts[:-1]
        return " ".join(parts).strip()

    def _queries_prefix_compatible(self, probe_query: Optional[str], action_query: Optional[str]) -> bool:
        probe_norm = self._normalize_query(probe_query)
        action_norm = self._normalize_query(action_query)
        if not probe_norm or not action_norm:
            return False
        if probe_norm == action_norm:
            return True
        probe_safe = self._trim_query_to_safe_prefix(probe_norm)
        if not probe_safe:
            return False
        if action_norm == probe_safe:
            return True
        return action_norm.startswith(probe_safe + " ")

    def _query_signature(self, query: Optional[str]) -> str:
        normalized = self._normalize_query(query)
        if not normalized:
            return ""
        tokens = re.findall(r"\w+", normalized)
        if not tokens:
            return ""
        stop_tokens = {
            "people", "exact",
            "entity", "information", "need", "still", "contain", "contains",
            "with", "and", "the", "a", "an"
        }
        filtered_tokens = []
        seen = set()
        for token in tokens:
            if token in stop_tokens:
                continue
            if token in seen:
                continue
            seen.add(token)
            filtered_tokens.append(token)
        if not filtered_tokens:
            filtered_tokens = []
            seen = set()
            for token in tokens:
                if token in seen:
                    continue
                seen.add(token)
                filtered_tokens.append(token)
        return " ".join(filtered_tokens)

    def _clean_query_candidate(self, query: Optional[str]) -> str:
        if not query:
            return ""
        query = str(query).strip().strip('"').strip("'")
        query = query.replace(SEARCH_WIKIPEDIA_PREFIX, " ").replace("]", " ")
        query = re.sub(r"^[\s\-:;,]+", "", query)
        query = re.sub(r"[\s\-:;,\.]+$", "", query)
        query = MULTISPACE_RE.sub(" ", query)
        return query.strip()

    def _tool_plan_min_query_len(self) -> int:
        return max(1, min(self.ste_min_query_len, 8))

    def _cache_seq_len(self, cache: DynamicCache) -> int:
        if cache is None:
            return 0
        if hasattr(cache, "get_seq_length"):
            try:
                return int(cache.get_seq_length())
            except Exception:
                pass
        for attr in ["_seen_tokens", "seen_tokens", "last_seen_seq_assign"]:
            if hasattr(cache, attr):
                try:
                    return int(getattr(cache, attr))
                except Exception:
                    pass
        return 0

    def _is_tool_plan_query_ready(self, query: Optional[str]) -> bool:
        safe_query = self._trim_query_to_safe_prefix(query)
        return bool(safe_query) and len(safe_query) >= self._tool_plan_min_query_len()

    def _get_cached_plan_token_ids(self, thought_plan: Optional[str]) -> List[int]:
        if not thought_plan:
            return []
        query_piece = self._clean_query_candidate(thought_plan)
        if not query_piece:
            return []
        cached = self.thought_plan_token_cache.get(query_piece)
        if cached is None:
            cached = self._encode_text(query_piece)
            self.thought_plan_token_cache[query_piece] = cached
        return cached
 
    def _maybe_boost_action_start(self, logits, phase, action_buffer, thought_plan, boost):
        prefix = action_buffer or ""
        target_prefix = ACTION_PREFIX_TEXT
        target_ids = self.action_prefix_ids
        prefix_ids = self._encode_text(prefix) if prefix else []
        matched = 0
        max_match = min(len(prefix_ids), len(target_ids))
        while matched < max_match and prefix_ids[matched] == target_ids[matched]:
            matched += 1
 
        if phase == 0:
            logits[0, self.base_action_id] += boost
            return logits

        if phase != 1:
            return logits
 
        if matched < len(target_ids):
            logits[0, target_ids[matched]] += boost
            return logits
 
        if thought_plan:
            query_ids = self._get_cached_plan_token_ids(thought_plan)
            if query_ids:
                typed_query = prefix[len(target_prefix):] if prefix.startswith(target_prefix) else ""
                typed_ids = self._encode_text(typed_query) if typed_query else []
                query_match = 0
                max_query_match = min(len(typed_ids), len(query_ids))
                while query_match < max_query_match and typed_ids[query_match] == query_ids[query_match]:
                    query_match += 1
                if query_match < len(query_ids):
                    logits[0, query_ids[query_match]] += boost
                elif not prefix.endswith("]") and self.action_suffix_ids:
                    logits[0, self.action_suffix_ids[0]] += boost
        return logits

    def _parse_action_query(self, action_buffer):
        query, is_closed = self._extract_search_query(action_buffer)

        # Fallback: support outputs like `Action: Query` without brackets
        if query is None and ACTION_MARKER in action_buffer:
            query = action_buffer.split(ACTION_MARKER)[-1].strip()
            is_closed = True

        if query is not None:
            query = query.strip()
        return query, is_closed

    def _truncate_text_by_words(self, text: str, max_words: Optional[int]) -> str:
        compact = MULTISPACE_RE.sub(" ", (text or "")).strip()
        if not compact:
            return ""
        max_words = self.observation_max_words if max_words is None else max(1, int(max_words))
        words = compact.split()
        if len(words) <= max_words:
            return compact
        return " ".join(words[:max_words]).strip() + " [CONTENT TRUNCATED]"

    def _search_wikipedia(self, query, topk, max_chars):
        search_url = "https://en.wikipedia.org/w/api.php"
        session = self.http_session

        if topk is None:
            topk = self.observation_topk_docs
        if max_chars is None:
            max_chars = self.observation_max_chars
        search_params = {
            "action": "query", "list": "search", "srsearch": query,
            "format": "json", "srlimit": topk, "utf8": 1
        }
        try:
            r = session.get(search_url, params=search_params, headers=self.http_headers, timeout=self.request_timeout)
            r.raise_for_status()
            search_items = r.json().get("query", {}).get("search", [])
            if not search_items:
                return []

            final_results = []
            for item in search_items:
                title = item['title']
                content_params = {
                    "action": "query", "prop": "extracts", "exintro": True,
                    "explaintext": True, "titles": title, "redirects": 1,
                    "format": "json", "utf8": 1
                }
                cr = session.get(search_url, params=content_params, headers=self.http_headers, timeout=self.request_timeout)
                cr.raise_for_status()
                cdata = cr.json()

                pages = cdata.get("query", {}).get("pages", {})
                for page_id, page_info in pages.items():
                    if page_id == "-1":
                        continue

                    extract_text = page_info.get("extract", "").strip()
                    if "may refer to" in extract_text[:200].lower():
                        continue

                    if extract_text:
                        truncated_text = self._truncate_text_by_words(extract_text)

                        final_results.append({
                            "title": page_info.get("title", title),
                            "text": truncated_text
                        })

                if len(final_results) >= topk:
                    break
            return final_results
        except requests.Timeout as e:
            print(f"\n[Wikipedia Tool Timeout] query={query!r} | timeout={self.request_timeout} | error={e}")
            return []
        except requests.RequestException as e:
            print(f"\n[Wikipedia Tool Request Error] query={query!r} | error={e}")
            return []
        except Exception as e:
            print(f"\n[Wikipedia Tool Error] {e}")
            return []

    def _token_overlap_score(self, query: str, text: str) -> float:
        query_tokens = set(re.findall(r"\w+", (query or "").lower()))
        text_tokens = set(re.findall(r"\w+", (text or "").lower()))
        if not query_tokens or not text_tokens:
            return 0.0
        overlap = len(query_tokens & text_tokens)
        return overlap / max(1, len(query_tokens))

    def _split_evidence_paragraphs(self, text: str) -> List[str]:
        if not text:
            return []
        raw_parts = re.split(r"\n\s*\n|(?<=[\.!?])\s+(?=[A-Z])", text)
        cleaned = []
        seen = set()
        for part in raw_parts:
            piece = re.sub(r"\s+", " ", part).strip()
            if not piece:
                continue
            lowered = piece.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            cleaned.append(piece)
        return cleaned

    def _select_relevant_observation_paragraphs(self, query: str, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        scored_paragraphs = []
        for doc in docs:
            title = doc.get("title", "")
            paragraphs = self._split_evidence_paragraphs(doc.get("text", ""))
            if not paragraphs:
                continue
            for para_idx, paragraph in enumerate(paragraphs):
                first_sentence_match = re.split(r"(?<=[\.!?])\s+", paragraph, maxsplit=1)
                first_sentence = first_sentence_match[0] if first_sentence_match else paragraph
                score_text = f"{title} {first_sentence}"
                score = self._token_overlap_score(query, score_text)
                scored_paragraphs.append({
                    "title": title,
                    "paragraph": paragraph,
                    "score": score,
                    "doc_rank": doc.get("doc_rank", 0),
                    "para_rank": para_idx,
                })
        scored_paragraphs.sort(key=lambda item: (-item["score"], item["doc_rank"], item["para_rank"]))
        return scored_paragraphs[:self.observation_topk_paragraphs]

    def _build_observation_text(self, query: str, docs: List[Dict[str, Any]]) -> str:
        selected = self._select_relevant_observation_paragraphs(query, docs)
        if not selected:
            return (
                f"Observation: No relevant Wikipedia evidence found for '{query}'. "
                "Do not keep appending generic words to the same query. "
                "If the query already centers on the same entity, do not search again with filler terms. "
                "Either search only the core entity name once, or try a genuinely different entity to search instead of ending immediately.\n"
                "Thought: Try switching to a different entity or relation for the next search, and do not end the reasoning yet just because this query repeated.\n"
            )

        lines = []
        for idx, item in enumerate(selected, 1):
            snippet = item["paragraph"].replace("\n", " ").strip()
            if len(snippet) > self.observation_max_chars:
                snippet = snippet[:self.observation_max_chars].rstrip() + " [CONTENT TRUNCATED]"
            lines.append(f"[{idx}] Title: {item['title']} | Evidence: {snippet}")
        joined = "\n".join(lines)
        return f"Observation: Wikipedia search results for '{query}':\n{joined}\nThought: "

    def _build_shadow_kv(self, obs_ids, seq_len_before_commit, base_big_cache, base_small_cache):
        if not self.enable_shadow_kv or base_small_cache is None or not obs_ids:
            return None
        try:
            shadow_small_cache = DynamicCache()
            shadow_small_cache.key_cache = [layer.clone() for layer in base_small_cache.key_cache]
            shadow_small_cache.value_cache = [layer.clone() for layer in base_small_cache.value_cache]
            for attr in ["_seen_tokens", "seen_tokens", "last_seen_seq_assign"]:
                if hasattr(base_small_cache, attr):
                    setattr(shadow_small_cache, attr, getattr(base_small_cache, attr))
            obs_tensor_small = torch.tensor([obs_ids], device=self.small_model.device)
            pos_small = torch.arange(
                seq_len_before_commit,
                seq_len_before_commit + len(obs_ids),
                device=self.small_model.device,
            ).unsqueeze(0)
            small_out = self.small_model(
                obs_tensor_small,
                past_key_values=shadow_small_cache,
                position_ids=pos_small,
                use_cache=True,
            )
            return {
                "small_cache": shadow_small_cache,
                "small_logits": small_out.logits[:, -1, :],
            }
        except Exception:
            return None

    def _prepare_tool_payload(self, query, source, seq_len_before_commit, base_big_cache, base_small_cache):
        tool_start_time = time.time()
        docs = self._search_wikipedia(query, self.observation_topk_docs, self.observation_max_chars)
        for idx, doc in enumerate(docs):
            doc["doc_rank"] = idx
        observation_text = self._build_observation_text(query, docs)
        obs_ids = self._encode_text(observation_text)
        raw_observation_token_count = sum(
            len(self._encode_text(self._truncate_text_by_words(doc.get("text", ""))))
            for doc in docs
        )
        shadow_state = None
        if seq_len_before_commit is not None and base_small_cache is not None:
            shadow_state = self._build_shadow_kv(obs_ids, seq_len_before_commit, base_big_cache, base_small_cache)
        if source == "prefetch":
            self.prefetch_stats["prefetch_tool_calls"] += 1
        else:
            self.prefetch_stats["sync_tool_calls"] += 1
        tool_elapsed_sec = time.time() - tool_start_time
        return {
            "query": query,
            "normalized_query": self._normalize_query(query),
            "docs": docs,
            "source": source,
            "fetched_at": time.time(),
            "observation_text": observation_text,
            "obs_ids": obs_ids,
            "raw_observation_token_count": raw_observation_token_count,
            "shadow_state": shadow_state,
            "seq_len_before_commit": seq_len_before_commit,
            "tool_time_sec": tool_elapsed_sec,
        }

    def _invalidate_pending_prefetch(self, reason, clear_all):
        pending = self.pending_tool_prefetch
        self.pending_tool_prefetch = None
        self.hidden_probe_query = None
        if pending is None:
            return
        if reason == "cancelled":
            self.prefetch_stats["cancelled"] += 1
        elif reason == "stale":
            self.prefetch_stats["stale"] += 1

    def _submit_prefetch_candidate(self, query, source_stage):
        if not self.enable_ste or self.ste_executor is None:
            return False

        normalized_query = self._normalize_query(query)
        effective_query = self._trim_query_to_safe_prefix(normalized_query)
        min_query_len = self._tool_plan_min_query_len()
        if not effective_query or len(effective_query) < min_query_len:
            return False

        pending = self.pending_tool_prefetch
        now = time.time()
        if pending is not None:
            pending_match_key = pending.get("safe_prefix") or pending.get("normalized_query")
            same_query = pending_match_key == effective_query
            is_fresh = (now - pending.get("start_time", now)) <= self.ste_ttl_sec
            if same_query and is_fresh:
                self.prefetch_stats["skipped_duplicate"] += 1
                self.prefetch_stats["hidden_probe_prefetch_duplicate"] += 1
                return False
            if not is_fresh:
                self._invalidate_pending_prefetch("stale", False)

        future = self.ste_executor.submit(self._prepare_tool_payload, effective_query, "prefetch", None, None, None)
        self.pending_tool_prefetch = {
            "query": query,
            "normalized_query": normalized_query,
            "safe_prefix": effective_query,
            "future": future,
            "start_time": now,
            "status": "pending",
            "source_stage": source_stage,
        }
        self.hidden_probe_query = effective_query
        self.prefetch_stats["submitted"] += 1
        self.prefetch_stats["compute_submitted"] += 1
        self.prefetch_stats["hidden_probe_prefetch_triggered"] += 1
        return True

    def _prefetch_tool_async(self, query):
        return self._submit_prefetch_candidate(query, "hidden_probe")

    def _consume_prefetch_if_match(self, query):
        normalized_query = self._normalize_query(query)
        pending = self.pending_tool_prefetch
        if pending is None:
            self.prefetch_stats["miss"] += 1
            return None

        pending_query = pending.get("safe_prefix") or pending.get("normalized_query")
        if not self._queries_prefix_compatible(pending_query, normalized_query):
            self.prefetch_stats["miss"] += 1
            return None

        self.prefetch_stats["reused"] += 1
        wait_start = time.time()
        try:
            payload = pending["future"].result(timeout=self.ste_wait_timeout)
            waited_ms = max(0.0, (time.time() - wait_start) * 1000.0)
            total_elapsed_ms = max(0.0, (time.time() - pending["start_time"]) * 1000.0)
            saved_wait_ms = max(0.0, total_elapsed_ms - waited_ms)
            self.prefetch_stats["hit"] += 1
            self.prefetch_stats["saved_wait_ms"] += saved_wait_ms
            self.prefetch_stats["prefetch_wait_time_ms"] += waited_ms
            self.prefetch_stats["prefetch_latency_ms"] += total_elapsed_ms
            if saved_wait_ms > 1e-6:
                self.prefetch_stats["prefetch_effective_hits"] += 1
            self.pending_tool_prefetch = None
            self.hidden_probe_query = None
            return payload
        except TimeoutError:
            self.prefetch_stats["timeout"] += 1
            return None
        except Exception:
            self.prefetch_stats["miss"] += 1
            self.pending_tool_prefetch = None
            self.hidden_probe_query = None
            return None

    def _commit_tool_payload(self, tool_payload, input_ids, big_cache, small_cache):
        tool_obs_str = tool_payload["observation_text"]
        self.tool_usage_trace.append(tool_payload)
        self.total_tool_time_sec += float(tool_payload.get("tool_time_sec", 0.0))
        print(tool_obs_str, end="")
        self.committed_output_chunks.append(tool_obs_str)
        obs_ids = tool_payload["obs_ids"]
        obs_tensor_big = torch.tensor([obs_ids], device=self.big_model.device)
        obs_tensor_small = torch.tensor([obs_ids], device=self.small_model.device)
        seq_len_before_commit = input_ids.shape[1]
        input_ids = torch.cat([input_ids, obs_tensor_big], dim=-1)
        shadow_state = tool_payload.get("shadow_state")
        reuse_small_shadow = shadow_state is not None and tool_payload.get("seq_len_before_commit") == seq_len_before_commit
        if reuse_small_shadow:
            small_cache.key_cache = shadow_state["small_cache"].key_cache
            small_cache.value_cache = shadow_state["small_cache"].value_cache
            for attr in ["_seen_tokens", "seen_tokens", "last_seen_seq_assign"]:
                if hasattr(shadow_state["small_cache"], attr):
                    setattr(small_cache, attr, getattr(shadow_state["small_cache"], attr))
            self.prefetch_stats["shadow_kv_committed"] += 1

        pos_big = torch.arange(seq_len_before_commit, seq_len_before_commit + len(obs_ids), device=self.big_model.device).unsqueeze(0)
        big_out = self.big_model(obs_tensor_big, past_key_values=big_cache, position_ids=pos_big, use_cache=True)
        if reuse_small_shadow:
            small_logits = shadow_state["small_logits"]
        else:
            pos_small = torch.arange(seq_len_before_commit, seq_len_before_commit + len(obs_ids), device=self.small_model.device).unsqueeze(0)
            small_out = self.small_model(obs_tensor_small, past_key_values=small_cache, position_ids=pos_small, use_cache=True)
            small_logits = small_out.logits[:, -1, :]
        return input_ids, big_out.logits[:, -1, :], small_logits, self._phase_after_tool_injection(tool_obs_str)

    def _materialize_fallback_payload(self, observation_text, input_ids):
        return {
            "query": None,
            "normalized_query": "",
            "docs": [],
            "source": "sync",
            "fetched_at": time.time(),
            "observation_text": observation_text,
            "tool_time_sec": 0.0,
            "obs_ids": self._encode_text(observation_text),
            "raw_observation_token_count": len(self._encode_text(observation_text)),
            "shadow_state": None,
            "seq_len_before_commit": input_ids.shape[1],
        }

    def _execute_tool_query(self, query, seq_len_before_commit, base_big_cache, base_small_cache):
        self.prefetch_stats["prefetch_opportunities"] += 1
        prefetched = self._consume_prefetch_if_match(query)
        if prefetched is not None:
            prefetched["seq_len_before_commit"] = seq_len_before_commit
            if prefetched.get("shadow_state") is None and self.enable_shadow_kv:
                prefetched["shadow_state"] = self._build_shadow_kv(
                    prefetched["obs_ids"],
                    seq_len_before_commit,
                    base_big_cache,
                    base_small_cache,
                )
            self.prefetch_stats["compute_hit"] += 1
            return prefetched
        self.prefetch_stats["compute_sync_fallback"] += 1
        return self._prepare_tool_payload(
            query,
            "sync",
            seq_len_before_commit,
            base_big_cache,
            base_small_cache,
        )

    def _execute_tool(self, action_buffer, input_ids, big_cache, small_cache):
        query, is_closed = self._parse_action_query(action_buffer)

        if not query:
            return self._materialize_fallback_payload("Observation: Error: Empty search query.\nThought: ", input_ids)

        if not is_closed:
            return self._materialize_fallback_payload("Observation: Error: Incomplete SearchWikipedia[...] block.\nThought: ", input_ids)

        return self._execute_tool_query(
            query,
            input_ids.shape[1],
            big_cache,
            small_cache,
        )

    def _handle_tool_trigger(self, action_buffer, input_ids, big_cache, small_cache):
        extracted_query, _ = self._parse_action_query(action_buffer)
        current_query = extracted_query if extracted_query else action_buffer.strip()
        normalized_query = self._normalize_query(current_query)
        query_signature = self._query_signature(current_query)
        self.hidden_probe_query = None

        is_repeat = normalized_query in self.query_history if normalized_query else False
        is_same_signature = query_signature in self.query_signature_history if query_signature else False
        if is_repeat or is_same_signature:
            self.repeat_action_count += 1
        else:
            self.repeat_action_count = 0

        if normalized_query:
            self.query_history.append(normalized_query)
            if len(self.query_history) > 8:
                self.query_history.pop(0)
        if query_signature:
            self.query_signature_history.append(query_signature)
            if len(self.query_signature_history) > 8:
                self.query_signature_history.pop(0)

        if self.repeat_action_count >= 3:
            tool_payload = self._materialize_fallback_payload(
                "Observation: Similar query detected. Avoid appending generic filler words to the same query. "
                "You may try one genuinely different relation- or entity-focused reformulation before giving the Final Answer.\n"
                "Thought: Either search with a meaningfully different relation/entity wording, or answer now if the evidence is already sufficient.\n",
                input_ids
            )
            self.repeat_action_count = 0
            self.query_history.clear()
            self.query_signature_history.clear()
        else:
            tool_payload = self._execute_tool(action_buffer, input_ids, big_cache, small_cache)
        return self._commit_tool_payload(tool_payload, input_ids, big_cache, small_cache)

    @torch.no_grad()
    def run_speculative(self, claim, max_gen):
        self.error_count = 0
        self.repeat_action_count = 0
        self.query_history = []
        self.query_signature_history = []
        self._reset_prefetch_state()
        self._reset_trace_state()

        few_shot_examples = (
            "Here are diverse HotpotQA-style multi-hop examples. Use Wikipedia search only when evidence is needed.\n\n"
            "--- Example 1: Comparison / nationality ---\n"
            "Question: Were Scott Derrickson and Ed Wood of the same nationality?\n"
            "Thought: I should verify the nationality of Scott Derrickson first, then compare it with Ed Wood.\n"
            "Action: SearchWikipedia[Scott Derrickson nationality]\n"
            "Observation: Wikipedia search results for 'Scott Derrickson nationality':\n"
            "[1] Title: Scott Derrickson | Evidence: Scott Derrickson is an American director, screenwriter and producer...\n"
            "Thought: Scott Derrickson is American. Now I need Ed Wood's nationality.\n"
            "Action: SearchWikipedia[Ed Wood nationality]\n"
            "Observation: Wikipedia search results for 'Ed Wood nationality':\n"
            "[1] Title: Ed Wood | Evidence: Edward Davis Wood Jr. was an American filmmaker, actor, writer, producer, and director...\n"
            "Thought: Both people are American, so the comparison answer is yes.\n"
            "Final Answer: yes\n\n"
            "--- Example 2: Bridge / film identification ---\n"
            "Question: Which film directed by Scott Derrickson stars Ethan Hawke?\n"
            "Thought: I need the Scott Derrickson film that specifically has Ethan Hawke in the cast.\n"
            "Action: SearchWikipedia[Scott Derrickson Ethan Hawke film]\n"
            "Observation: Wikipedia search results for 'Scott Derrickson Ethan Hawke film':\n"
            "[1] Title: Sinister (film) | Evidence: Sinister is a 2012 supernatural horror film directed by Scott Derrickson. It stars Ethan Hawke...\n"
            "Thought: The evidence directly names the film.\n"
            "Final Answer: Sinister\n\n"
            "--- Example 3: Multi-hop / spouse to birthplace ---\n"
            "Question: In which city was the wife of James Henry Miller born?\n"
            "Thought: I first need to identify James Henry Miller's wife, then find her birthplace city.\n"
            "Action: SearchWikipedia[James Henry Miller wife]\n"
            "Observation: Wikipedia search results for 'James Henry Miller wife':\n"
            "[1] Title: James Henry Miller | Evidence: James Henry Miller married the American writer Isabelle Webb...\n"
            "Thought: The wife is Isabelle Webb. Now I need her birthplace.\n"
            "Action: SearchWikipedia[Isabelle Webb birthplace]\n"
            "Observation: Wikipedia search results for 'Isabelle Webb birthplace':\n"
            "[1] Title: Isabelle Webb | Evidence: Isabelle Webb was born in Boston, Massachusetts, and later became a writer...\n"
            "Thought: The question asks for a city, and the city is Boston.\n"
            "Final Answer: Boston\n\n"
            "--- Example 4: Multi-hop / organization headquarters ---\n"
            "Question: The Oberoi family is part of a hotel company that has a head office in what city?\n"
            "Thought: I need to identify the hotel company tied to the Oberoi family, then find that company's head office city.\n"
            "Action: SearchWikipedia[Oberoi family hotel company]\n"
            "Observation: Wikipedia search results for 'Oberoi family hotel company':\n"
            "[1] Title: Oberoi family | Evidence: The Oberoi family is associated with The Oberoi Group, a luxury hotel company...\n"
            "Thought: The company is The Oberoi Group. Now I need its head office city.\n"
            "Action: SearchWikipedia[The Oberoi Group head office city]\n"
            "Observation: Wikipedia search results for 'The Oberoi Group head office city':\n"
            "[1] Title: The Oberoi Group | Evidence: The Oberoi Group is headquartered in Delhi, India...\n"
            "Thought: The city asked for is Delhi.\n"
            "Final Answer: Delhi\n\n"
            "--- Example 5: Naming relation ---\n"
            "Question: Musician and satirist Allie Goertz wrote a song about Milhouse, and Milhouse was named after whom?\n"
            "Thought: I need the person after whom Milhouse was named.\n"
            "Action: SearchWikipedia[Milhouse named after whom]\n"
            "Observation: Wikipedia search results for 'Milhouse named after whom':\n"
            "[1] Title: Milhouse Van Houten | Evidence: Matt Groening named Milhouse after U.S. president Richard Nixon, whose middle name was Milhous...\n"
            "Thought: The person named in the evidence is Richard Nixon.\n"
            "Final Answer: Richard Nixon\n\n"
            "--- Example 6: Date / founding year ---\n"
            "Question: In what year was the magazine that published Edgar Allan Poe founded?\n"
            "Thought: I need to identify the magazine connected to Edgar Allan Poe, then find its founding year.\n"
            "Action: SearchWikipedia[Edgar Allan Poe magazine founded]\n"
            "Observation: Wikipedia search results for 'Edgar Allan Poe magazine founded':\n"
            "[1] Title: Burton's Gentleman's Magazine | Evidence: Edgar Allan Poe worked for Burton's Gentleman's Magazine...\n"
            "Thought: Now I should search the exact magazine to get the founding year cleanly.\n"
            "Action: SearchWikipedia[Burton's Gentleman's Magazine]\n"
            "Observation: Wikipedia search results for 'Burton's Gentleman's Magazine':\n"
            "[1] Title: Burton's Gentleman's Magazine | Evidence: Burton's Gentleman's Magazine was founded in 1837...\n"
            "Thought: The answer is the year 1837.\n"
            "Final Answer: 1837\n\n"
            "--- Example 7: Place / country ---\n"
            "Question: In which country is the city located where the headquarters of Peugeot is based?\n"
            "Thought: I need the headquarters city of Peugeot, then the country containing that city.\n"
            "Action: SearchWikipedia[Peugeot headquarters city]\n"
            "Observation: Wikipedia search results for 'Peugeot headquarters city':\n"
            "[1] Title: Peugeot | Evidence: Peugeot's headquarters are located in Poissy...\n"
            "Thought: The city is Poissy. Now I need the country of Poissy.\n"
            "Action: SearchWikipedia[Poissy country]\n"
            "Observation: Wikipedia search results for 'Poissy country':\n"
            "[1] Title: Poissy | Evidence: Poissy is a commune in the Yvelines department in France...\n"
            "Thought: The country asked for is France.\n"
            "Final Answer: France\n\n"
            "--- Example 8: Person / parent relation ---\n"
            "Question: Who is the mother of the actor who played Frodo Baggins?\n"
            "Thought: I need the actor who played Frodo Baggins, then I need that actor's mother.\n"
            "Action: SearchWikipedia[Frodo Baggins actor]\n"
            "Observation: Wikipedia search results for 'Frodo Baggins actor':\n"
            "[1] Title: Elijah Wood | Evidence: Elijah Wood portrayed Frodo Baggins in The Lord of the Rings film trilogy...\n"
            "Thought: The actor is Elijah Wood. Now I need his mother.\n"
            "Action: SearchWikipedia[Elijah Wood mother]\n"
            "Observation: Wikipedia search results for 'Elijah Wood mother':\n"
            "[1] Title: Elijah Wood | Evidence: He is the son of Debbie and Warren Wood...\n"
            "Thought: The mother named in the evidence is Debbie Wood.\n"
            "Final Answer: Debbie Wood\n\n"
            "--- Examples End ---\n\n"
        )
        system_content = (
            "You are an elite HotpotQA question-answering engine. Solve with ReAct when needed. "
            "Your goal is to answer each question using only Wikipedia evidence returned by the tool.\n\n"
            "### TASK UNDERSTANDING ###\n"
            "Always identify exactly what the question asks for: person, city, country, year, title, yes/no, organization, or another short answer type.\n"
            "Keep that target answer type explicit during reasoning so you do not return the wrong kind of entity.\n"
            "### DECISION POLICY ###\n"
            "Before every Action, explain in Thought what fact, entity, or relation must be verified next.\n"
            "Use the tool when factual evidence from Wikipedia is needed. Do not invent evidence and do not skip missing hops.\n"
            "If the answer is already directly supported by the current Observation, stop searching and give the final answer.\n"
            "### TOOL RULES ###\n"
            "1. Use the standard ReAct order: Thought -> Action -> Observation.\n"
            "2. Keep each search query concise, literal, and searchable. Prefer entity names plus the needed relation, such as `Ed Wood nationality`.\n"
            "3. Never repeat the exact same search query unless you are deliberately correcting a malformed query.\n"
            "4. If a search fails, change the target entity or relation in a meaningful way instead of appending vague filler words.\n"
            "5. If a page title or evidence is irrelevant, back up and search the core entity name or the missing relation directly.\n"
            "### THOUGHT STYLE ###\n"
            "Each Thought should briefly state: current goal, missing evidence, and immediate next step.\n"
            "Keep Thought short, concrete, and action-guiding.\n"
            "Do not write hidden plans, alternative branches, or long explanations.\n"
            "### QUESTION TYPES TO HANDLE ###\n"
            "You may see comparison questions, bridge questions, multi-hop entity tracing, birthplace questions, parent/spouse relations, headquarters locations, dates/years, naming relations, country/city lookups, and yes/no questions.\n"
            "For comparison or yes/no questions, verify both sides before answering.\n"
            "For bridge questions, first identify the intermediate entity, then query the missing target attribute.\n"
            "For relation questions, search the exact entity plus the needed relation.\n"
            "### SEARCH STRATEGY ###\n"
            "If a Wikipedia search returns irrelevant or weak results, try the exact entity name without extra words first.\n"
            "For example, instead of \"Arthur's Magazine founding year\", search \"Arthur's Magazine\".\n"
            "Only after the exact entity search fails should you conclude the evidence is missing.\n"
            "### OUTPUT RULES ###\n"
            "Use the ReAct format with Thought, optional Action, and Observation.\n"
            "The final answer must be a short answer entity or label, not a sentence or explanation.\n"
            "If the target is a person, film, city, country, date, nationality, or yes/no label, output only that entity or label text.\n"
            "Do not write `The answer is ...` or any explanatory sentence in the final answer.\n"
            "Always end with exactly one line in the format: Final Answer: <short answer>\n"
        )
        user_content = (
            f"{few_shot_examples}"
            f"Question: {claim}\n"
            "Output: Begin with 'Thought: ' and make the next step explicit before any Action.\n"
        )
        self.system_prompt = self._build_prompt_text(system_content, user_content)

        input_ids = self.tokenizer(
            self.system_prompt,
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids.to(self.big_model.device)
        printed_chunks = []
        big_cache, small_cache = DynamicCache(), DynamicCache()

        big_out = self.big_model(input_ids, past_key_values=big_cache, use_cache=True)
        small_out = self.small_model(input_ids.to(self.small_model.device), past_key_values=small_cache, use_cache=True)

        big_logits = big_out.logits[:, -1, :]
        small_logits = small_out.logits[:, -1, :]

        start_time, start_len = time.time(), input_ids.shape[1]
        accepted_count, total_draft = 0, 0
        current_phase = 0
        action_buffer = ""
        current_thought_plan = None
        latest_tool_plan_query = None
        generated_preview = ""
 
        def process_token(t_val, phase, action_buf, thought_plan, latest_plan_query, allow_trigger):
            token_text = self._decode_token(t_val)
            new_phase = self._update_phase(t_val, phase)
            if new_phase == 0 and SEARCH_WIKIPEDIA_PREFIX not in action_buf and len(action_buf) > 256:
                action_buf = action_buf[-256:]
            action_buf = (action_buf + token_text)[-4000:]

            triggered = False

            if SEARCH_WIKIPEDIA_PREFIX in action_buf:
                extracted_query, is_closed = self._extract_search_query(action_buf)
                if extracted_query is not None and is_closed:
                    self.prefetch_stats["action_seen"] += 1
                    self.prefetch_stats["action_closed"] += 1
                    current_action_plan = self._clean_query_candidate(extracted_query)
                    thought_plan = current_action_plan
                    latest_plan_query = self._normalize_query(current_action_plan)
                    if allow_trigger:
                        triggered = True
            return new_phase, action_buf, thought_plan, latest_plan_query, triggered
 
        def maybe_prefetch_from_draft(token_id, draft_phase, draft_action_buffer, draft_thought_plan, draft_latest_plan_query, prefetch_allowed):
            token_text = self._decode_token(token_id)
            draft_phase = self._update_phase(token_id, draft_phase)
            if draft_phase == 0 and SEARCH_WIKIPEDIA_PREFIX not in draft_action_buffer and len(draft_action_buffer) > 256:
                draft_action_buffer = draft_action_buffer[-256:]
            draft_action_buffer = (draft_action_buffer + token_text)[-4000:]

            if not prefetch_allowed or draft_phase != 1:
                return draft_phase, draft_action_buffer, draft_thought_plan, draft_latest_plan_query, False

            if SEARCH_WIKIPEDIA_PREFIX not in draft_action_buffer:
                return draft_phase, draft_action_buffer, draft_thought_plan, draft_latest_plan_query, False

            draft_query, is_closed = self._extract_search_query(draft_action_buffer)
            if draft_query is None:
                return draft_phase, draft_action_buffer, draft_thought_plan, draft_latest_plan_query, False

            cleaned_draft_query = self._clean_query_candidate(draft_query)
            safe_draft_query = self._trim_query_to_safe_prefix(cleaned_draft_query)
            if not safe_draft_query:
                return draft_phase, draft_action_buffer, draft_thought_plan, draft_latest_plan_query, False

            if is_closed or self._is_tool_plan_query_ready(cleaned_draft_query):
                self.prefetch_stats["hidden_probe_seen"] += 1
                self.prefetch_stats["hidden_probe_ready"] += 1
                if safe_draft_query != self.hidden_probe_query:
                    submitted = self._submit_prefetch_candidate(cleaned_draft_query, "hidden_probe")
                    if submitted:
                        draft_thought_plan = cleaned_draft_query
                        draft_latest_plan_query = self._normalize_query(cleaned_draft_query)
                    else:
                        self.prefetch_stats["hidden_probe_prefetch_duplicate"] += 1

            return draft_phase, draft_action_buffer, draft_thought_plan, draft_latest_plan_query, False

        def hidden_probe_then_rollback(base_small_cache, base_small_logits, base_phase, base_action_buffer, base_thought_plan, base_latest_plan_query):
            if not self.enable_hidden_probe or base_phase != 1:
                return
            checkpoint_len = self._cache_seq_len(base_small_cache)
            probe_phase = base_phase
            probe_action_buffer = base_action_buffer
            probe_thought_plan = base_thought_plan
            probe_latest_plan_query = base_latest_plan_query
            probe_logits = base_small_logits

            for step in range(self.hidden_probe_steps):
                probe_logits = self._maybe_boost_action_start(
                    probe_logits,
                    probe_phase,
                    probe_action_buffer,
                    probe_thought_plan,
                    7.5,
                )
                next_probe = torch.argmax(probe_logits, dim=-1, keepdim=True)
                probe_token_id = int(next_probe[0, 0])
                probe_phase, probe_action_buffer, probe_thought_plan, probe_latest_plan_query, _ = maybe_prefetch_from_draft(
                    probe_token_id,
                    probe_phase,
                    probe_action_buffer,
                    probe_thought_plan,
                    probe_latest_plan_query,
                    probe_phase == 1,
                )
                has_tool_signal = SEARCH_WIKIPEDIA_PREFIX in probe_action_buffer
                has_prefetch = self.pending_tool_prefetch is not None
                if has_tool_signal and has_prefetch:
                    break
                if probe_token_id == self.eos_id:
                    break
                pos = torch.tensor([[checkpoint_len + step]], device=next_probe.device)
                probe_out = self.small_model(
                    next_probe,
                    past_key_values=base_small_cache,
                    position_ids=pos,
                    use_cache=True,
                )
                probe_logits = probe_out.logits[:, -1, :]
            self._crop_cache(base_small_cache, checkpoint_len)

        def update_generated_preview(token_id, preview_text, max_chars):
            piece = self._decode_token(token_id)
            preview_text = (preview_text + piece)[-max_chars:]
            return preview_text

        def has_final_answer(preview_text):
            return "Final Answer:" in preview_text or "final answer:" in preview_text

        def should_stop(tid, preview_text):
            if tid == self.eos_id:
                return True
            if tid == self.eot_id:
                return has_final_answer(preview_text)
            return False

        while (input_ids.shape[1] - start_len) < max_gen:
            if not has_final_answer(generated_preview):
                big_logits[0, self.eot_id] = -float('inf')
                big_logits[0, self.eos_id] = -float('inf')

            probs = F.softmax(big_logits, dim=-1)
            entropy = float((-torch.sum(probs * torch.log(probs + 1e-10), dim=-1)).detach().cpu())
            checkpoint_len = big_cache.get_seq_length()

            if entropy > self.threshold:
                big_logits = self._maybe_boost_action_start(big_logits, current_phase, action_buffer, current_thought_plan)
                next_token = torch.argmax(big_logits, dim=-1, keepdim=True)
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                t_val = int(next_token[0, 0])
                current_phase, action_buffer, current_thought_plan, latest_tool_plan_query, tool_triggered = process_token(
                    t_val, current_phase, action_buffer, current_thought_plan, latest_tool_plan_query
                )
                generated_preview = update_generated_preview(t_val, generated_preview)
                token_piece = self._decode_token(t_val)
                printed_chunks.append(token_piece)
                self.committed_output_chunks.append(token_piece)
                self._record_trace(t_val, "big_direct")

                if should_stop(t_val, generated_preview):
                    break

                pos = torch.tensor([[checkpoint_len]], device=input_ids.device)
                big_out = self.big_model(next_token, past_key_values=big_cache, position_ids=pos, use_cache=True)
                small_out = self.small_model(next_token.to(self.small_model.device), past_key_values=small_cache, position_ids=pos, use_cache=True)
                big_logits, small_logits = big_out.logits[:, -1, :], small_out.logits[:, -1, :]

                if tool_triggered:
                    input_ids, big_logits, small_logits, current_phase = self._handle_tool_trigger(action_buffer, input_ids, big_cache, small_cache)
                    action_buffer, current_thought_plan, latest_tool_plan_query = "", None, None
                    continue
            else:
                draft_tokens = []
                draft_phase = current_phase
                draft_action_buffer = action_buffer
                draft_thought_plan = current_thought_plan
                draft_latest_plan_query = latest_tool_plan_query
 
                if not has_final_answer(generated_preview):
                    small_logits[0, self.eot_id] = -float('inf')
                    small_logits[0, self.eos_id] = -float('inf')
 
                hidden_probe_allowed = current_phase == 1
                hidden_probe_then_rollback(
                    small_cache,
                    small_logits.clone(),
                    current_phase,
                    action_buffer,
                    current_thought_plan,
                    latest_tool_plan_query,
                )
                small_logits = self._maybe_boost_action_start(small_logits, current_phase, action_buffer, current_thought_plan, 7.5)
                temp_input = torch.argmax(small_logits, dim=-1, keepdim=True)
                for i in range(self.lookahead):
                    temp_token_id = int(temp_input[0, 0])
                    if temp_token_id == self.nl_id or temp_token_id == self.eot_id:
                        boost = 10.0
                        if current_phase == 0:
                            small_logits[0, self.base_action_id] += boost
                        elif current_phase == 1:
                            small_logits[0, self.base_obs_id] += boost
                        elif current_phase == 3:
                            small_logits[0, self.base_action_id] += boost
                        if not has_final_answer(generated_preview):
                            small_logits[0, self.eot_id] = -float('inf')
                        temp_input = torch.argmax(small_logits, dim=-1, keepdim=True)

                    draft_tokens.append(temp_input)
                    draft_phase, draft_action_buffer, draft_thought_plan, draft_latest_plan_query, _ = maybe_prefetch_from_draft(
                        temp_token_id,
                        draft_phase,
                        draft_action_buffer,
                        draft_thought_plan,
                        draft_latest_plan_query,
                        hidden_probe_allowed,
                    )
                    if temp_token_id == self.eos_id:
                        break
                    if temp_token_id == self.eot_id and has_final_answer(generated_preview):
                        break

                    p_id = torch.tensor([[checkpoint_len + i]], device=temp_input.device)
                    s_out = self.small_model(temp_input, past_key_values=small_cache, position_ids=p_id, use_cache=True)
                    small_logits = s_out.logits[:, -1, :]
                    if not has_final_answer(generated_preview):
                        small_logits[0, self.eot_id] = -float('inf')
                    temp_input = torch.argmax(small_logits, dim=-1, keepdim=True)

                draft_seq = torch.cat(draft_tokens, dim=-1).to(self.big_model.device)
                actual_draft_len = draft_seq.shape[1]
                total_draft += actual_draft_len
                v_pos = torch.arange(checkpoint_len, checkpoint_len + actual_draft_len, device=input_ids.device).unsqueeze(0)
                verify_out = self.big_model(draft_seq, past_key_values=big_cache, position_ids=v_pos, use_cache=True)
                verify_logits = verify_out.logits

                n_matches = 0
                tool_triggered = False
                for i in range(actual_draft_len):
                    target_logits = big_logits if i == 0 else verify_logits[:, i - 1, :]
                    correct_token = torch.argmax(target_logits, dim=-1, keepdim=True)
                    correct_token_id = int(correct_token[0, 0])
                    draft_token_id = int(draft_tokens[i][0, 0])
                    if correct_token_id == draft_token_id:
                        n_matches += 1
                        accepted_count += 1
                        input_ids = torch.cat([input_ids, correct_token], dim=-1)
                        t_val = correct_token_id
                        current_phase, action_buffer, current_thought_plan, latest_tool_plan_query, tool_triggered = process_token(
                            t_val, current_phase, action_buffer, current_thought_plan, latest_tool_plan_query
                        )
                        generated_preview = update_generated_preview(t_val, generated_preview)
                        token_piece = self._decode_token(t_val)
                        printed_chunks.append(token_piece)
                        self.committed_output_chunks.append(token_piece)
                        self._record_trace(t_val, "small_draft")
                        if should_stop(t_val, generated_preview) or tool_triggered:
                            break
                    else:
                        break

                new_len = checkpoint_len + n_matches
                self._crop_cache(big_cache, new_len)
                self._crop_cache(small_cache, new_len)
                if should_stop(int(input_ids[0, -1]), generated_preview):
                    break
                if tool_triggered:
                    input_ids, big_logits, small_logits, current_phase = self._handle_tool_trigger(action_buffer, input_ids, big_cache, small_cache)
                    action_buffer, current_thought_plan, latest_tool_plan_query = "", None, None
                    continue

                f_logits = big_logits.clone() if n_matches == 0 else verify_logits[:, n_matches - 1, :].clone()
                if not has_final_answer(generated_preview):
                    f_logits[0, self.eot_id] = -float('inf')
                    f_logits[0, self.eos_id] = -float('inf')

                if n_matches < actual_draft_len:
                    rejected_id = int(draft_tokens[n_matches][0, 0])
                    f_logits[0, rejected_id] = -float('inf')
                    next_resample_id = int(torch.argmax(f_logits, dim=-1)[0])
                    if next_resample_id == self.nl_id or next_resample_id == self.eot_id:
                        res_alpha = 3.0
                        if current_phase == 0:
                            f_logits[0, self.base_action_id] += res_alpha
                        elif current_phase == 1:
                            f_logits[0, self.base_obs_id] += res_alpha
                        elif current_phase == 3:
                            f_logits[0, self.base_action_id] += res_alpha

                final_correct = torch.argmax(f_logits, dim=-1, keepdim=True)
                input_ids = torch.cat([input_ids, final_correct], dim=-1)
                t_val = int(final_correct[0, 0])
                current_phase, action_buffer, current_thought_plan, latest_tool_plan_query, tool_triggered = process_token(
                    t_val, current_phase, action_buffer, current_thought_plan, latest_tool_plan_query
                )
                generated_preview = update_generated_preview(t_val, generated_preview)
                token_piece = self._decode_token(t_val)
                printed_chunks.append(token_piece)
                self.committed_output_chunks.append(token_piece)
                self._record_trace(t_val, "big_resample")

                if should_stop(t_val, generated_preview):
                    break
                sync_pos = torch.tensor([[new_len]], device=input_ids.device)
                big_out = self.big_model(final_correct, past_key_values=big_cache, position_ids=sync_pos, use_cache=True)
                small_out = self.small_model(final_correct.to(self.small_model.device), past_key_values=small_cache, position_ids=sync_pos, use_cache=True)
                big_logits, small_logits = big_out.logits[:, -1, :], small_out.logits[:, -1, :]

                if tool_triggered:
                    input_ids, big_logits, small_logits, current_phase = self._handle_tool_trigger(action_buffer, input_ids, big_cache, small_cache)
                    action_buffer, current_thought_plan, latest_tool_plan_query = "", None, None
                    continue

        if printed_chunks:
            print("".join(printed_chunks), end="", flush=True)
        dur = time.time() - start_time
        gen_text = "".join(self.committed_output_chunks).strip()
        total_tokens = input_ids.shape[1] - start_len
        
        # Count reasoning steps as a key ReAct/ReWoo comparison metric
        step_count = gen_text.lower().count("thought:")
        action_count = gen_text.lower().count("action:")
        committed_observation_tokens = sum(
            len(item.get("obs_ids", []))
            for item in getattr(self, "tool_usage_trace", [])
        )
        raw_observation_tokens = sum(
            item.get("raw_observation_token_count", len(item.get("obs_ids", [])))
            for item in getattr(self, "tool_usage_trace", [])
        )
        latency_sec = dur
        committed_tps = total_tokens / dur if dur > 0 else 0.0
        accept_rate = accepted_count / total_draft if total_draft > 0 else 0
        
        # Count LLM-generated tokens only, excluding injected observations
        llm_tokens = total_tokens - committed_observation_tokens
        llm_tps = llm_tokens / dur if dur > 0 else 0.0
        submitted = self.prefetch_stats["submitted"]
        hits = self.prefetch_stats["hit"]
        opportunities = self.prefetch_stats["prefetch_opportunities"]
        self.last_run_stats = {
            "ste_enabled": self.enable_ste,
            "ste_submitted": submitted,
            "ste_hit": hits,
            "ste_miss": self.prefetch_stats["miss"],
            "ste_timeout": self.prefetch_stats["timeout"],
            "ste_stale": self.prefetch_stats["stale"],
            "ste_cancelled": self.prefetch_stats["cancelled"],
            "ste_reused": self.prefetch_stats["reused"],
            "ste_skipped_duplicate": self.prefetch_stats["skipped_duplicate"],
            "ste_saved_wait_ms": round(self.prefetch_stats["saved_wait_ms"], 2),
            "ste_prefetch_opportunities": opportunities,
            "ste_prefetch_effective_hits": self.prefetch_stats["prefetch_effective_hits"],
            "ste_prefetch_wait_time_ms": round(self.prefetch_stats["prefetch_wait_time_ms"], 2),
            "ste_prefetch_latency_ms": round(self.prefetch_stats["prefetch_latency_ms"], 2),
            "ste_hit_rate_by_submission": round(hits / max(1, submitted), 4),
            "ste_hit_rate_by_opportunity": round(hits / max(1, opportunities), 4),
            "ste_effective_hit_rate": round(self.prefetch_stats["prefetch_effective_hits"] / max(1, opportunities), 4),
            "compute_submitted": self.prefetch_stats["compute_submitted"],
            "compute_hit": self.prefetch_stats["compute_hit"],
            "compute_reject_discard": self.prefetch_stats["compute_reject_discard"],
            "compute_sync_fallback": self.prefetch_stats["compute_sync_fallback"],
            "shadow_kv_committed": self.prefetch_stats["shadow_kv_committed"],
            "sync_tool_calls": self.prefetch_stats["sync_tool_calls"],
            "prefetch_tool_calls": self.prefetch_stats["prefetch_tool_calls"],
            "hidden_probe_seen": self.prefetch_stats["hidden_probe_seen"],
            "hidden_probe_ready": self.prefetch_stats["hidden_probe_ready"],
            "hidden_probe_prefetch_triggered": self.prefetch_stats["hidden_probe_prefetch_triggered"],
            "hidden_probe_prefetch_duplicate": self.prefetch_stats["hidden_probe_prefetch_duplicate"],
            "action_seen": self.prefetch_stats["action_seen"],
            "action_closed": self.prefetch_stats["action_closed"],
            "big_direct_tokens": self.trace_stats["big_direct_tokens"],
            "small_drafted_tokens": self.trace_stats["small_drafted_tokens"],
            "accepted_draft_tokens": self.trace_stats["accepted_draft_tokens"],
            "big_resample_tokens": self.trace_stats["big_resample_tokens"],
            "latency_sec": round(latency_sec, 4),
            "committed_tokens": total_tokens,
            "committed_tps": round(committed_tps, 2),
            "llm_tokens": llm_tokens,
            "llm_tps": round(llm_tps, 2),
            "committed_observation_tokens": committed_observation_tokens,
            "raw_observation_tokens": raw_observation_tokens,
            "accept_rate": round(accept_rate, 4),
            "step_count": step_count,
            "action_count": action_count,
            "total_tool_time_sec": round(self.total_tool_time_sec, 4),
            "generation_trace": self._render_trace_summary(),
        }
        return dur, total_tokens, accept_rate, gen_text


# ==========================================
# Dataset loading logic
# ==========================================
def load_fever_dataset(dataset_path: str):
    print(f">>> Loading HotpotQA dataset: {dataset_path}")
    if dataset_path.endswith(".jsonl"):
        fever_rows = []
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                fever_rows.append(json.loads(line))
        full_dataset = Dataset.from_list(fever_rows)
    elif dataset_path.endswith(".arrow"):
        try:
            full_dataset = Dataset.from_file(dataset_path)
        except Exception as e:
            raise RuntimeError(
                f"Unable to read Arrow file directly: {dataset_path}\n"
                f"Please confirm that datasets/pyarrow is installed and that the file is a HuggingFace datasets Arrow shard.\n"
                f"Original error: {e}"
            )
    else:
        try:
            loaded = load_dataset(dataset_path)
            if isinstance(loaded, dict):
                split_name = next(iter(loaded.keys()))
                full_dataset = loaded[split_name]
            else:
                full_dataset = loaded
        except Exception as e:
            raise ValueError(f"Unsupported dataset_path format: {dataset_path}\nOriginal error: {e}")

    print(f">>> Dataset loaded successfully with {len(full_dataset)} rows. Columns: {list(full_dataset.column_names)}")
    return full_dataset


CLAIM_FIELD_CANDIDATES = ["question", "text", "claim", "sentence", "query", "sentence1", "premise", "hypothesis"]
LABEL_FIELD_CANDIDATES = ["answer", "label", "labels", "gold_label", "target"]


def extract_claim_and_label(entry: Dict[str, Any]):
    claim_value = get_first_available(entry, CLAIM_FIELD_CANDIDATES, None)
    label_value = get_first_available(entry, LABEL_FIELD_CANDIDATES, None)

    if isinstance(label_value, list):
        label_value = label_value[0] if label_value else ""
    elif isinstance(label_value, dict):
        label_value = label_value.get("text", label_value.get("answer", ""))

    claim = "" if claim_value is None else str(claim_value).strip()
    ref_label = "" if label_value is None else str(label_value).strip()
    return claim, ref_label


# ==========================================
# Main evaluation entry
# ==========================================
def main(config):
    big_model_path = config["big_model_path"]
    small_model_path = config["small_model_path"]
    dataset_path = config["dataset_path"]
    wiki_data_path = config["wiki_data_path"]
    threshold = config["threshold"]
    lookahead = config["lookahead"]
    enable_ste = config["enable_ste"]
    ste_max_workers = config["ste_max_workers"]
    ste_min_query_len = config["ste_min_query_len"]
    ste_wait_timeout = config["ste_wait_timeout"]
    ste_ttl_sec = config["ste_ttl_sec"]
    observation_topk_docs = config["observation_topk_docs"]
    observation_topk_paragraphs = config["observation_topk_paragraphs"]
    observation_max_words = config["observation_max_words"]
    observation_max_chars = config["observation_max_chars"]
    enable_shadow_kv = config["enable_shadow_kv"]
    enable_hidden_probe = config["enable_hidden_probe"]
    hidden_probe_steps = config["hidden_probe_steps"]
    num_test = config["num_test"]
    engine = ReActSpeculativeEngine(
        big_model_path,
        small_model_path,
        wiki_data_path,
        threshold,
        lookahead,
        enable_ste,
        ste_max_workers,
        ste_min_query_len,
        ste_wait_timeout,
        ste_ttl_sec,
        observation_topk_docs,
        observation_topk_paragraphs,
        observation_max_words,
        observation_max_chars,
        enable_shadow_kv,
        enable_hidden_probe,
        hidden_probe_steps,
    )

    try:
        full_dataset = load_fever_dataset(dataset_path)
        num_test = min(int(num_test), len(full_dataset))
        test_data = full_dataset.select(range(num_test))

        results = []
        correct_count, valid_count = 0, 0

        for i, entry in enumerate(test_data):
            claim, ref_label = extract_claim_and_label(entry)
            print(f"\n\n{'=' * 30} [Case {i + 1}/{num_test}] {'=' * 30}")
            print(f"Original fields: {list(entry.keys())}")
            if not claim:
                print("[Skipped] Unable to parse text")
                continue

            print(f"Question: {claim[:180]}...")
            print(f"Reference Answer: {ref_label if ref_label else 'N/A'}")

            try:
                dur, total_tokens, acc_rate, generated_text = engine.run_speculative(claim, 2048)
                pred_label = extract_final_answer(generated_text)
                is_hit = bool(ref_label) and exact_match_score(pred_label, ref_label) == 1.0
                f1 = f1_score(pred_label, ref_label) if ref_label else 0.0
                if ref_label:
                    valid_count += 1
                    if is_hit:
                        correct_count += 1

                print("\n[Full Output Trace]\n" + generated_text)
                print(f"\n[Evaluation] Predicted answer: {pred_label} | Reference answer: {ref_label if ref_label else 'N/A'} | EM: {'✅' if is_hit else '❌'} | F1: {f1:.3f}")
                if engine.last_run_stats:
                    print(
                        "[Reasoning Stats] "
                        f"latency={engine.last_run_stats['latency_sec']:.2f}s | "
                        f"committed_tps={engine.last_run_stats['committed_tps']} | "
                        f"committed_tokens={engine.last_run_stats['committed_tokens']} | "
                        f"committed_obs_tokens={engine.last_run_stats['committed_observation_tokens']} | "
                        f"raw_obs_tokens={engine.last_run_stats['raw_observation_tokens']} | "
                        f"accept_rate={engine.last_run_stats['accept_rate']:.2%} | "
                        f"tool_time={engine.last_run_stats['total_tool_time_sec']:.2f}s | "
                        f"big_direct={engine.last_run_stats['big_direct_tokens']} | "
                        f"small_accept={engine.last_run_stats['accepted_draft_tokens']} | "
                        f"big_resample={engine.last_run_stats['big_resample_tokens']}"
                    )
                    print(
                        "[STE Stats] "
                        f"submitted={engine.last_run_stats['ste_submitted']} | "
                        f"hit={engine.last_run_stats['ste_hit']} | "
                        f"opportunities={engine.last_run_stats['ste_prefetch_opportunities']} | "
                        f"timeout={engine.last_run_stats['ste_timeout']} | "
                        f"saved_wait_ms={engine.last_run_stats['ste_saved_wait_ms']} | "
                        f"hit_rate/submitted={engine.last_run_stats['ste_hit_rate_by_submission']:.2%} | "
                        f"hit_rate/opportunity={engine.last_run_stats['ste_hit_rate_by_opportunity']:.2%}"
                    )
                    print(
                        "[Hidden Probe Stats] "
                        f"seen={engine.last_run_stats['hidden_probe_seen']} | "
                        f"ready={engine.last_run_stats['hidden_probe_ready']} | "
                        f"triggered={engine.last_run_stats['hidden_probe_prefetch_triggered']} | "
                        f"duplicate={engine.last_run_stats['hidden_probe_prefetch_duplicate']}"
                    )
                    print(
                        "[Action Stats] "
                        f"closed_actions={engine.last_run_stats['action_closed']} | "
                        f"action_seen={engine.last_run_stats['action_seen']}"
                    )
                    print(engine.last_run_stats["generation_trace"])

                res_entry = {
                    "id": i,
                    "question": claim,
                    "ref_answer": ref_label,
                    "pred_answer": pred_label,
                    "exact_match": is_hit,
                    "f1_score": round(f1, 4),
                    "latency_sec": round(dur, 4),
                    "committed_tokens": total_tokens,
                    "committed_tps": round(engine.last_run_stats.get("committed_tps", 0.0), 2),
                    "accept_rate": round(acc_rate, 4),
                    "total_tool_time_sec": round(engine.last_run_stats.get("total_tool_time_sec", 0.0), 4),
                    "step_count": engine.last_run_stats.get("step_count", 0),
                    "action_count": engine.last_run_stats.get("action_count", 0),
                    "raw_output": generated_text,
                    "available_fields": list(entry.keys()),
                    **engine.last_run_stats,
                }
                results.append(res_entry)
            except Exception as e:
                print(f"Error case {i}: {e}")
                continue

        if results:
            df = pd.DataFrame(results)
            df.to_csv(engine.log_csv, index=False, encoding='utf-8-sig')
            print(
                f"\nEvaluation completed. Average latency: {df['latency_sec'].mean():.2f}s, "
                f"Median latency: {df['latency_sec'].median():.2f}s, "
                f"P90 latency: {df['latency_sec'].quantile(0.9):.2f}s, "
                f"Committed TPS: {df['committed_tps'].mean():.2f}, "
                f"Acceptance rate: {df['accept_rate'].mean():.2%}, "
                f"Total tool time: {df['total_tool_time_sec'].mean():.2f}s"
            )
            if 'f1_score' in df.columns and 'exact_match' in df.columns:
                print(
                    f"[Accuracy] EM: {df['exact_match'].mean():.2%} | "
                    f"F1: {df['f1_score'].mean():.3f}"
                )
            if 'step_count' in df.columns and 'action_count' in df.columns:
                print(
                    f"[Reasoning Steps] Average steps: {df['step_count'].mean():.2f} | "
                    f"Median steps: {df['step_count'].median():.0f} | "
                    f"Average Action count: {df['action_count'].mean():.2f}"
                )
            if 'llm_tps' in df.columns:
                print(
                    f"[LLM Efficiency] LLM TPS: {df['llm_tps'].mean():.2f} | "
                    f"LLM tokens/sample: {df['llm_tokens'].mean():.0f}"
                )
            if 'ste_saved_wait_ms' in df.columns:
                print(
                    f"Average STE saved wait: {df['ste_saved_wait_ms'].mean():.2f} ms | "
                    f"STE submission hit rate: {df['ste_hit'].sum() / max(1, df['ste_submitted'].sum()):.2%} | "
                    f"STE opportunity hit rate: {df['ste_hit'].sum() / max(1, df['ste_prefetch_opportunities'].sum()):.2%}"
                )
            if 'hidden_probe_prefetch_triggered' in df.columns:
                print(
                    f"Hidden Probe trigger rate: {df['hidden_probe_prefetch_triggered'].sum() / max(1, len(df)):.2f} | "
                    f"Hidden Probe duplicate suppression count: {df['hidden_probe_prefetch_duplicate'].sum()}"
                )
            if valid_count > 0:
                print(f"Final EM: {correct_count / valid_count:.2%}")
                if 'f1_score' in df.columns:
                    print(f"Final F1: {df['f1_score'].mean():.3f}")
                if 'step_count' in df.columns:
                    print(f"Average reasoning steps: {df['step_count'].mean():.2f}")
    finally:
        engine.close()


if __name__ == "__main__":
    runtime_config = json.loads(sys.argv[1])
    main(runtime_config)
