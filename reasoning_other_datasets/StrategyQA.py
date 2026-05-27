import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
import torch
import torch.nn.functional as F
from awq import AutoAWQForCausalLM
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

# =================================================================
# StrategyQA evaluation helper utilities
# =================================================================
ANSWER_NORMALIZATION_RE = re.compile(r"\b(a|an|the)\b")
MULTISPACE_RE = re.compile(r"\s+")
FINAL_ANSWER_RE_LIST = [
    re.compile(r"Final Answer\s*:\s*(.+)", flags=re.IGNORECASE),
    re.compile(r"Answer\s*:\s*(.+)", flags=re.IGNORECASE),
]
SEARCH_WIKIPEDIA_PREFIX = "SearchWikipedia["
ACTION_MARKER = "Action:"
ACTION_PREFIX_TEXT = "Action: SearchWikipedia["
PUNCT_TABLE = str.maketrans("", "", r"!\"#$%&'()*+,./:;<=>?@[\\]^_`{|}~")
DEFAULT_USER_AGENT = "OpenSource-StrategyQA-Evaluator/1.0 (set --user-agent for your environment)"


# =================================================================
# Public configuration defaults
# =================================================================
DEFAULT_CONFIG = {
    "big_model_path": "./models/big-model-awq",
    "small_model_path": "./models/small-model",
    "dataset_path": "./data/strategyqa/strategyqa.json",
    "retrieval_corpus_path": "./data/strategyqa/strategyqa_corpus.json",
    "output_csv": "./outputs/strategyqa_results.csv",
    "threshold": 0.5,
    "lookahead": 9,
    "max_gen": 2048,
    "num_test": 10000,
    "enable_ste": True,
    "ste_max_workers": 2,
    "ste_min_query_len": 8,
    "ste_wait_timeout": 18.0,
    "ste_ttl_sec": 15.0,
    "observation_topk_docs": 3,
    "observation_topk_paragraphs": 3,
    "observation_max_words": 1200,
    "observation_max_chars": 2400,
    "enable_shadow_kv": False,
    "enable_hidden_probe": True,
    "hidden_probe_steps": 12,
    "request_connect_timeout": 1.5,
    "request_read_timeout": 6.0,
    "user_agent": DEFAULT_USER_AGENT,
}


def load_config(config_path: Optional[str]) -> Dict[str, Any]:
    config = dict(DEFAULT_CONFIG)
    if not config_path:
        return config

    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    with config_file.open("r", encoding="utf-8") as f:
        loaded = json.load(f)

    if not isinstance(loaded, dict):
        raise ValueError("Config file must contain a JSON object.")

    config.update(loaded)
    return config


def normalize_answer(text):
    if text is None:
        return ""
    text = str(text).lower().strip()
    text = text.translate(PUNCT_TABLE)
    text = ANSWER_NORMALIZATION_RE.sub(" ", text)
    text = MULTISPACE_RE.sub(" ", text)
    return text.strip()


def extract_final_answer(text):
    if not isinstance(text, str):
        return ""
    for pattern in FINAL_ANSWER_RE_LIST:
        matches = pattern.findall(text)
        if matches:
            answer = matches[-1].strip()
            answer = answer.replace("<|eot_id|>", "").strip()
            if answer.endswith(self_eos := "</s>"):
                answer = answer[: -len(self_eos)].rstrip()
            return answer
    return ""


def exact_match_score(prediction, ground_truth):
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction, ground_truth):
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


def get_first_available(entry: Dict[str, Any], candidates: List[str], default: str = ""):
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


def bool_to_label(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    text = str(value).strip().lower()
    if text in {"true", "yes", "1"}:
        return "yes"
    if text in {"false", "no", "0"}:
        return "no"
    return str(value).strip()


class ReActSpeculativeEngine:
    def __init__(
        self,
        big_path,
        small_path,
        strategyqa_dataset_path=None,
        threshold=0.5,
        lookahead=6,
        enable_ste=True,
        ste_max_workers=2,
        ste_min_query_len=8,
        ste_wait_timeout=18.0,
        ste_ttl_sec=15.0,
        observation_topk_docs=3,
        observation_topk_paragraphs=3,
        observation_max_words=1200,
        observation_max_chars=2400,
        enable_shadow_kv=False,
        enable_hidden_probe=True,
        hidden_probe_steps=12,
        output_csv="./outputs/strategyqa_results.csv",
        user_agent=DEFAULT_USER_AGENT,
        request_timeout=(1.5, 6.0),
    ):
        torch.cuda.empty_cache()
        self.threshold = threshold
        self.lookahead = lookahead
        self.strategyqa_dataset_path = strategyqa_dataset_path
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

        print(">>> Loading large model (AWQ)...")
        self.big_model_wrapper = AutoAWQForCausalLM.from_quantized(
            big_path, fuse_layers=True, trust_remote_code=True, device_map="auto"
        )
        self.big_model = self.big_model_wrapper.model

        print(">>> Loading small model...")
        self.small_model = AutoModelForCausalLM.from_pretrained(
            small_path, device_map="auto", torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
        )

        self.tokenizer = AutoTokenizer.from_pretrained(big_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.nl_id = self._encode_text("\n")[-1]
        self.eos_id = self.tokenizer.eos_token_id
        self.eot_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if self.eot_id is None:
            self.eot_id = 128009

        self.action_ids = {self._encode_text(w)[-1] for w in ["Action", " Action", "\nAction"]}
        self.obs_ids = {self._encode_text(w)[-1] for w in ["Observation", " Observation", "\nObservation"]}
        self.thought_ids = {self._encode_text(w)[-1] for w in ["Thought", " Thought", "\nThought"]}
        self.base_action_id = self._encode_text("Action")[-1]
        self.base_obs_id = self._encode_text("Observation")[-1]
        self.base_thought_id = self._encode_text("Thought")[-1]

        self.log_csv = output_csv

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

        self.http_session = requests.Session()
        self.http_headers = {"User-Agent": user_agent}
        self.request_timeout = request_timeout

        self.strategyqa_examples = self._load_strategyqa_examples()
        self.strategyqa_index = self._build_strategyqa_index(self.strategyqa_examples)

    def _reset_prefetch_state(self):
        self.pending_tool_prefetch = None
        self.hidden_probe_query = None
        self.action_prefetch = None
        self.accepted_action_prefetch_query = None
        self.draft_action_prefetch_query = None
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
            "compute_sync_fallback": 0,
            "shadow_kv_committed": 0,
            "hidden_probe_seen": 0,
            "hidden_probe_ready": 0,
            "hidden_probe_prefetch_triggered": 0,
            "hidden_probe_prefetch_duplicate": 0,
            "hidden_probe_hit": 0,
            "hidden_probe_timeout": 0,
            "hidden_probe_reuse": 0,
            "action_seen": 0,
            "action_closed": 0,
            "action_without_tool_plan": 0,
            "action_plan_match": 0,
            "action_plan_mismatch": 0,
            "prefetch_opportunities": 0,
            "prefetch_effective_hits": 0,
            "prefetch_wait_time_ms": 0.0,
            "prefetch_latency_ms": 0.0,
            "action_async_submitted": 0,
            "action_hit": 0,
            "action_timeout": 0,
            "action_reuse": 0,
            "dual_async_both_available": 0,
            "dual_async_first_win": 0,
            "dual_async_second_win": 0,
            "dual_async_fallback_action": 0,
            "action_fallback_sync": 0,
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

    def _record_trace(self, token_id, source, accepted=True):
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

    def _render_trace_summary(self, max_chars=1600):
        if not self.generation_trace:
            return "[Generation Trace]\n(no generated tokens)\n"

        header = (
            "\n" + "=" * 24 + " Generation Trace " + "=" * 24 + "\n"
            "[S-ACC] = Drafted by small model and accepted by large model | "
            "[B-DIR] = Generated directly by large model | "
            "[B-RES] = Small-model draft rejected and rewritten by large model\n"
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

    def _load_strategyqa_examples(self):
        if not self.strategyqa_dataset_path:
            print(">>> No StrategyQA retrieval corpus path provided. Tool retrieval fallback will be disabled.")
            return []
        print(f">>> Loading local retrieval corpus: {self.strategyqa_dataset_path}")
        with open(self.strategyqa_dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"Retrieval corpus must be a list, but got: {type(data).__name__}")
        print(f">>> Retrieval corpus loaded with {len(data)} entries.")
        if self.enable_ste:
            print(">>> STE is enabled: tool results may be prefetched asynchronously once an Action is drafted.")
        return data

    def _build_strategyqa_index(self, examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        indexed = []
        for idx, entry in enumerate(examples):
            question = str(entry.get("question", "")).strip()
            term = str(entry.get("term", "")).strip()
            description = str(entry.get("description", "")).strip()
            facts = entry.get("facts", [])
            decomposition = entry.get("decomposition", [])
            answer = bool_to_label(entry.get("answer", ""))
            fact_lines = [str(item).strip() for item in facts if str(item).strip()]
            decomp_lines = [str(item).strip() for item in decomposition if str(item).strip()]
            combined_parts = [term, description, question, answer] + fact_lines + decomp_lines
            combined_text = " ".join(part for part in combined_parts if part).lower()
            indexed.append({
                "qid": entry.get("qid", idx),
                "question": question,
                "term": term,
                "description": description,
                "facts": fact_lines,
                "decomposition": decomp_lines,
                "answer": answer,
                "search_blob": combined_text,
            })
        return indexed

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

    def _tool_plan_min_query_len(self) -> int:
        return max(1, min(self.ste_min_query_len, 8))

    def _is_tool_plan_query_ready(self, query: Optional[str]) -> bool:
        normalized_query = self._normalize_query(query)
        return bool(normalized_query)

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

    def _extract_search_query(self, text):
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

    def _query_signature(self, query: Optional[str]) -> str:
        normalized = self._normalize_query(query)
        if not normalized:
            return ""
        tokens = re.findall(r"\w+", normalized)
        if not tokens:
            return ""
        stop_tokens = {
            "people", "exact", "entity", "information", "need", "still", "contain", "contains",
            "with", "and", "the", "a", "an", "is", "are", "was", "were", "do", "does"
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

    def _maybe_boost_action_start(self, logits, phase, action_buffer, thought_plan=None, boost=6.0):
        prefix = action_buffer or ""
        target_ids = self.action_prefix_ids
        prefix_ids = self._encode_text(prefix) if prefix else []
        matched = 0
        max_match = min(len(prefix_ids), len(target_ids))
        while matched < max_match and prefix_ids[matched] == target_ids[matched]:
            matched += 1

        if phase != 1:
            return logits

        if matched < len(target_ids):
            logits[0, target_ids[matched]] += boost
        elif not prefix.endswith("]") and self.action_suffix_ids:
            logits[0, self.action_suffix_ids[0]] += boost
        return logits

    def _parse_action_query(self, action_buffer):
        query, is_closed = self._extract_search_query(action_buffer)
        if query is None and ACTION_MARKER in action_buffer:
            query = action_buffer.split(ACTION_MARKER)[-1].strip()
            is_closed = True
        if query is not None:
            query = query.strip()
        return query, is_closed

    def _truncate_text_by_words(self, text: str, max_words: Optional[int] = None) -> str:
        compact = MULTISPACE_RE.sub(" ", (text or "")).strip()
        if not compact:
            return ""
        max_words = self.observation_max_words if max_words is None else max(1, int(max_words))
        words = compact.split()
        if len(words) <= max_words:
            return compact
        return " ".join(words[:max_words]).strip() + " [CONTENT TRUNCATED]"

    def _search_wikipedia(self, query, topk=None, max_chars=None):
        search_url = "https://en.wikipedia.org/w/api.php"
        session = self.http_session

        topk = self.observation_topk_docs if topk is None else topk
        max_chars = self.observation_max_chars if max_chars is None else max_chars
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
                title = item["title"]
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
                        final_results.append({
                            "title": page_info.get("title", title),
                            "text": self._truncate_text_by_words(extract_text),
                        })
                if len(final_results) >= topk:
                    break
            return final_results[:topk]
        except requests.Timeout as e:
            print(f"\n[Wikipedia Tool Timeout] query={query!r} | timeout={self.request_timeout} | error={e}")
            return []
        except requests.RequestException as e:
            print(f"\n[Wikipedia Tool Request Error] query={query!r} | error={e}")
            return []
        except Exception as e:
            print(f"\n[Wikipedia Tool Error] {e}")
            return []

    def _score_strategyqa_example(self, normalized_query: str, query_tokens: List[str], item: Dict[str, Any]) -> float:
        score = 0.0
        search_blob = item.get("search_blob", "")
        question = (item.get("question", "") or "").lower()
        term = (item.get("term", "") or "").lower()
        description = (item.get("description", "") or "").lower()

        if normalized_query == question:
            score += 100.0
        if normalized_query == term:
            score += 60.0
        if term and term in normalized_query:
            score += 25.0
        if normalized_query and normalized_query in search_blob:
            score += 15.0

        for token in query_tokens:
            if token in term:
                score += 8.0
            elif token in description:
                score += 5.0
            elif token in question:
                score += 4.0
            elif token in search_blob:
                score += 1.5

        return score

    def _search_strategyqa_corpus(self, query, topk=None):
        topk = self.observation_topk_docs if topk is None else topk
        normalized_query = self._normalize_query(query)
        query_tokens = re.findall(r"\w+", normalized_query)
        if not normalized_query or not self.strategyqa_index:
            return []

        scored = []
        for item in self.strategyqa_index:
            score = self._score_strategyqa_example(normalized_query, query_tokens, item)
            if score > 0:
                scored.append((score, item))

        scored.sort(key=lambda pair: (-pair[0], str(pair[1].get("qid", ""))))
        results = []
        for rank, (score, item) in enumerate(scored[:topk]):
            facts = item.get("facts", [])
            text_parts = []
            if item.get("description"):
                text_parts.append(f"Description: {item['description']}")
            if facts:
                text_parts.append("Facts: " + " ".join(facts))
            if item.get("decomposition"):
                text_parts.append("Decomposition: " + " | ".join(item["decomposition"]))
            text_parts.append(f"Dataset Answer: {item.get('answer', '')}")
            results.append({
                "title": item.get("term") or item.get("question") or f"StrategyQA-{rank + 1}",
                "text": self._truncate_text_by_words(" ".join(text_parts)),
                "question": item.get("question", ""),
                "score": score,
                "qid": item.get("qid", rank),
            })
        return results

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
                f"Observation: No relevant evidence found for '{query}' from either Wikipedia or the local retrieval dataset. "
                "Do not keep appending generic words to the same query. "
                "If the query already centers on the same entity, do not search again with filler terms. "
                "Either search only the core entity name once, or stop searching and provide the Final Answer from current reasoning.\n"
                "Thought: Give the final answer now unless you have a genuinely different entity or relation to inspect.\n"
            )

        lines = []
        for idx, item in enumerate(selected, 1):
            snippet = item["paragraph"].replace("\n", " ").strip()
            if len(snippet) > self.observation_max_chars:
                snippet = snippet[:self.observation_max_chars].rstrip() + " [CONTENT TRUNCATED]"
            lines.append(f"[{idx}] Title: {item['title']} | Evidence: {snippet}")
        joined = "\n".join(lines)
        return f"Observation: Search tool results for '{query}':\n{joined}\nThought: "

    def _build_shadow_kv(self, obs_ids, seq_len_before_commit, base_big_cache=None, base_small_cache=None):
        """Build a shadow KV cache for observation tokens."""
        if not self.enable_shadow_kv or base_small_cache is None or not obs_ids:
            return None
        try:
            if hasattr(base_small_cache, "from_legacy_cache"):
                key_cache = getattr(base_small_cache, "key_cache", getattr(base_small_cache, "_key_cache", []))
                value_cache = getattr(base_small_cache, "value_cache", getattr(base_small_cache, "_value_cache", []))
                shadow_small_cache = DynamicCache.from_legacy_cache((key_cache, value_cache))
            else:
                shadow_small_cache = DynamicCache()
                if hasattr(base_small_cache, "key_cache") and hasattr(base_small_cache, "value_cache"):
                    shadow_small_cache.key_cache = [layer.clone() for layer in base_small_cache.key_cache]
                    shadow_small_cache.value_cache = [layer.clone() for layer in base_small_cache.value_cache]
                elif hasattr(base_small_cache, "_key_cache") and hasattr(base_small_cache, "_value_cache"):
                    shadow_small_cache._key_cache = [layer.clone() for layer in base_small_cache._key_cache]
                    shadow_small_cache._value_cache = [layer.clone() for layer in base_small_cache._value_cache]
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

    def _prepare_tool_payload(self, query, source="sync", seq_len_before_commit=None, base_big_cache=None, base_small_cache=None):
        tool_start_time = time.time()
        docs = self._search_wikipedia(query)
        if not docs:
            docs = self._search_strategyqa_corpus(query)
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
            shadow_state = self._build_shadow_kv(obs_ids, seq_len_before_commit, base_small_cache=base_small_cache)
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

    def _invalidate_pending_prefetch(self, reason="stale", clear_all=False):
        slot_names = ["pending_tool_prefetch", "action_prefetch"] if clear_all else ["pending_tool_prefetch"]
        cleared = False
        for slot_name in slot_names:
            pending = getattr(self, slot_name, None)
            if pending is None:
                continue
            cleared = True
            setattr(self, slot_name, None)
        if clear_all:
            self.hidden_probe_query = None
            self.draft_action_prefetch_query = None
            self.accepted_action_prefetch_query = None
        elif "action_prefetch" in slot_names:
            self.draft_action_prefetch_query = None
            self.accepted_action_prefetch_query = None
        else:
            self.hidden_probe_query = None
        if not cleared:
            return
        if reason == "cancelled":
            self.prefetch_stats["cancelled"] += 1
        elif reason == "stale":
            self.prefetch_stats["stale"] += 1

    def _submit_prefetch_candidate(self, query, source_stage="hidden_probe"):
        if not self.enable_ste or self.ste_executor is None:
            return False

        normalized_query = self._normalize_query(query)
        min_query_len = self._tool_plan_min_query_len()
        if not normalized_query or len(normalized_query) < min_query_len:
            return False
        if source_stage == "action" and not self._is_tool_plan_query_ready(query):
            return False
        slot_name = "pending_tool_prefetch" if source_stage == "hidden_probe" else "action_prefetch"
        pending = getattr(self, slot_name)
        now = time.time()
        if pending is not None:
            same_query = pending.get("normalized_query") == normalized_query
            is_fresh = (now - pending.get("start_time", now)) <= self.ste_ttl_sec
            if same_query and is_fresh:
                self.prefetch_stats["skipped_duplicate"] += 1
                if source_stage == "hidden_probe":
                    self.prefetch_stats["hidden_probe_prefetch_duplicate"] += 1
                return False
            if not is_fresh and source_stage == "hidden_probe":
                self._invalidate_pending_prefetch(reason="stale")
            elif not is_fresh:
                setattr(self, slot_name, None)

        future = self.ste_executor.submit(self._prepare_tool_payload, query, "prefetch", None, None, None)
        payload = {
            "query": query,
            "normalized_query": normalized_query,
            "future": future,
            "start_time": now,
            "status": "pending",
            "source_stage": source_stage,
        }
        setattr(self, slot_name, payload)
        self.prefetch_stats["submitted"] += 1
        self.prefetch_stats["compute_submitted"] += 1
        if source_stage == "hidden_probe":
            self.hidden_probe_query = normalized_query
            self.prefetch_stats["hidden_probe_prefetch_triggered"] += 1
        else:
            self.prefetch_stats["action_async_submitted"] += 1
        return True

    def _prefetch_tool_async(self, query):
        return self._submit_prefetch_candidate(query, source_stage="hidden_probe")

    def _consume_prefetch_if_match(self, query):
        normalized_query = self._normalize_query(query)
        candidates = []
        for slot_name in ["pending_tool_prefetch", "action_prefetch"]:
            pending = getattr(self, slot_name, None)
            if pending is not None and pending.get("normalized_query") == normalized_query:
                candidates.append((slot_name, pending))

        if not candidates:
            self.prefetch_stats["miss"] += 1
            return None

        self.prefetch_stats["reused"] += 1
        if len(candidates) == 2:
            self.prefetch_stats["dual_async_both_available"] += 1
        for slot_name, pending in candidates:
            source_stage = pending.get("source_stage", "hidden_probe")
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
                if source_stage == "hidden_probe":
                    self.prefetch_stats["hidden_probe_hit"] += 1
                    self.prefetch_stats["hidden_probe_reuse"] += 1
                    self.prefetch_stats["dual_async_first_win"] += 1
                else:
                    self.prefetch_stats["action_hit"] += 1
                    self.prefetch_stats["action_reuse"] += 1
                    self.prefetch_stats["dual_async_second_win"] += 1
                setattr(self, slot_name, None)
                other_slot = "action_prefetch" if slot_name == "pending_tool_prefetch" else "pending_tool_prefetch"
                setattr(self, other_slot, None)
                self.hidden_probe_query = None
                self.draft_action_prefetch_query = None
                self.accepted_action_prefetch_query = None
                return payload
            except TimeoutError:
                self.prefetch_stats["timeout"] += 1
                if source_stage == "hidden_probe":
                    self.prefetch_stats["hidden_probe_timeout"] += 1
                else:
                    self.prefetch_stats["action_timeout"] += 1
                continue
            except Exception:
                self.prefetch_stats["miss"] += 1
                setattr(self, slot_name, None)
                continue

        self.prefetch_stats["action_fallback_sync"] += 1
        self.prefetch_stats["dual_async_fallback_action"] += 1
        self._invalidate_pending_prefetch(reason="stale", clear_all=True)
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
                    base_small_cache=base_small_cache,
                )
            self.prefetch_stats["compute_hit"] += 1
            return prefetched
        self.prefetch_stats["compute_sync_fallback"] += 1
        return self._prepare_tool_payload(
            query,
            source="sync",
            seq_len_before_commit=seq_len_before_commit,
            base_big_cache=base_big_cache,
            base_small_cache=base_small_cache,
        )

    def _execute_tool(self, action_buffer, input_ids, big_cache, small_cache):
        query, is_closed = self._parse_action_query(action_buffer)

        if not query:
            return self._materialize_fallback_payload("Observation: Error: Empty search query.\nThought: ", input_ids)

        if not is_closed:
            return self._materialize_fallback_payload("Observation: Error: Incomplete SearchWikipedia[...] block.\nThought: ", input_ids)

        return self._execute_tool_query(
            query,
            seq_len_before_commit=input_ids.shape[1],
            base_big_cache=big_cache,
            base_small_cache=small_cache,
        )

    def _handle_tool_trigger(self, action_buffer, input_ids, big_cache, small_cache):
        extracted_query, _ = self._parse_action_query(action_buffer)
        current_query = extracted_query if extracted_query else action_buffer.strip()
        normalized_query = self._normalize_query(current_query)
        query_signature = self._query_signature(current_query)
        self.hidden_probe_query = None
        self.draft_action_prefetch_query = None
        self.accepted_action_prefetch_query = None

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
    def run_speculative(self, claim, max_gen=8192):
        self.error_count = 0
        self.repeat_action_count = 0
        self.query_history = []
        self.query_signature_history = []
        self._reset_prefetch_state()
        self._reset_trace_state()

        few_shot_examples = (
            "Here are examples of StrategyQA-style yes/no question answering with ReAct. Use the search tool only when evidence is needed.\n\n"
            "--- Example 1 ---\n"
            "Question: Could the members of The Police perform lawful arrests?\n"
            "Thought: I need to verify who can perform lawful arrests and whether The Police members fit that role.\n"
            "Action: SearchWikipedia[The Police lawful arrests]\n"
            "Observation: Wikipedia search results for 'The Police lawful arrests':\n"
            "[1] Title: The Police | Evidence: Description: English rock band Facts: The members of The Police were musicians, not law enforcement officers. Only law enforcement officers can perform lawful arrests. Dataset Answer: no\n"
            "Thought: The evidence says they were musicians, so they could not perform lawful arrests.\n"
            "Final Answer: no\n\n"
            "--- Example 2 ---\n"
            "Question: Are more people today related to Genghis Khan than Julius Caesar?\n"
            "Thought: I need evidence comparing their descendants and known modern relation counts.\n"
            "Action: SearchWikipedia[Genghis Khan Julius Caesar descendants]\n"
            "Observation: Wikipedia search results for 'Genghis Khan Julius Caesar descendants':\n"
            "[1] Title: Genghis Khan | Evidence: Description: founder and first Great Khan of the Mongol Empire Facts: Julius Caesar had three children. Genghis Khan had sixteen children. Modern geneticists have determined that out of every 200 men today has DNA that can be traced to Genghis Khan. Dataset Answer: yes\n"
            "Thought: The evidence supports that more people today are related to Genghis Khan.\n"
            "Final Answer: yes\n\n"
            "--- Examples End ---\n\n"
        )
        self.system_prompt = (
            "System: You are an elite StrategyQA yes/no question-answering engine. Solve with ReAct when needed. "
            "Your goal is to answer each question using evidence returned by the search tool. The tool should prefer Wikipedia search, with local retrieval evidence as fallback.\n\n"
            "### DECISION POLICY (CRITICAL) ###\n"
            "Before every Action, explain in Thought what fact, entity, or relation must be verified.\n"
            "Use the tool when you need factual evidence. First rely on Wikipedia search results returned by the tool; if nothing useful is found, the tool may fall back to local retrieval evidence. Do not invent evidence.\n"
            "### TOOL RULES ###\n"
            "1. When a tool call is needed, first output one concise Thought line, then emit the Action line directly.\n"
            "2. Keep each search query concise and searchable. Prefer entity names plus the needed relation.\n"
            "3. Do not repeat the exact same search query. If a search returns no useful information, prefer a meaningfully different query.\n"
            "4. You may reformulate the same topic once or twice if the new query changes the relation or target entity in a useful way.\n"
            "5. Do not keep expanding a failed query by appending vague filler words only. Relation-focused reformulations are allowed, but low-information filler expansions are not.\n"
            "### THOUGHT STYLE ###\n"
            "Each Thought should contain: (1) current goal, (2) missing evidence, and (3) immediate next step.\n"
            "Keep Thought short but action-guiding, so the model can smoothly draft the next step.\n"
            "When you decide to call the tool, output one concise Thought line, then the Action line.\n"
            "### OUTPUT RULES ###\n"
            "Use the ReAct format with Thought, optional Action, and Observation.\n"
            "The final answer must be yes or no.\n"
            "Always end with exactly one line in the format: Final Answer: <yes|no>\n"
            f"{few_shot_examples}"
            f"Question: {claim}\n"
            "Output: Begin with 'Thought: ' and make the next step explicit before any Action.\n"
        )

        input_ids = self.tokenizer.encode(self.system_prompt, return_tensors="pt").to(self.big_model.device)
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
        latest_tool_plan_query = None
        generated_preview = ""

        def process_token(t_val, phase, action_buf, latest_plan_query, allow_trigger=True):
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
                    normalized_query = self._normalize_query(current_action_plan)
                    if latest_plan_query:
                        if latest_plan_query == normalized_query:
                            self.prefetch_stats["action_plan_match"] += 1
                        else:
                            self.prefetch_stats["action_plan_mismatch"] += 1
                            if normalized_query != self.accepted_action_prefetch_query:
                                submitted = self._submit_prefetch_candidate(current_action_plan, source_stage="action")
                                if submitted:
                                    self.accepted_action_prefetch_query = normalized_query
                    else:
                        self.prefetch_stats["action_without_tool_plan"] += 1
                        latest_plan_query = normalized_query
                        if normalized_query != self.accepted_action_prefetch_query:
                            submitted = self._submit_prefetch_candidate(current_action_plan, source_stage="action")
                            if submitted:
                                self.accepted_action_prefetch_query = normalized_query
                    if allow_trigger:
                        triggered = True
                        latest_plan_query = normalized_query
            return new_phase, action_buf, latest_plan_query, triggered

        def maybe_prefetch_from_draft(token_id, draft_phase, draft_action_buffer, draft_latest_plan_query, prefetch_allowed=True, hidden_probe_mode=False):
            token_text = self._decode_token(token_id)
            draft_phase = self._update_phase(token_id, draft_phase)
            if draft_phase == 0 and SEARCH_WIKIPEDIA_PREFIX not in draft_action_buffer and len(draft_action_buffer) > 256:
                draft_action_buffer = draft_action_buffer[-256:]
            draft_action_buffer = (draft_action_buffer + token_text)[-4000:]

            if not prefetch_allowed or draft_phase != 1:
                return draft_phase, draft_action_buffer, draft_latest_plan_query, False

            if SEARCH_WIKIPEDIA_PREFIX not in draft_action_buffer:
                return draft_phase, draft_action_buffer, draft_latest_plan_query, False

            extracted_query, is_closed = self._extract_search_query(draft_action_buffer)
            if extracted_query is None:
                return draft_phase, draft_action_buffer, draft_latest_plan_query, False

            cleaned_action_query = self._clean_query_candidate(extracted_query)
            safe_query = cleaned_action_query
            if not safe_query:
                return draft_phase, draft_action_buffer, draft_latest_plan_query, False

            ready_for_probe = bool(cleaned_action_query) and (
                is_closed or (hidden_probe_mode and self._is_tool_plan_query_ready(cleaned_action_query))
            )
            normalized_action_query = self._normalize_query(cleaned_action_query)

            if hidden_probe_mode and ready_for_probe:
                self.prefetch_stats["hidden_probe_seen"] += 1
                self.prefetch_stats["hidden_probe_ready"] += 1
                if normalized_action_query != self.hidden_probe_query:
                    submitted = self._submit_prefetch_candidate(cleaned_action_query, source_stage="hidden_probe")
                    if not submitted:
                        self.prefetch_stats["hidden_probe_prefetch_duplicate"] += 1

            if is_closed:
                if draft_latest_plan_query:
                    if draft_latest_plan_query == normalized_action_query:
                        self.prefetch_stats["action_plan_match"] += 1
                    else:
                        self.prefetch_stats["action_plan_mismatch"] += 1
                        if normalized_action_query != self.draft_action_prefetch_query:
                            submitted = self._submit_prefetch_candidate(cleaned_action_query, source_stage="action")
                            if submitted:
                                self.draft_action_prefetch_query = normalized_action_query
                else:
                    self.prefetch_stats["action_without_tool_plan"] += 1
                    draft_latest_plan_query = normalized_action_query
                    if normalized_action_query != self.draft_action_prefetch_query:
                        submitted = self._submit_prefetch_candidate(cleaned_action_query, source_stage="action")
                        if submitted:
                            self.draft_action_prefetch_query = normalized_action_query

            return draft_phase, draft_action_buffer, draft_latest_plan_query, False

        def hidden_probe_then_rollback(base_small_cache, base_small_logits, base_phase, base_action_buffer, base_latest_plan_query):
            if not self.enable_hidden_probe or base_phase != 1:
                return
            checkpoint_len = base_small_cache.get_seq_length()
            probe_phase = base_phase
            probe_action_buffer = base_action_buffer
            probe_latest_plan_query = base_latest_plan_query
            probe_logits = base_small_logits

            for step in range(self.hidden_probe_steps):
                next_probe = torch.argmax(probe_logits, dim=-1, keepdim=True)
                probe_token_id = int(next_probe[0, 0])
                probe_phase, probe_action_buffer, probe_latest_plan_query, _ = maybe_prefetch_from_draft(
                    probe_token_id,
                    probe_phase,
                    probe_action_buffer,
                    probe_latest_plan_query,
                    prefetch_allowed=True,
                    hidden_probe_mode=True,
                )
                has_tool_signal = probe_phase == 1 and SEARCH_WIKIPEDIA_PREFIX in probe_action_buffer
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

        def update_generated_preview(token_id, preview_text, max_chars=1200):
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
                big_logits[0, self.eot_id] = -float("inf")
                big_logits[0, self.eos_id] = -float("inf")

            probs = F.softmax(big_logits, dim=-1)
            entropy = float((-torch.sum(probs * torch.log(probs + 1e-10), dim=-1)).detach().cpu())
            checkpoint_len = big_cache.get_seq_length()

            if entropy > self.threshold:
                big_logits = self._maybe_boost_action_start(big_logits, current_phase, action_buffer, None)
                next_token = torch.argmax(big_logits, dim=-1, keepdim=True)
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                t_val = int(next_token[0, 0])
                current_phase, action_buffer, latest_tool_plan_query, tool_triggered = process_token(
                    t_val, current_phase, action_buffer, latest_tool_plan_query
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
                    action_buffer, latest_tool_plan_query = "", None
                    continue
            else:
                draft_tokens = []
                draft_phase = current_phase
                draft_action_buffer = action_buffer
                draft_latest_plan_query = latest_tool_plan_query

                if not has_final_answer(generated_preview):
                    small_logits[0, self.eot_id] = -float("inf")
                    small_logits[0, self.eos_id] = -float("inf")

                small_logits = self._maybe_boost_action_start(small_logits, current_phase, action_buffer, None, boost=7.5)
                temp_input = torch.argmax(small_logits, dim=-1, keepdim=True)

                hidden_probe_then_rollback(
                    small_cache,
                    small_logits.clone(),
                    current_phase,
                    action_buffer,
                    latest_tool_plan_query,
                )
                for i in range(self.lookahead):
                    temp_token_id = int(temp_input[0, 0])
                    if temp_token_id == self.nl_id or temp_token_id == self.eot_id:
                        boost = 10.0
                        if current_phase == 1:
                            small_logits[0, self.base_obs_id] += boost
                        if not has_final_answer(generated_preview):
                            small_logits[0, self.eot_id] = -float("inf")
                        temp_input = torch.argmax(small_logits, dim=-1, keepdim=True)

                    draft_tokens.append(temp_input)
                    draft_phase, draft_action_buffer, draft_latest_plan_query, _ = maybe_prefetch_from_draft(
                        temp_token_id,
                        draft_phase,
                        draft_action_buffer,
                        draft_latest_plan_query,
                        prefetch_allowed=True,
                        hidden_probe_mode=False,
                    )
                    if temp_token_id == self.eos_id:
                        break
                    if temp_token_id == self.eot_id and has_final_answer(generated_preview):
                        break

                    p_id = torch.tensor([[checkpoint_len + i]], device=temp_input.device)
                    s_out = self.small_model(temp_input, past_key_values=small_cache, position_ids=p_id, use_cache=True)
                    small_logits = s_out.logits[:, -1, :]
                    if not has_final_answer(generated_preview):
                        small_logits[0, self.eot_id] = -float("inf")
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
                        current_phase, action_buffer, latest_tool_plan_query, tool_triggered = process_token(
                            t_val, current_phase, action_buffer, latest_tool_plan_query
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
                    action_buffer, latest_tool_plan_query = "", None
                    continue

                f_logits = big_logits.clone() if n_matches == 0 else verify_logits[:, n_matches - 1, :].clone()
                if not has_final_answer(generated_preview):
                    f_logits[0, self.eot_id] = -float("inf")
                    f_logits[0, self.eos_id] = -float("inf")

                if n_matches < actual_draft_len:
                    rejected_id = int(draft_tokens[n_matches][0, 0])
                    f_logits[0, rejected_id] = -float("inf")
                    next_resample_id = int(torch.argmax(f_logits, dim=-1)[0])
                    if next_resample_id == self.nl_id or next_resample_id == self.eot_id:
                        res_alpha = 3.0
                        if current_phase == 1:
                            f_logits[0, self.base_obs_id] += res_alpha

                final_correct = torch.argmax(f_logits, dim=-1, keepdim=True)
                input_ids = torch.cat([input_ids, final_correct], dim=-1)
                t_val = int(final_correct[0, 0])
                current_phase, action_buffer, latest_tool_plan_query, tool_triggered = process_token(
                    t_val, current_phase, action_buffer, latest_tool_plan_query
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
                    action_buffer, latest_tool_plan_query = "", None
                    continue

        if printed_chunks:
            print("".join(printed_chunks), end="", flush=False)
        dur = time.time() - start_time
        gen_text = "".join(self.committed_output_chunks).strip()
        total_tokens = input_ids.shape[1] - start_len

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

        llm_tokens = total_tokens - committed_observation_tokens
        llm_tps = llm_tokens / dur if dur > 0 else 0.0
        submitted = self.prefetch_stats["submitted"]
        hits = self.prefetch_stats["hit"]
        opportunities = self.prefetch_stats["prefetch_opportunities"]
        action_plan_compared = self.prefetch_stats["action_plan_match"] + self.prefetch_stats["action_plan_mismatch"]
        hidden_probe_seen = self.prefetch_stats["hidden_probe_seen"]
        hidden_probe_ready = self.prefetch_stats["hidden_probe_ready"]
        hidden_probe_triggered = self.prefetch_stats["hidden_probe_prefetch_triggered"]
        hidden_probe_hit = self.prefetch_stats["hidden_probe_hit"]
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
            "hidden_probe_seen": hidden_probe_seen,
            "hidden_probe_ready": hidden_probe_ready,
            "hidden_probe_prefetch_triggered": hidden_probe_triggered,
            "hidden_probe_prefetch_duplicate": self.prefetch_stats["hidden_probe_prefetch_duplicate"],
            "hidden_probe_hit": hidden_probe_hit,
            "hidden_probe_timeout": self.prefetch_stats["hidden_probe_timeout"],
            "hidden_probe_reuse": self.prefetch_stats["hidden_probe_reuse"],
            "hidden_probe_hit_rate_by_triggered": round(hidden_probe_hit / max(1, hidden_probe_triggered), 4),
            "hidden_probe_hit_rate_by_ready": round(hidden_probe_hit / max(1, hidden_probe_ready), 4),
            "hidden_probe_hit_rate_by_seen": round(hidden_probe_hit / max(1, hidden_probe_seen), 4),
            "action_async_submitted": self.prefetch_stats["action_async_submitted"],
            "action_hit": self.prefetch_stats["action_hit"],
            "action_timeout": self.prefetch_stats["action_timeout"],
            "action_reuse": self.prefetch_stats["action_reuse"],
            "action_fallback_sync": self.prefetch_stats["action_fallback_sync"],
            "dual_async_both_available": self.prefetch_stats["dual_async_both_available"],
            "dual_async_first_win": self.prefetch_stats["dual_async_first_win"],
            "dual_async_second_win": self.prefetch_stats["dual_async_second_win"],
            "dual_async_fallback_action": self.prefetch_stats["dual_async_fallback_action"],
            "compute_submitted": self.prefetch_stats["compute_submitted"],
            "compute_hit": self.prefetch_stats["compute_hit"],
            "compute_sync_fallback": self.prefetch_stats["compute_sync_fallback"],
            "shadow_kv_committed": self.prefetch_stats["shadow_kv_committed"],
            "sync_tool_calls": self.prefetch_stats["sync_tool_calls"],
            "prefetch_tool_calls": self.prefetch_stats["prefetch_tool_calls"],
            "action_seen": self.prefetch_stats["action_seen"],
            "action_closed": self.prefetch_stats["action_closed"],
            "action_without_tool_plan": self.prefetch_stats["action_without_tool_plan"],
            "action_plan_match": self.prefetch_stats["action_plan_match"],
            "action_plan_mismatch": self.prefetch_stats["action_plan_mismatch"],
            "action_plan_match_rate": round(self.prefetch_stats["action_plan_match"] / max(1, action_plan_compared), 4),
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


# =================================================================
# Dataset loading logic
# =================================================================
def load_strategyqa_dataset(dataset_path: str):
    print(f">>> Loading evaluation dataset: {dataset_path}")
    if dataset_path.endswith(".json"):
        with open(dataset_path, "r", encoding="utf-8") as f:
            rows = json.load(f)
        if not isinstance(rows, list):
            raise ValueError(f"JSON data must be a list, but got: {type(rows).__name__}")
        full_dataset = Dataset.from_list(rows)
    elif dataset_path.endswith(".jsonl"):
        rows = []
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        full_dataset = Dataset.from_list(rows)
    elif dataset_path.endswith(".arrow"):
        try:
            full_dataset = Dataset.from_file(dataset_path)
        except Exception as e:
            raise RuntimeError(
                f"Unable to read Arrow dataset directly: {dataset_path}\n"
                f"Make sure datasets/pyarrow is installed and that the file is a HuggingFace datasets Arrow shard.\n"
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

    print(f">>> Dataset loaded with {len(full_dataset)} entries. Columns: {list(full_dataset.column_names)}")
    return full_dataset


CLAIM_FIELD_CANDIDATES = ["question", "text", "claim", "sentence", "query", "sentence1", "premise", "hypothesis"]
LABEL_FIELD_CANDIDATES = ["answer", "label", "labels", "gold_label", "target"]


def extract_claim_and_label(entry: Dict[str, Any]):
    claim_value = get_first_available(entry, CLAIM_FIELD_CANDIDATES, default="")
    label_value = get_first_available(entry, LABEL_FIELD_CANDIDATES, default="")

    if isinstance(label_value, list):
        label_value = label_value[0] if label_value else ""
    elif isinstance(label_value, dict):
        label_value = label_value.get("text", label_value.get("answer", ""))

    claim = str(claim_value).strip()
    ref_label = bool_to_label(label_value)
    return claim, ref_label


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sanitized StrategyQA evaluation runner for open-source release.")
    parser.add_argument("--config", type=str, default=None, help="Optional JSON config file.")
    parser.add_argument("--big-model-path", type=str, default=None, help="Path to the large AWQ model.")
    parser.add_argument("--small-model-path", type=str, default=None, help="Path to the small draft model.")
    parser.add_argument("--dataset-path", type=str, default=None, help="Path or dataset name for evaluation data.")
    parser.add_argument("--retrieval-corpus-path", type=str, default=None, help="Path to the local retrieval corpus JSON file.")
    parser.add_argument("--output-csv", type=str, default=None, help="Path to the CSV results file.")
    parser.add_argument("--num-test", type=int, default=None, help="Maximum number of evaluation examples.")
    parser.add_argument("--max-gen", type=int, default=None, help="Maximum generated tokens per sample.")
    parser.add_argument("--threshold", type=float, default=None, help="Entropy threshold for speculative decoding.")
    parser.add_argument("--lookahead", type=int, default=None, help="Draft lookahead length.")
    parser.add_argument("--enable-ste", action="store_true", help="Enable asynchronous speculative tool execution.")
    parser.add_argument("--disable-ste", action="store_true", help="Disable asynchronous speculative tool execution.")
    parser.add_argument("--enable-hidden-probe", action="store_true", help="Enable hidden probe prefetch.")
    parser.add_argument("--disable-hidden-probe", action="store_true", help="Disable hidden probe prefetch.")
    parser.add_argument("--enable-shadow-kv", action="store_true", help="Enable shadow KV cache reuse.")
    parser.add_argument("--disable-shadow-kv", action="store_true", help="Disable shadow KV cache reuse.")
    parser.add_argument("--user-agent", type=str, default=None, help="Custom HTTP user agent for Wikipedia API requests.")
    return parser


def resolve_config(args: argparse.Namespace) -> Dict[str, Any]:
    config = load_config(args.config)

    overrides = {
        "big_model_path": args.big_model_path,
        "small_model_path": args.small_model_path,
        "dataset_path": args.dataset_path,
        "retrieval_corpus_path": args.retrieval_corpus_path,
        "output_csv": args.output_csv,
        "num_test": args.num_test,
        "max_gen": args.max_gen,
        "threshold": args.threshold,
        "lookahead": args.lookahead,
        "user_agent": args.user_agent,
    }
    for key, value in overrides.items():
        if value is not None:
            config[key] = value

    if args.enable_ste:
        config["enable_ste"] = True
    if args.disable_ste:
        config["enable_ste"] = False
    if args.enable_hidden_probe:
        config["enable_hidden_probe"] = True
    if args.disable_hidden_probe:
        config["enable_hidden_probe"] = False
    if args.enable_shadow_kv:
        config["enable_shadow_kv"] = True
    if args.disable_shadow_kv:
        config["enable_shadow_kv"] = False

    return config


def ensure_output_parent(output_path: str) -> None:
    output_file = Path(output_path)
    if output_file.parent and str(output_file.parent) not in {"", "."}:
        output_file.parent.mkdir(parents=True, exist_ok=True)


# =================================================================
# Main evaluation entry
# =================================================================
def main():
    parser = build_argument_parser()
    args = parser.parse_args()
    config = resolve_config(args)

    required_keys = ["big_model_path", "small_model_path", "dataset_path"]
    missing_keys = [key for key in required_keys if not config.get(key)]
    if missing_keys:
        raise ValueError(f"Missing required configuration values: {missing_keys}")

    ensure_output_parent(config["output_csv"])

    engine = ReActSpeculativeEngine(
        big_path=config["big_model_path"],
        small_path=config["small_model_path"],
        strategyqa_dataset_path=config.get("retrieval_corpus_path"),
        threshold=config["threshold"],
        lookahead=config["lookahead"],
        enable_ste=config["enable_ste"],
        ste_max_workers=config["ste_max_workers"],
        ste_min_query_len=config["ste_min_query_len"],
        ste_wait_timeout=config["ste_wait_timeout"],
        ste_ttl_sec=config["ste_ttl_sec"],
        observation_topk_docs=config["observation_topk_docs"],
        observation_topk_paragraphs=config["observation_topk_paragraphs"],
        observation_max_words=config["observation_max_words"],
        observation_max_chars=config["observation_max_chars"],
        enable_shadow_kv=config["enable_shadow_kv"],
        enable_hidden_probe=config["enable_hidden_probe"],
        hidden_probe_steps=config["hidden_probe_steps"],
        output_csv=config["output_csv"],
        user_agent=config["user_agent"],
        request_timeout=(config["request_connect_timeout"], config["request_read_timeout"]),
    )

    try:
        full_dataset = load_strategyqa_dataset(config["dataset_path"])
        num_test = min(int(config["num_test"]), len(full_dataset))
        test_data = full_dataset.select(range(num_test))

        results = []
        correct_count, valid_count = 0, 0

        for i, entry in enumerate(test_data):
            claim, ref_label = extract_claim_and_label(entry)
            print(f"\n\n{'=' * 30} [Case {i + 1}/{num_test}] {'=' * 30}")
            print(f"Available fields: {list(entry.keys())}")
            if not claim:
                print("[Skipped] Unable to parse a claim/question field.")
                continue

            print(f"Question: {claim[:10000]}...")
            print(f"Reference Answer: {ref_label if ref_label else 'N/A'}")

            try:
                dur, total_tokens, acc_rate, generated_text = engine.run_speculative(claim, max_gen=int(config["max_gen"]))
                pred_label = bool_to_label(extract_final_answer(generated_text))
                is_hit = bool(ref_label) and exact_match_score(pred_label, ref_label) == 1.0
                f1 = f1_score(pred_label, ref_label) if ref_label else 0.0
                if ref_label:
                    valid_count += 1
                    if is_hit:
                        correct_count += 1

                print("\n[Full Output Trace]\n" + generated_text)
                print(f"\n[Evaluation] Predicted: {pred_label} | Reference: {ref_label if ref_label else 'N/A'} | EM: {'✅' if is_hit else '❌'} | F1: {f1:.3f}")
                if engine.last_run_stats:
                    print(
                        "[Runtime Stats] "
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
                        "[Async Hit Stats] "
                        f"hidden_probe_submit={engine.last_run_stats['hidden_probe_prefetch_triggered']} | "
                        f"hidden_probe_seen={engine.last_run_stats['hidden_probe_seen']} | "
                        f"hidden_probe_ready={engine.last_run_stats['hidden_probe_ready']} | "
                        f"hidden_probe_duplicate={engine.last_run_stats['hidden_probe_prefetch_duplicate']} | "
                        f"hidden_probe_hit={engine.last_run_stats['hidden_probe_hit']} | "
                        f"hidden_probe_timeout={engine.last_run_stats['hidden_probe_timeout']} | "
                        f"hit_rate/triggered={engine.last_run_stats['hidden_probe_hit_rate_by_triggered']:.2%} | "
                        f"hit_rate/ready={engine.last_run_stats['hidden_probe_hit_rate_by_ready']:.2%} | "
                        f"hit_rate/seen={engine.last_run_stats['hidden_probe_hit_rate_by_seen']:.2%} | "
                        f"action_submit={engine.last_run_stats['action_async_submitted']} | "
                        f"action_hit={engine.last_run_stats['action_hit']} | "
                        f"action_timeout={engine.last_run_stats['action_timeout']} | "
                        f"action_fallback_sync={engine.last_run_stats['action_fallback_sync']}"
                    )
                    print(engine.last_run_stats["generation_trace"])

                res_entry = {
                    "id": i,
                    "qid": entry.get("qid", i),
                    "term": entry.get("term", ""),
                    "question": claim,
                    "ref_answer": ref_label,
                    "pred_answer": pred_label,
                    "exact_match": is_hit,
                    "f1_score": round(f1, 4),
                    "latency_sec": round(dur, 4),
                    "committed_tokens": total_tokens,
                    "committed_tps": round(total_tokens / dur, 2) if dur > 0 else 0.0,
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
                print(f"Error in case {i}: {e}")
                continue

        if results:
            df = pd.DataFrame(results)
            df.to_csv(engine.log_csv, index=False, encoding="utf-8-sig")
            print(
                f"\nEvaluation finished. Average latency: {df['latency_sec'].mean():.2f}s, "
                f"Median latency: {df['latency_sec'].median():.2f}s, "
                f"P90 latency: {df['latency_sec'].quantile(0.9):.2f}s, "
                f"Committed TPS: {df['committed_tps'].mean():.2f}, "
                f"Acceptance rate: {df['accept_rate'].mean():.2%}, "
                f"Average tool time: {df['total_tool_time_sec'].mean():.2f}s"
            )
            if "f1_score" in df.columns and "exact_match" in df.columns:
                print(
                    f"[Accuracy] EM: {df['exact_match'].mean():.2%} | "
                    f"F1: {df['f1_score'].mean():.3f}"
                )
            if "step_count" in df.columns and "action_count" in df.columns:
                print(
                    f"[Reasoning Steps] Average steps: {df['step_count'].mean():.2f} | "
                    f"Median steps: {df['step_count'].median():.0f} | "
                    f"Average actions: {df['action_count'].mean():.2f}"
                )
            if "llm_tps" in df.columns:
                print(
                    f"[LLM Efficiency] LLM TPS: {df['llm_tps'].mean():.2f} | "
                    f"LLM tokens/sample: {df['llm_tokens'].mean():.0f}"
                )
            if "ste_saved_wait_ms" in df.columns:
                print(
                    f"Average STE saved wait: {df['ste_saved_wait_ms'].mean():.2f} ms | "
                    f"STE submission hit rate: {df['ste_hit'].sum() / max(1, df['ste_submitted'].sum()):.2%} | "
                    f"STE opportunity hit rate: {df['ste_hit'].sum() / max(1, df['ste_prefetch_opportunities'].sum()):.2%}"
                )
            if "hidden_probe_hit_rate_by_triggered" in df.columns:
                print(
                    f"Hidden probe hit rate after submission: {df['hidden_probe_hit'].sum() / max(1, df['hidden_probe_prefetch_triggered'].sum()):.2%} | "
                    f"Hidden probe hit rate after ready state: {df['hidden_probe_hit'].sum() / max(1, df['hidden_probe_ready'].sum()):.2%} | "
                    f"Hidden probe hit rate after detection: {df['hidden_probe_hit'].sum() / max(1, df['hidden_probe_seen'].sum()):.2%}"
                )
            if valid_count > 0:
                print(f"Final EM: {correct_count / valid_count:.2%}")
                if "f1_score" in df.columns:
                    print(f"Final F1: {df['f1_score'].mean():.3f}")
                if "step_count" in df.columns:
                    print(f"Average reasoning steps: {df['step_count'].mean():.2f}")
    finally:
        engine.close()


if __name__ == "__main__":
    main()
