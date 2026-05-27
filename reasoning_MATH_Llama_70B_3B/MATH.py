import os
import argparse
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from awq import AutoAWQForCausalLM
from datasets import Dataset
import pandas as pd
import time
import re
import requests
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, TimeoutError

# =================================================================
# Evaluation helper functions
# =================================================================
def extract_boxed_content(text):
    if not isinstance(text, str):
        return ""
    start_idx = text.rfind("\\boxed{")
    if start_idx == -1:
        return ""
    content_start = start_idx + 7
    brace_count = 1
    for i in range(content_start, len(text)):
        if text[i] == '{': brace_count += 1
        elif text[i] == '}': brace_count -= 1
        if brace_count == 0: return text[content_start:i].strip()
    return text[content_start:].strip()

def is_equivalent(pred, ref):
    if not pred or not ref: return False
    
    import sympy
    from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application

    # --- Internal helper: normalize strings ---
    def normalize_math_str(s):

        s = s.strip().lower()
        # Remove LaTeX wrappers
        s = s.replace("$", "").replace("\\(", "").replace("\\)", "")
        # Convert common LaTeX operators into plain-text operators
        s = s.replace("\\frac{", "(").replace("}{", ")/(").replace("}", ")")
        s = s.replace("\\sqrt{", "sqrt(").replace("}", ")")
        s = s.replace("\\sqrt", "sqrt")
        s = s.replace("\\pi", "pi").replace("\\infty", "oo")
        s = s.replace("^", "**").replace("{", "(").replace("}", ")")
        s = s.replace("\\cdot", "*").replace("\\times", "*")
        s = s.replace("\\", "") # Remove remaining backslashes
        return s

    # 1. First pass: exact normalized string match
    pred_norm = normalize_math_str(pred)
    ref_norm = normalize_math_str(ref)
    if pred_norm == ref_norm: return True

    # 2. Second pass: symbolic simplification
    def try_parse_to_sympy(raw, normed):
        transformations = standard_transformations + (implicit_multiplication_application,)
        # Path A: try LaTeX parsing
        try:
            from sympy.parsing.latex import parse_latex
            return parse_latex(raw)
        except:
            pass
        # Path B: try plain-text parsing for non-LaTeX outputs
        try:
            return parse_expr(normed, transformations=transformations)
        except:
            return None

    p_expr = try_parse_to_sympy(pred, pred_norm)
    r_expr = try_parse_to_sympy(ref, ref_norm)

    if p_expr is not None and r_expr is not None:
        try:
            # equals() uses symbolic simplification and numeric probing, which
            # handles cases such as sqrt(3)-1 and -1+sqrt(3)
            if p_expr.equals(r_expr): return True
        except:
            pass

    # 3. Third pass: numeric fallback for cases that cannot be symbolized
    def extract_and_eval(s):
        try:
            # Only works when the string can be parsed as a numeric expression
            return float(sympy.sympify(s).evalf())
        except:
            return None

    p_val = extract_and_eval(pred_norm)
    r_val = extract_and_eval(ref_norm)
    if p_val is not None and r_val is not None:
        if abs(p_val - r_val) < 1e-6: return True

    return False

class ReActSpeculativeEngine:
    def __init__(
        self,
        big_path,
        small_path,
        threshold=0.4,
        lookahead=9,
        enable_ste=True,
        ste_max_workers=2,
        ste_min_expr_len=6,
        ste_wait_timeout=5.0,
        ste_ttl_sec=15.0,
        enable_hidden_probe=True,
        hidden_probe_steps=12,
    ):
        torch.cuda.empty_cache()
        self.threshold = threshold
        self.lookahead = lookahead
        self.enable_ste = enable_ste
        self.ste_min_expr_len = ste_min_expr_len
        self.ste_wait_timeout = ste_wait_timeout
        self.ste_ttl_sec = ste_ttl_sec
        self.enable_hidden_probe = enable_hidden_probe and enable_ste
        self.hidden_probe_steps = max(1, int(hidden_probe_steps))
        
        # 1. Load models
        print(f">>> Loading large model (quantized)...")
        self.big_model_wrapper = AutoAWQForCausalLM.from_quantized(
            big_path, fuse_layers=True, trust_remote_code=True, device_map="auto"
        )
        self.big_model = self.big_model_wrapper.model
        
        print(f">>> Loading small model...")
        self.small_model = AutoModelForCausalLM.from_pretrained(
            small_path, device_map="auto", torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(big_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.is_qwen = "qwen" in big_path.lower() or "qwen" in self.tokenizer.__class__.__name__.lower()
            
        # =================================================================
        # State machine initialization
        # =================================================================
        self.encode_cache = {}
        self.decode_cache = {}
        self.nl_id = self._encode_text("\n")[-1]
        self.eos_id = self.tokenizer.eos_token_id
        self.eot_id = None
        for tok in ["<|im_end|>", "<|eot_id|>"]:
            try:
                tok_id = self.tokenizer.convert_tokens_to_ids(tok)
                if tok_id is not None and tok_id != self.tokenizer.unk_token_id:
                    self.eot_id = tok_id
                    break
            except Exception:
                pass
        if self.eot_id is None:
            self.eot_id = self.eos_id
        
        self.action_ids = {self._encode_text(w)[-1] for w in ["Action", " Action", "\nAction"]}
        self.obs_ids = {self._encode_text(w)[-1] for w in ["Observation", " Observation", "\nObservation"]}
        self.thought_ids = {self._encode_text(w)[-1] for w in ["Thought", " Thought", "\nThought"]}
        self.tool_plan_ids = {self._encode_text(w)[-1] for w in ["ToolPlan", " ToolPlan", "\nToolPlan"]}
        
        self.base_action_id = self._encode_text("Action")[-1]
        self.base_obs_id = self._encode_text("Observation")[-1]
        self.base_thought_id = self._encode_text("Thought")[-1]
        
        self.log_csv = os.getenv("RESULT_CSV_PATH", "evaluation_results.csv")
        self.ste_executor = ThreadPoolExecutor(max_workers=ste_max_workers) if self.enable_ste else None
        self.pending_tool_prefetch = None
        self.hidden_probe_expr = None
        self.action_prefetch = None
        self.prefetch_stats = {}
        self.last_run_stats = {}
        self.total_tool_time_sec = 0.0
        self._reset_prefetch_state()

        self.wolfram_appid = os.getenv("WOLFRAM_APPID", "").strip()
        self.wolfram_short_answers_url = "http://api.wolframalpha.com/v1/result"
        self.wolfram_timeout = (5, 20)
        self.foreground_http_session = requests.Session()
        self.stream_output = True
        
        # Track recent expressions to interrupt oscillating tool-call loops
        self.no_solution_count = 0
        self.error_count = 0
        self.expr_history = []
        self.repeat_action_count = 0
        self.repeat_action_interrupt_count = 0
        self.tool_call_count = 0

    def _reset_prefetch_state(self):
        self.pending_tool_prefetch = None
        self.hidden_probe_expr = None
        self.action_prefetch = None
        self.accepted_action_prefetch_expr = None
        self.draft_action_prefetch_expr = None
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

    def close(self):
        if getattr(self, "foreground_http_session", None) is not None:
            try:
                self.foreground_http_session.close()
            except Exception:
                pass
            self.foreground_http_session = None
        if getattr(self, "ste_executor", None) is not None:
            try:
                self.ste_executor.shutdown(wait=False, cancel_futures=True)
            except TypeError:
                self.ste_executor.shutdown(wait=False)
            except Exception:
                pass
            self.ste_executor = None

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

    def _update_phase(self, token_val, current_phase):
        """Helper: silently update the state machine."""
        if token_val in self.action_ids: return 1
        elif token_val in self.obs_ids: return 2
        elif token_val in self.thought_ids: return 0
        elif token_val in self.tool_plan_ids: return 3
        return current_phase

    def _crop_cache(self, cache: DynamicCache, length: int):
        if cache is None: return cache
        if hasattr(cache, "crop"):
            try:
                cache.crop(length)
                return cache
            except: pass
        k_attr = "key_cache" if hasattr(cache, "key_cache") else "_key_cache"
        v_attr = "value_cache" if hasattr(cache, "value_cache") else "_value_cache"
        keys, values = getattr(cache, k_attr, []), getattr(cache, v_attr, [])
        for i in range(len(keys)):
            keys[i] = keys[i][:, :, :length, :]
            values[i] = values[i][:, :, :length, :]
        for attr in ["_seen_tokens", "seen_tokens", "last_seen_seq_assign"]:
            if hasattr(cache, attr): setattr(cache, attr, length)
        return cache

    # =================================================================
    # Tool expression extraction and handling
    # =================================================================
    def _extract_calculate_expr(self, text):
        start = text.find("Calculate[")
        if start == -1:
            return None, False

        i = start + len("Calculate[")
        depth = 1
        expr_chars = []

        while i < len(text):
            ch = text[i]
            if ch == '[':
                depth += 1
                expr_chars.append(ch)
            elif ch == ']':
                depth -= 1
                if depth == 0:
                    return "".join(expr_chars).strip(), True
                expr_chars.append(ch)
            else:
                expr_chars.append(ch)
            i += 1

        return "".join(expr_chars).strip(), False

    def _normalize_tool_expr(self, expr_str):
        return re.sub(r"\s+", " ", expr_str.strip()) if expr_str else ""

    def _tool_plan_min_expr_len(self) -> int:
        return max(1, min(self.ste_min_expr_len, 12))

    def _is_tool_plan_expr_ready(self, expr_str):
        normalized_expr = self._normalize_tool_expr(expr_str)
        if not normalized_expr:
            return False

        if len(normalized_expr) < self._tool_plan_min_expr_len():
            return False

        if normalized_expr.endswith(('=', '+', '-', '*', '/', '^', '(', '[', '{', ',', ':')):
            return False

        pairs = {'(': ')', '[': ']', '{': '}'}
        closing = {v: k for k, v in pairs.items()}
        stack = []
        for ch in normalized_expr:
            if ch in pairs:
                stack.append(ch)
            elif ch in closing:
                if not stack or stack[-1] != closing[ch]:
                    return False
                stack.pop()
        if stack:
            return False

        lowered_expr = normalized_expr.lower()
        if 'solve' in lowered_expr and ' for ' not in lowered_expr and '=' in normalized_expr:
            return False

        return True

    def _prepare_tool_payload(self, expr_str, source="sync"):
        tool_start_time = time.time()
        observation_text = self._execute_tool(f"Action: Calculate[{expr_str}]")
        if source == "prefetch":
            self.prefetch_stats["prefetch_tool_calls"] += 1
        else:
            self.prefetch_stats["sync_tool_calls"] += 1
        tool_elapsed_sec = time.time() - tool_start_time
        return {
            "expr": expr_str,
            "normalized_expr": self._normalize_tool_expr(expr_str),
            "source": source,
            "fetched_at": time.time(),
            "observation_text": observation_text,
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
            self.hidden_probe_expr = None
            self.draft_action_prefetch_expr = None
            self.accepted_action_prefetch_expr = None
        elif "action_prefetch" in slot_names:
            self.draft_action_prefetch_expr = None
            self.accepted_action_prefetch_expr = None
        else:
            self.hidden_probe_expr = None
        if not cleared:
            return
        if reason == "cancelled":
            self.prefetch_stats["cancelled"] += 1
        elif reason == "stale":
            self.prefetch_stats["stale"] += 1

    def _submit_prefetch_candidate(self, expr_str, source_stage="hidden_probe"):
        if not self.enable_ste or self.ste_executor is None:
            return False

        normalized_expr = self._normalize_tool_expr(expr_str)
        min_expr_len = self._tool_plan_min_expr_len()
        if not normalized_expr or len(normalized_expr) < min_expr_len:
            return False

        if source_stage == "action" and not self._is_tool_plan_expr_ready(expr_str):
            return False

        slot_name = "pending_tool_prefetch" if source_stage == "hidden_probe" else "action_prefetch"
        pending = getattr(self, slot_name)
        now = time.time()
        if pending is not None:
            same_expr = pending.get("normalized_expr") == normalized_expr
            is_fresh = (now - pending.get("start_time", now)) <= self.ste_ttl_sec
            if same_expr and is_fresh:
                self.prefetch_stats["skipped_duplicate"] += 1
                if source_stage == "hidden_probe":
                    self.prefetch_stats["hidden_probe_prefetch_duplicate"] += 1
                return False
            if not is_fresh and source_stage == "hidden_probe":
                self._invalidate_pending_prefetch(reason="stale")
            elif not is_fresh:
                setattr(self, slot_name, None)

        future = self.ste_executor.submit(self._prepare_tool_payload, expr_str, "prefetch")
        payload = {
            "expr": expr_str,
            "normalized_expr": normalized_expr,
            "future": future,
            "start_time": now,
            "status": "pending",
            "source_stage": source_stage,
        }
        setattr(self, slot_name, payload)
        if source_stage == "hidden_probe":
            self.hidden_probe_expr = normalized_expr
        self.prefetch_stats["submitted"] += 1
        self.prefetch_stats["compute_submitted"] += 1
        if source_stage == "hidden_probe":
            self.prefetch_stats["hidden_probe_prefetch_triggered"] += 1
        else:
            self.prefetch_stats["action_async_submitted"] += 1
        return True

    def _prefetch_tool_async(self, expr_str):
        return self._submit_prefetch_candidate(expr_str, source_stage="hidden_probe")

    def _consume_prefetch_if_match(self, expr_str):
        normalized_expr = self._normalize_tool_expr(expr_str)
        candidates = []
        for slot_name in ["pending_tool_prefetch", "action_prefetch"]:
            pending = getattr(self, slot_name, None)
            if pending is not None and pending.get("normalized_expr") == normalized_expr:
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
                self.hidden_probe_expr = None
                self.draft_action_prefetch_expr = None
                self.accepted_action_prefetch_expr = None
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

    def _maybe_prefetch_from_draft(
        self,
        token_id,
        draft_phase,
        draft_buffer,
        draft_latest_plan_expr,
        allow_action_prefetch=True,
        hidden_probe_mode=False,
    ):
        """Detect Action during drafting and trigger prefetch; initial probing is handled by the hidden probe."""
        token_text = self._decode_token(token_id)
        new_phase = self._update_phase(token_id, draft_phase)

        if new_phase in [1, 3]:
            if new_phase != draft_phase:
                draft_buffer = ""
            draft_buffer = (draft_buffer + token_text)[-512:]
        else:
            draft_buffer = ""

        if new_phase == 1 and "Calculate[" in draft_buffer:
            extracted_expr, is_closed = self._extract_calculate_expr(draft_buffer)
            if extracted_expr is not None:
                cleaned_expr = self._normalize_tool_expr(extracted_expr)
                ready_for_probe = bool(cleaned_expr) and (
                    is_closed or (hidden_probe_mode and self._is_tool_plan_expr_ready(extracted_expr))
                )
                if ready_for_probe:
                    if hidden_probe_mode:
                        self.prefetch_stats["hidden_probe_seen"] += 1
                        self.prefetch_stats["hidden_probe_ready"] += 1
                        if cleaned_expr != self.hidden_probe_expr:
                            submitted = self._submit_prefetch_candidate(extracted_expr, source_stage="hidden_probe")
                            if not submitted:
                                self.prefetch_stats["hidden_probe_prefetch_duplicate"] += 1
                    elif is_closed:
                        if draft_latest_plan_expr:
                            if draft_latest_plan_expr == cleaned_expr:
                                self.prefetch_stats["action_plan_match"] += 1
                            else:
                                self.prefetch_stats["action_plan_mismatch"] += 1
                        else:
                            self.prefetch_stats["action_without_tool_plan"] += 1
                        draft_latest_plan_expr = cleaned_expr
                        if allow_action_prefetch and cleaned_expr != self.draft_action_prefetch_expr:
                            submitted = self._submit_prefetch_candidate(extracted_expr, source_stage="action")
                            if submitted:
                                self.draft_action_prefetch_expr = cleaned_expr

        return new_phase, draft_buffer, draft_latest_plan_expr

    def _preprocess_wolfram_expr(self, expr_str):
        expr_str = expr_str.strip()
        expr_str = expr_str.replace('×', '*').replace('÷', '/')
        expr_str = expr_str.replace('\\cdot', '*')
        expr_str = re.sub(r"\s+", " ", expr_str)
        return expr_str

    def _query_wolfram_short_answers(self, expr_str, session=None):
        if not self.wolfram_appid:
            raise ValueError("Missing WOLFRAM_APPID environment variable.")

        session = session or self.foreground_http_session

        # Use /v1/result and the query parameter key "i"
        url = "http://api.wolframalpha.com/v1/result"
        params = {
            "appid": self.wolfram_appid,
            "i": expr_str,  # The short-answer endpoint requires the 'i' key
        }
        
        # requests handles URL encoding automatically
        response = session.get(
            url,
            params=params,
            timeout=self.wolfram_timeout,
        )
        
        # If Wolfram cannot parse the query, return a structured status payload
        if response.status_code == 501:
            return {
                "status": "unparsed",
                "answer": "Wolfram could not understand the query. Try rephrasing with 'solve', 'simplify', or 'factor'."
            }
            
        response.raise_for_status()
        answer_text = response.text.strip()
        
        if not answer_text:
            raise ValueError("Empty response from Wolfram short-answers API.")
            
        # Return a structured status payload for successful responses as well
        return {
            "status": "ok",
            "answer": answer_text
        }

    def _format_wolfram_observation(self, expr_str, tool_result):
        lowered_expr = expr_str.lower()
        status = tool_result.get("status", "ok")
        answer_text = tool_result.get("answer", "")
        lowered_answer = answer_text.lower()

        # Inspect both status and answer text for parse failures
        if status == "unparsed" or "could not understand" in lowered_answer or "did not understand" in lowered_answer:
            prefix = "[ToolError]"
            thought = (
        "Wolfram failed. If this was a pure numeric calculation, remove 'simplify' and use raw math like Calculate[ 64/32 ]. "
        "If it was an equation, ensure you used 'solve'. If you are stuck, you MUST solve it manually in Thought."
    )
        elif any(op in expr_str for op in ['<=', '>=', '<', '>', '=']):
            prefix = "[Solution]"
            thought = "Use the exact solution above. Do not switch to decimals unless requested."
        elif any(keyword in lowered_answer for keyword in ["approximately", "approx.", "decimal"]):
            prefix = "[Result]"
            thought = "Use this numeric result to answer the question."
        elif any(keyword in lowered_expr for keyword in ["simplify", "expand", "factor"]):
            prefix = "[Simplified]"
            thought = "Use this algebraic form to reach the target."
        else:
            prefix = "[Result]"
            thought = "Use this exact result to continue the reasoning."

        return f"Observation: {prefix}: {answer_text}\nThought: {thought}\n"

    def _execute_tool(self, action_buffer, session=None):
        """
        Main tool entry point: call the Wolfram short-answers API and wrap the
        result as an Observation block.
        """
        expr_str, is_closed = self._extract_calculate_expr(action_buffer)
        if expr_str is None:
            expr_str = action_buffer.replace("Action:", "").replace("Action", "").replace("\n", "").strip()

        if not expr_str:
            return "Observation: Error: Empty calculation. Think first, then call the calculator with one complete expression.\nThought: "

        if not is_closed and "Calculate[" in action_buffer:
            return "Observation: Error: Incomplete Calculate[...] block. Finish the full expression before using the tool.\nThought: "

        preprocessed_expr = self._preprocess_wolfram_expr(expr_str)

        try:
            answer_text = self._query_wolfram_short_answers(preprocessed_expr, session=session)
            self.no_solution_count, self.error_count = 0, 0
            return self._format_wolfram_observation(preprocessed_expr, answer_text)
        except Exception as e:
            self.error_count += 1
            if self.error_count >= 3:
                self.error_count = 0
                return "Observation: FATAL ERROR: Repeated malformed or failed Wolfram tool calls. Stop using the calculator and reason manually.\nThought: "
            return f"Observation: Error: {str(e)}\nThought: Think through the algebraic setup more carefully before trying another tool call.\n"

    def _execute_tool_query(self, expr_str):
        self.prefetch_stats["prefetch_opportunities"] += 1
        prefetched = self._consume_prefetch_if_match(expr_str)
        if prefetched is not None:
            self.prefetch_stats["compute_hit"] += 1
            return prefetched
        self.prefetch_stats["compute_sync_fallback"] += 1
        return self._prepare_tool_payload(expr_str, source="sync")

    def _inject_tool_result(self, tool_obs_str, input_ids, current_len, big_cache, small_cache):
        """Inject tool output into both KV caches and refresh logits."""
        if self.stream_output:
            print(tool_obs_str, end="", flush=True)
        obs_ids = self._encode_text(tool_obs_str)
        obs_tensor = torch.tensor([obs_ids], device=self.big_model.device)

        pos = torch.arange(current_len, current_len + len(obs_ids), device=self.big_model.device).unsqueeze(0)

        big_out = self.big_model(obs_tensor, past_key_values=big_cache, position_ids=pos, use_cache=True)
        small_out = self.small_model(obs_tensor.to(self.small_model.device), past_key_values=small_cache, position_ids=pos, use_cache=True)

        big_logits = big_out.logits[:, -1, :]
        small_logits = small_out.logits[:, -1, :]

        input_ids.append(obs_ids)
        current_len += len(obs_ids)
        return input_ids, current_len, big_logits, small_logits

    def _handle_tool_trigger(self, action_buffer, input_ids, current_len, big_cache, small_cache):
        extracted_expr, _ = self._extract_calculate_expr(action_buffer)
        current_expr = self._normalize_tool_expr(extracted_expr if extracted_expr else action_buffer.strip())
        self.hidden_probe_expr = None
        self.draft_action_prefetch_expr = None
        self.accepted_action_prefetch_expr = None
        
        if current_expr in self.expr_history:
            self.repeat_action_count += 1
        else:
            self.expr_history.append(current_expr)
            if len(self.expr_history) > 5:
                self.expr_history.pop(0)
            self.repeat_action_count = 0
            
        if self.repeat_action_count >= 2:
            # Use a generic interruption path to force manual reasoning or a final answer
            tool_payload = {
                "observation_text": (
                    "Observation: [System Error] Tool execution is stuck in an infinite loop. You MUST immediately STOP using the Calculate tool.\n"
                    "Thought: The calculator is failing continuously. I must stop trying to use it. I will rely entirely on my own manual algebraic reasoning (like completing the square or simple arithmetic) to move forward, or I will output the Final Answer now.\n"
                ),
                "tool_time_sec": 0.0,
            }
            self.repeat_action_interrupt_count += 1
            self.repeat_action_count = 0
        else:
            tool_payload = self._execute_tool_query(current_expr)
        
        self.tool_call_count += 1
        self.total_tool_time_sec += float(tool_payload.get("tool_time_sec", 0.0))
        return self._inject_tool_result(tool_payload["observation_text"], input_ids, current_len, big_cache, small_cache)


    def _hidden_probe_then_rollback(self, base_small_cache, base_small_logits, base_phase, base_buffer, base_latest_plan_expr):
        if not self.enable_hidden_probe or base_phase != 1:
            return
        checkpoint_len = base_small_cache.get_seq_length()
        probe_phase = base_phase
        probe_buffer = base_buffer
        probe_latest_plan_expr = base_latest_plan_expr
        probe_logits = base_small_logits

        for step in range(self.hidden_probe_steps):
            next_probe = torch.argmax(probe_logits, dim=-1, keepdim=True)
            probe_token_id = int(next_probe[0, 0])
            probe_phase, probe_buffer, probe_latest_plan_expr = self._maybe_prefetch_from_draft(
                probe_token_id,
                probe_phase,
                probe_buffer,
                probe_latest_plan_expr,
                allow_action_prefetch=True,
                hidden_probe_mode=True,
            )
            has_tool_signal = probe_phase == 1 and "Calculate[" in probe_buffer
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

    @torch.no_grad()
    def run_speculative(self, question, max_gen=1024):
        self.no_solution_count = 0
        self.error_count = 0
        self.repeat_action_count = 0
        self.repeat_action_interrupt_count = 0
        self.expr_history = [] # Clear history for each new question
        self.tool_call_count = 0
        self._reset_prefetch_state()

        few_shot_examples = (
            "Here are examples of solving math problems with careful reasoning. The calculator is only a fallback and not the default path.\n\n"
            "--- Example 1: No Tool Needed ---\n"
            "Question: If x+y=10 and x-y=4, find x.\n"
            "Thought: The target is x. This is a short conceptual step, so I should reason directly. Adding the equations gives 2*x=14, hence x=7. No calculator is needed.\n"
            "Final Answer: \\boxed{7}\n\n"
            "--- Example 2: Think First, Then One Tool Call (Strict Syntax) ---\n"
            "Question: Solve 3x + 7 = 22.\n"
            "Thought: The target is x. This step is a genuine algebraic solve, so one calculator call is justified. I must use explicit multiplication '*' and standard solver syntax.\n"
            "Action: Calculate[ solve 3*x + 7 = 22 for x ]\n"
            "Observation: [Solution]: x = 5\n"
            "Thought: I now have the exact value required by the question, so I should stop.\n"
            "Final Answer: \\boxed{5}\n\n"
            "--- Example 3: Definitions First, Tool Later If Needed ---\n"
            "Question: A rectangle has perimeter 30 and length 9. Find its area.\n"
            "Thought: The target is the area. First use the definition of perimeter: 2*(L+W)=30, so L+W=15 and W=6. This is easy reasoning, so no calculator is needed.\n"
            "Thought: The area is 9*6=54.\n"
            "Final Answer: \\boxed{54}\n\n"
            "--- Example 4: Simplify a Rational Expression by Factoring ---\n"
            "Question: Simplify (x^2 - 4) / (x^2 - 2x).\n"
            "Thought: I need to simplify the rational expression. Factoring numerator and denominator will reveal cancellation.\n"
            "Action: Calculate[ factor x^2 - 4 ]\n"
            "Observation: [Simplified]: (x - 2)*(x + 2)\n"
            "Action: Calculate[ factor x^2 - 2*x ]\n"
            "Observation: [Simplified]: (x - 2)*x\n"
            "Thought: Now I can cancel the common factor (x-2), provided x ≠ 2. The simplified form is (x+2)/x.\n"
            "Final Answer: \\boxed{\\frac{x+2}{x}}\n\n"

            "--- Example 5: Expand and Combine Like Terms ---\n"
            "Question: Expand and simplify (2x+3)(x-4) - (x-1)^2.\n"
            "Thought: This is a pure algebraic expansion and combination. I will use the tool to avoid sign errors.\n"
            "Action: Calculate[ simplify (2*x+3)*(x-4) - (x-1)^2 ]\n"
            "Observation: [Simplified]: x^2 - 12*x - 13\n"
            "Thought: The tool returned the fully simplified polynomial.\n"
            "Final Answer: \\boxed{x^2 - 12x - 13}\n\n"
            "--- Example 6: For All Quantifier (Find p such that inequality holds for every q>0) ---\n"
            "Question: Find all p such that for every q>0, (p+1)q > 2q.\n"
            "Thought: The problem requires the inequality to hold for ALL q>0. I cannot solve for p in terms of q and then guess. Instead, I will simplify algebraically first, then use the tool's 'Reduce' or 'ForAll' capability to eliminate the quantifier.\n"
            "Thought: Divide both sides by q (since q>0, direction unchanged): p+1 > 2 → p > 1. But to demonstrate the tool method for complex cases:\n"
            "Action: Calculate[ Reduce[ForAll[q, q>0, (p+1)*q > 2*q], p, Reals] ]\n"
            "Observation: p > 1\n"
            "Thought: The tool directly returns the range of p that works for all q>0. This matches manual simplification.\n"
            "Final Answer: \\boxed{(1, \\infty)}\n\n"
            "--- Example 7: Complex ForAll with Quadratic ---\n"
            "Question: Find all p such that for every q>0, 3(p+3)q > 2p^2 q.\n"
            "Thought: Since q>0, divide both sides by q to get 3(p+3) > 2p^2. Rearranged: 2p^2 - 3*p - 9 < 0. Now I need the p range that satisfies this for all q>0 — note that q has disappeared, so the condition is independent of q. Solve the quadratic inequality.\n"
            "Action: Calculate[ solve 2*p^2 - 3*p - 9 < 0 for p ]\n"
            "Observation: -3/2 < p < 3\n"
            "Thought: But the original denominator (p+q) requires p+q>0 for all q>0, which forces p ≥ 0. Intersecting with (-1.5, 3) gives [0,3).\n"
            "Final Answer: \\boxed{[0,3)}\n\n"
            "--- Examples End ---\n\n"
            "--- Example 8: Recursive Operation (Step-by-step) ---\n"
            "Question: Define # by: r # 0 = r, r # s = s # r, (r+1) # s = (r # s) + s + 1. Find 11 # 5.\n"
            "Thought: This is recursive. I'll compute 0#5 = 5 (base). Then use (r+1)#5 = (r#5)+5+1 repeatedly.\n"
            "Action: Calculate[ 5 + 5 + 1 ]\n"
            "Observation: 11\n"
            "Thought: So 1#5=11. Next, 2#5 = 1#5 + 5 + 1\n"
            "Action: Calculate[ 11 + 5 + 1 ]\n"
            "Observation: 17\n"
            "Thought: 2#5=17. Continue: 3#5 = 17+5+1=23, 4#5=23+5+1=29, 5#5=29+5+1=35.\n"
            "Thought: By commutativity, 11#5 = 5#11. Now compute 5#11 similarly: 0#11=11, 1#11=11+11+1=23, 2#11=23+11+1=35, 3#11=35+11+1=47, 4#11=47+11+1=59, 5#11=59+11+1=71.\n"
            "Final Answer: \\boxed{71}\n\n"
        )
        system_content = (
            "You are an elite mathematical reasoning engine. Your primary goal is absolute accuracy. "
            "LLMs are unreliable for mental algebra and sign changes during transposition. Therefore, you MUST delegate all formal algebraic manipulations to the Calculate tool to ensure reliability.\n\n"

            "### WOLFRAM API SYNTAX RULES (STRICT HIERARCHY) ###\n"
            "To eliminate [ToolError], you must map the mathematical object to the correct keyword strictly:\n"
            "1. EQUATIONS (Any string containing '='):\n"
            "   - MANDATORY: Use `solve <equation> for <variable>`. \n"
            "   - FORBIDDEN: NEVER use 'simplify' or 'arithmetic' on an equation. It will fail.\n"
            "   - Example: Calculate[ solve x^2 + 4*x + y^2 - 6*y = 3 for x,y ]\n"
            "   - Incorrect: Calculate[ simplify x^2 + 4*x + y^2 - 6*y = 3 ]\n"
            "2. EXPRESSIONS (No '='):\n"
            "   - MANDATORY: Use `simplify`, `factor`, or `expand`.\n"
            "   - Example: Calculate[ factor x^2 + 5*x + 6 ]\n"
            "3. RAW ARITHMETIC:\n"
            "   - Only for pure numeric calculations without variables.\n"
            "   - Example: Calculate[ 16.8 + 18.2 ]\n"
            "4. MANDATORY SYNTAX:\n"
            "   - Use `*` for ALL multiplications (e.g., `3*x`, `(a+b)*(c+d)`).\n"
            "   - NO ENGLISH instructions (e.g., 'round', 'limit', 'value of') inside Calculate[...].\n\n"

            "### DECISION POLICY: ZERO MENTAL ALGEBRA (CRITICAL) ###\n"
            "1. NO MENTAL TRANSPOSITION: Do NOT isolate variables or move terms across the '=' sign in your Thought. \n"
            "   - WRONG: 'Since -2x + y = k and x = -8.4, then k = 16.8 + y.' (This leads to fatal sign errors!)\n"
            "   - RIGHT: 'I will substitute the values into the raw equation and use the solver to find the target.' \n"
            "     -> Action: Calculate[ solve -2*(-8.4) + 9.8 = k for k ]\n"
            "2. EXHAUSTIVE ANALYSIS: If a problem involves multiple cases, pairings, or combinations, you MUST list ALL possible scenarios in Thought before using the tool to compare them.\n"
            "3. DOMAIN VIGILANCE: For rational expressions or inequalities, always identify values where the expression is undefined (e.g., denominator = 0) in Thought first.\n\n"

            "### TOOL RULES ###\n"
            "1. When a tool call is needed, first state one concise reasoning line in Thought, then emit the Action line directly.\n"
            "2. Allowed: Calculate[ <expr> ] for exact arithmetic or simplification.\n"
            "3. Allowed: Calculate[ solve <eq> for <var> ] for any algebraic isolation or solving.\n"
            "4. Forbidden: inline substitutions such as Calculate[a*x+b=0, x=2]. Perform substitutions in Thought first.\n"
            "5. If the calculator returns [ToolError], analyze if you used 'simplify' on an '=' equation. Rewrite using 'solve' or check for missing `*` signs.\n"
            "6. **Tip:** For some complex or multi‑step calculation, often rely on the Calculate tool rather than mental algebra.\n"
            "### THOUGHT DISCIPLINE ###\n"
            "Every Action must be justified. Prefer saying: 'I will use the solver to avoid potential sign errors in manual transposition.'\n"
            "When you decide to call the tool, output one concise reasoning line, then the Action line.\n"
            "Prefer exact forms (e.g., 20/3, sqrt(5)). Do not convert to decimals unless the question explicitly requests an approximation.\n"
            "Always end your reasoning with Final Answer: \\boxed{result}. Do not generate any extra text after the final answer.\n\n"
            f"{few_shot_examples}"
        )

        # 2. Build the chat-format prompt expected by the tokenizer
        user_content = f"Question: {question}\nSolve this step-by-step using the ReAct format. Begin with 'Thought: ' and think carefully before any Action."
        if hasattr(self.tokenizer, "apply_chat_template") and getattr(self.tokenizer, "chat_template", None):
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
            ]
            prompt_str = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            prompt_str = f"System: {system_content}\n\nUser: {user_content}\nAssistant: "

        prompt_ids = self.tokenizer.encode(prompt_str, add_special_tokens=False)

        # 3. Force the generation prefix to start with "Thought: "
        thought_prefix_ids = self.tokenizer.encode("Thought: ", add_special_tokens=False)

        # 4. Manage context with Python token buffers to avoid frequent torch.cat
        input_id_chunks = [prompt_ids, thought_prefix_ids]
        flat_input_ids = prompt_ids + thought_prefix_ids
        input_ids_tensor = torch.tensor([flat_input_ids], device=self.big_model.device)
        current_len = len(flat_input_ids)
        big_cache, small_cache = DynamicCache(), DynamicCache()

        big_out = self.big_model(input_ids_tensor, past_key_values=big_cache, use_cache=True)
        small_out = self.small_model(input_ids_tensor.to(self.small_model.device), past_key_values=small_cache, use_cache=True)
        
        big_logits = big_out.logits[:, -1, :]
        small_logits = small_out.logits[:, -1, :] 
        
        start_time, start_len = time.time(), current_len
        accepted_count, total_draft = 0, 0
        current_phase = 0
        latest_tool_plan_expr = None

        active_buffer = ""

        def process_token(t_val, phase, buffer, latest_plan_expr):
            """Process tokens and detect Action, keeping only hidden-probe and Action-prefetch paths."""
            new_phase = self._update_phase(t_val, phase)
            text_chunk = self._decode_token(t_val)
            track_buffer = new_phase in [0, 1, 3]

            if track_buffer:
                if new_phase != phase:
                    buffer = ""
                buffer = (buffer + text_chunk)[-512:]
            else:
                buffer = ""

            triggered = False

            # Action-stage detection with explicit prefetching
            if new_phase == 1 and "Calculate[" in buffer:
                extracted_expr, is_closed = self._extract_calculate_expr(buffer)
                if extracted_expr is not None and is_closed:
                    self.prefetch_stats["action_seen"] += 1
                    self.prefetch_stats["action_closed"] += 1
                    normalized_action_expr = self._normalize_tool_expr(extracted_expr)
                    if latest_plan_expr:
                        if latest_plan_expr == normalized_action_expr:
                            self.prefetch_stats["action_plan_match"] += 1
                        else:
                            self.prefetch_stats["action_plan_mismatch"] += 1
                            if normalized_action_expr != self.accepted_action_prefetch_expr:
                                submitted = self._submit_prefetch_candidate(extracted_expr, source_stage="action")
                                if submitted:
                                    self.accepted_action_prefetch_expr = normalized_action_expr
                    else:
                        self.prefetch_stats["action_without_tool_plan"] += 1
                        latest_plan_expr = normalized_action_expr
                        if normalized_action_expr != self.accepted_action_prefetch_expr:
                            submitted = self._submit_prefetch_candidate(extracted_expr, source_stage="action")
                            if submitted:
                                self.accepted_action_prefetch_expr = normalized_action_expr
                    if is_closed:
                        triggered = True
                        latest_plan_expr = normalized_action_expr

            return new_phase, buffer, latest_plan_expr, triggered

        while (current_len - start_len) < max_gen:
            probs = F.softmax(big_logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1).item()
            checkpoint_len = big_cache.get_seq_length()

            # --- Mode A: high entropy, generate directly with the large model ---
            if entropy > self.threshold:
                next_token = torch.argmax(big_logits, dim=-1, keepdim=True)
                t_val = next_token.item()
                input_id_chunks.append([t_val])
                current_len += 1
                
                current_phase, active_buffer, latest_tool_plan_expr, tool_triggered = process_token(
                    t_val, current_phase, active_buffer, latest_tool_plan_expr
                )
                if self.stream_output:
                    print(self._decode_token(t_val), end="", flush=True)
                
                if t_val == self.eos_id or t_val == self.eot_id: break

                pos = torch.tensor([[checkpoint_len]], device=next_token.device)
                big_out = self.big_model(next_token, past_key_values=big_cache, position_ids=pos, use_cache=True)
                small_out = self.small_model(next_token.to(self.small_model.device), past_key_values=small_cache, position_ids=pos, use_cache=True)
                big_logits, small_logits = big_out.logits[:, -1, :], small_out.logits[:, -1, :]
                
                if tool_triggered:
                    input_id_chunks, current_len, big_logits, small_logits = self._handle_tool_trigger(
                        active_buffer, input_id_chunks, current_len, big_cache, small_cache
                    )
                    current_phase, active_buffer, latest_tool_plan_expr = 0, "", None
                    continue
                
            # --- Mode B: low entropy, draft with the small model and verify with the large model ---
            else:
                draft_tokens = []
                draft_phase = current_phase
                draft_action_buffer = active_buffer
                draft_latest_plan_expr = latest_tool_plan_expr
                draft_action_prefetch_expr = None
                
                temp_input = torch.argmax(small_logits, dim=-1, keepdim=True)

                self._hidden_probe_then_rollback(
                    small_cache,
                    small_logits.clone(),
                    current_phase,
                    active_buffer,
                    latest_tool_plan_expr,
                )
                
                for i in range(self.lookahead):
                    # Boost newline transitions toward the next structural tag
                    if temp_input.item() == self.nl_id:
                        boost = 7
                        if current_phase == 0: small_logits[0, self.base_action_id] += boost
                        elif current_phase == 1: small_logits[0, self.base_obs_id] += boost
                        elif current_phase == 2: small_logits[0, self.base_thought_id] += boost
                        elif current_phase == 3: small_logits[0, self.base_action_id] += boost
                        temp_input = torch.argmax(small_logits, dim=-1, keepdim=True)

                    draft_tokens.append(temp_input)
                    
                    # Detect Action during drafting and trigger prefetching
                    draft_phase, draft_action_buffer, draft_latest_plan_expr = self._maybe_prefetch_from_draft(
                        temp_input.item(),
                        draft_phase,
                        draft_action_buffer,
                        draft_latest_plan_expr,
                        allow_action_prefetch=True,
                        hidden_probe_mode=False,
                    )
                    
                    # Stop drafting early once a complete Action has been detected
                    if self.enable_ste and draft_phase == 1 and "Calculate[" in draft_action_buffer:
                        extracted_draft_expr, draft_is_closed = self._extract_calculate_expr(draft_action_buffer)
                        if extracted_draft_expr is not None and draft_is_closed:
                            draft_action_prefetch_expr = extracted_draft_expr
                            break
                    
                    if temp_input.item() == self.eos_id or temp_input.item() == self.eot_id: break
                    
                    p_id = torch.tensor([[checkpoint_len + i]], device=temp_input.device)
                    s_out = self.small_model(temp_input, past_key_values=small_cache, position_ids=p_id, use_cache=True)
                    small_logits = s_out.logits[:, -1, :] 
                    temp_input = torch.argmax(small_logits, dim=-1, keepdim=True)

                # After drafting, trigger prefetch if an Action was detected
                if self.enable_ste and draft_action_prefetch_expr:
                    normalized_draft_action_expr = self._normalize_tool_expr(draft_action_prefetch_expr)
                    if normalized_draft_action_expr and normalized_draft_action_expr != self.draft_action_prefetch_expr:
                        submitted = self._submit_prefetch_candidate(draft_action_prefetch_expr, source_stage="action")
                        if submitted:
                            self.draft_action_prefetch_expr = normalized_draft_action_expr

                # Verify drafted tokens with the large model
                draft_seq = torch.cat(draft_tokens, dim=-1).to(self.big_model.device)
                actual_draft_len = draft_seq.shape[1]
                total_draft += actual_draft_len
                
                v_pos = torch.arange(checkpoint_len, checkpoint_len + actual_draft_len, device=draft_seq.device).unsqueeze(0)
                verify_out = self.big_model(draft_seq, past_key_values=big_cache, position_ids=v_pos, use_cache=True)
                verify_logits = verify_out.logits
                
                n_matches = 0
                tool_triggered = False
                for i in range(actual_draft_len):
                    target_logits = big_logits if i == 0 else verify_logits[:, i-1, :]
                    correct_token = torch.argmax(target_logits, dim=-1, keepdim=True)
                    
                    if correct_token.item() == draft_tokens[i].item():
                        n_matches += 1
                        accepted_count += 1
                        t_val = correct_token.item()
                        input_id_chunks.append([t_val])
                        current_len += 1
                        
                        current_phase, active_buffer, latest_tool_plan_expr, tool_triggered = process_token(
                            t_val, current_phase, active_buffer, latest_tool_plan_expr
                        )
                        if self.stream_output:
                            print(self._decode_token(t_val), end="", flush=True)
                        
                        if t_val == self.eos_id or t_val == self.eot_id or tool_triggered:
                            break
                    else: 
                        break
                
                new_len = checkpoint_len + n_matches
                self._crop_cache(big_cache, new_len)
                self._crop_cache(small_cache, new_len)
                
                if current_len > start_len and input_id_chunks[-1][-1] in {self.eos_id, self.eot_id}: break
                
                if tool_triggered:
                    input_id_chunks, current_len, big_logits, small_logits = self._handle_tool_trigger(
                        active_buffer, input_id_chunks, current_len, big_cache, small_cache
                    )
                    current_phase = 0
                    active_buffer = ""
                    latest_tool_plan_expr = None
                    continue
                
                # Handle rejected tokens and resample
                f_logits = big_logits.clone() if n_matches == 0 else verify_logits[:, n_matches-1, :].clone()
                
                if n_matches < actual_draft_len:
                    rejected_id = draft_tokens[n_matches].item()
                    rejected_text = self._decode_token(rejected_id)
                    
                    if any(char.isdigit() for char in rejected_text):
                        f_logits[0, rejected_id] = -float('inf')
                    else:
                        alpha = 2 if current_phase == 0 else 6
                        lm_weights = self.big_model.get_output_embeddings().weight
                        sim = F.cosine_similarity(lm_weights, lm_weights[rejected_id].unsqueeze(0), dim=-1)
                        penalty = F.relu(sim) * alpha
                        f_logits -= penalty.unsqueeze(0)
                    
                    if torch.argmax(f_logits, dim=-1).item() == self.nl_id:
                        res_alpha = 1.5
                        if current_phase == 0: f_logits[0, self.base_action_id] += res_alpha
                        elif current_phase == 1: f_logits[0, self.base_obs_id] += res_alpha
                        elif current_phase == 2: f_logits[0, self.base_thought_id] += res_alpha
                        elif current_phase == 3: f_logits[0, self.base_action_id] += res_alpha

                final_correct = torch.argmax(f_logits, dim=-1, keepdim=True)
                t_val = final_correct.item()
                input_id_chunks.append([t_val])
                current_len += 1
                
                current_phase, active_buffer, latest_tool_plan_expr, tool_triggered = process_token(
                    t_val, current_phase, active_buffer, latest_tool_plan_expr
                )
                if self.stream_output:
                    print(self._decode_token(t_val), end="", flush=True)

                if t_val == self.eos_id or t_val == self.eot_id: break
                
                sync_pos = torch.tensor([[new_len]], device=final_correct.device)
                big_out = self.big_model(final_correct, past_key_values=big_cache, position_ids=sync_pos, use_cache=True)
                small_out = self.small_model(final_correct.to(self.small_model.device), past_key_values=small_cache, position_ids=sync_pos, use_cache=True)
                big_logits, small_logits = big_out.logits[:, -1, :], small_out.logits[:, -1, :]

                if tool_triggered:
                    input_id_chunks, current_len, big_logits, small_logits = self._handle_tool_trigger(
                        active_buffer, input_id_chunks, current_len, big_cache, small_cache
                    )
                    current_phase = 0
                    active_buffer = ""
                    latest_tool_plan_expr = None
                    continue

        dur = time.time() - start_time
        generated_token_ids = []
        consumed = 0
        for chunk in input_id_chunks:
            chunk_len = len(chunk)
            if consumed + chunk_len <= start_len:
                consumed += chunk_len
                continue
            start_idx = max(0, start_len - consumed)
            generated_token_ids.extend(chunk[start_idx:])
            consumed += chunk_len
        gen_text = self.tokenizer.decode(generated_token_ids, skip_special_tokens=False)
        gen_text = gen_text.replace("<|im_end|>", "").replace("<|eot_id|>", "").strip()
        generated_tokens = current_len - start_len
        pure_generation_time = max(dur - self.total_tool_time_sec, 1e-8)
        overall_tps = generated_tokens / dur if dur > 0 else 0.0
        pure_tps = generated_tokens / pure_generation_time if pure_generation_time > 0 else 0.0
        prefetch_hit_rate = (self.prefetch_stats["hit"] / self.tool_call_count) if self.tool_call_count > 0 else 0.0
        
        self.last_run_stats = {
            "duration": dur,
            "generated_tokens": generated_tokens,
            "accept_rate": (accepted_count/total_draft if total_draft > 0 else 0),
            "generated_text": gen_text,
            "overall_tps": overall_tps,
            "pure_tps": pure_tps,
            "tool_exec_time": self.total_tool_time_sec,
            "tool_call_count": self.tool_call_count,
            "prefetch_submit_count": self.prefetch_stats["submitted"],
            "prefetch_hit_count": self.prefetch_stats["hit"],
            "prefetch_hit_rate": prefetch_hit_rate,
            "prefetch_wait_time": self.prefetch_stats["prefetch_wait_time_ms"] / 1000.0,
            "prefetch_timeout_count": self.prefetch_stats["timeout"],
            "prefetch_saved_wait_ms": self.prefetch_stats["saved_wait_ms"],
            "prefetch_effective_hits": self.prefetch_stats["prefetch_effective_hits"],
            "action_async_submitted": self.prefetch_stats["action_async_submitted"],
            "action_hit": self.prefetch_stats["action_hit"],
            "compute_sync_fallback": self.prefetch_stats["compute_sync_fallback"],
            "action_plan_match": self.prefetch_stats["action_plan_match"],
            "action_plan_mismatch": self.prefetch_stats["action_plan_mismatch"],
            "hidden_probe_seen": self.prefetch_stats["hidden_probe_seen"],
            "hidden_probe_ready": self.prefetch_stats["hidden_probe_ready"],
            "hidden_probe_prefetch_triggered": self.prefetch_stats["hidden_probe_prefetch_triggered"],
            "hidden_probe_prefetch_duplicate": self.prefetch_stats["hidden_probe_prefetch_duplicate"],
            "hidden_probe_hit": self.prefetch_stats["hidden_probe_hit"],
            "hidden_probe_timeout": self.prefetch_stats["hidden_probe_timeout"],
            "hidden_probe_reuse": self.prefetch_stats["hidden_probe_reuse"],
            "prefetch_stats": dict(self.prefetch_stats),
        }
        return self.last_run_stats

# ==========================================
# Main evaluation entry
# ==========================================
def build_arg_parser():
    parser = argparse.ArgumentParser(description="Anonymous math evaluation runner.")
    parser.add_argument("--large-model-path", default=os.getenv("LARGE_MODEL_PATH", "./models/large_model"), help="Path to the large model directory.")
    parser.add_argument("--small-model-path", default=os.getenv("SMALL_MODEL_PATH", "./models/small_model"), help="Path to the small model directory.")
    parser.add_argument("--dataset-path", default=os.getenv("DATASET_PATH", "./data/math_eval.arrow"), help="Path to the evaluation dataset file.")
    parser.add_argument("--result-csv", default=os.getenv("RESULT_CSV_PATH", "evaluation_results.csv"), help="Output CSV path for evaluation results.")
    parser.add_argument("--threshold", type=float, default=float(os.getenv("THRESHOLD", "0.40")), help="Entropy threshold for switching between direct generation and speculative decoding.")
    parser.add_argument("--lookahead", type=int, default=int(os.getenv("LOOKAHEAD", "11")), help="Speculative decoding lookahead steps.")
    parser.add_argument("--enable-ste", dest="enable_ste", action="store_true", default=os.getenv("ENABLE_STE", "true").lower() in {"1", "true", "yes", "on"}, help="Enable asynchronous tool prefetching.")
    parser.add_argument("--disable-ste", dest="enable_ste", action="store_false", help="Disable asynchronous tool prefetching.")
    parser.add_argument("--ste-max-workers", type=int, default=int(os.getenv("STE_MAX_WORKERS", "2")), help="Maximum worker threads for tool prefetching.")
    parser.add_argument("--ste-min-expr-len", type=int, default=int(os.getenv("STE_MIN_EXPR_LEN", "6")), help="Minimum expression length required before prefetching.")
    parser.add_argument("--ste-wait-timeout", type=float, default=float(os.getenv("STE_WAIT_TIMEOUT", "5.0")), help="Timeout in seconds when waiting for prefetched tool results.")
    parser.add_argument("--ste-ttl-sec", type=float, default=float(os.getenv("STE_TTL_SEC", "15.0")), help="Time-to-live in seconds for prefetched tool payloads.")
    parser.add_argument("--hidden-probe-steps", type=int, default=int(os.getenv("HIDDEN_PROBE_STEPS", "12")), help="Number of hidden probe steps used before rollback.")
    parser.add_argument("--max-gen", type=int, default=int(os.getenv("MAX_GEN", "2048")), help="Maximum generated tokens per sample.")
    parser.add_argument("--sample-count", type=int, default=int(os.getenv("SAMPLE_COUNT", "10000")), help="Number of selected samples to evaluate.")
    parser.add_argument("--selection-pool", type=int, default=int(os.getenv("SELECTION_POOL", "1000")), help="Number of candidate samples to keep after ranking.")
    parser.add_argument("--stream-output", action="store_true", default=os.getenv("STREAM_OUTPUT", "false").lower() in {"1", "true", "yes", "on"}, help="Stream decoded model output to stdout.")
    return parser


def main():
    args = build_arg_parser().parse_args()

    engine = ReActSpeculativeEngine(
        big_path=args.large_model_path,
        small_path=args.small_model_path,
        threshold=args.threshold,
        lookahead=args.lookahead,
        enable_ste=args.enable_ste,
        ste_max_workers=args.ste_max_workers,
        ste_min_expr_len=args.ste_min_expr_len,
        ste_wait_timeout=args.ste_wait_timeout,
        ste_ttl_sec=args.ste_ttl_sec,
        hidden_probe_steps=args.hidden_probe_steps,
    )
    engine.stream_output = args.stream_output
    engine.log_csv = args.result_csv
    
    print(">>> Loading dataset from configured path...")
    full_dataset = Dataset.from_file(args.dataset_path)
    
    print(">>> Prioritizing Level 4 and Level 5 problems; backfilling with lower levels if needed...")

    def get_level_score(example):
        level_raw = str(example.get('level', '')).strip()
        digits = ''.join(ch for ch in level_raw if ch.isdigit())
        if digits:
            try:
                return int(digits)
            except ValueError:
                pass
        return -1

    candidate_entries = []
    fallback_entries = []
    seen_questions = set()

    for idx, entry in enumerate(full_dataset):
        question_text = entry.get('problem', entry.get('question', ''))
        if question_text in seen_questions:
            continue
        seen_questions.add(question_text)

        level_score = get_level_score(entry)
        if level_score >= 4:
            candidate_entries.append((idx, level_score, question_text))
        elif level_score >= 0:
            fallback_entries.append((idx, level_score, question_text))

    candidate_entries.sort(key=lambda item: (-item[1], -len(item[2]), item[0]))
    fallback_entries.sort(key=lambda item: (-item[1], -len(item[2]), item[0]))

    selected_indices = [idx for idx, _, _ in candidate_entries[:args.selection_pool]]

    if len(selected_indices) < args.selection_pool:
        remain = args.selection_pool - len(selected_indices)
        print(f">>> Fewer than {args.selection_pool} Level 4/5 items found; backfilling with {remain} lower-level samples.")
        selected_indices.extend(idx for idx, _, _ in fallback_entries[:remain])

    if not selected_indices:
        print(">>> No valid level field found; falling back to ranking by problem length...")
        ranked_by_length = sorted(
            enumerate(full_dataset),
            key=lambda item: len(item[1].get('problem', item[1].get('question', ''))),
            reverse=True
        )
        selected_indices = [idx for idx, _ in ranked_by_length[:args.selection_pool]]

    num_test = min(args.sample_count, len(selected_indices))
    test_data = full_dataset.select(selected_indices[:num_test])
    print(f">>> Selection complete. Preparing to evaluate {num_test} samples.")
    
    results = []
    correct_count = 0
    
    try:
        for i, entry in enumerate(test_data):
            question = entry.get('problem', entry.get('question', ""))
            reference_full = str(entry.get('solution', entry.get('solution', "")))
            
            print(f"\n\n[Progress] Evaluating sample {i+1}/{num_test}")
            
            try:
                case_stats = engine.run_speculative(question, max_gen=args.max_gen)
                
                pred_ans = extract_boxed_content(case_stats["generated_text"])
                ref_ans = extract_boxed_content(reference_full) if "\\boxed" in reference_full else reference_full.strip()
                
                is_hit = is_equivalent(pred_ans, ref_ans)
                if is_hit: correct_count += 1
                    
                print(f"\n[Evaluation] Predicted answer: {pred_ans} | Reference answer: {ref_ans} | Correct: {'yes' if is_hit else 'no'}")
                print(
                    "[Case Metrics] "
                    f"duration={case_stats['duration']:.2f}s | "
                    f"total_tokens={case_stats['generated_tokens']} | "
                    f"tps={case_stats['overall_tps']:.2f} | "
                    f"pure_tps={case_stats['pure_tps']:.2f} | "
                    f"tool_exec_time={case_stats['tool_exec_time']:.2f}s | "
                    f"tool_calls={case_stats['tool_call_count']} | "
                    f"prefetch_submitted={case_stats['prefetch_submit_count']} | "
                    f"prefetch_hit={case_stats['prefetch_hit_count']} | "
                    f"prefetch_hit_rate={case_stats['prefetch_hit_rate']:.2%} | "
                    f"prefetch_wait_time={case_stats['prefetch_wait_time']:.4f}s | "
                    f"prefetch_timeout={case_stats['prefetch_timeout_count']} | "
                    f"prefetch_saved_wait_ms={case_stats['prefetch_saved_wait_ms']:.2f} | "
                    f"hidden_probe_hit={case_stats['hidden_probe_hit']} | "
                    f"hidden_probe_timeout={case_stats['hidden_probe_timeout']} | "
                    f"action_submit={case_stats['action_async_submitted']} | "
                    f"action_hit={case_stats['action_hit']} | "
                    f"sync_fallback={case_stats['compute_sync_fallback']} | "
                    f"accept_rate={case_stats['accept_rate']:.2%}"
                )
                
                res_entry = {
                    "id": i,
                    "question": question,
                    "ref_ans": ref_ans,
                    "pred_ans": pred_ans,
                    "is_correct": is_hit,
                    "duration": round(case_stats["duration"], 2),
                    "total_tokens": case_stats["generated_tokens"],
                    "tokens_per_sec": round(case_stats["overall_tps"], 2),
                    "pure_tokens_per_sec": round(case_stats["pure_tps"], 2),
                    "tool_exec_time": round(case_stats["tool_exec_time"], 2),
                    "tool_call_count": case_stats["tool_call_count"],
                    "prefetch_submit_count": case_stats["prefetch_submit_count"],
                    "prefetch_hit_count": case_stats["prefetch_hit_count"],
                    "prefetch_hit_rate": round(case_stats["prefetch_hit_rate"], 4),
                    "prefetch_wait_time": round(case_stats["prefetch_wait_time"], 4),
                    "prefetch_timeout_count": case_stats["prefetch_timeout_count"],
                    "prefetch_saved_wait_ms": round(case_stats["prefetch_saved_wait_ms"], 2),
                    "prefetch_effective_hits": case_stats["prefetch_effective_hits"],
                    "hidden_probe_hit": case_stats["hidden_probe_hit"],
                    "hidden_probe_timeout": case_stats["hidden_probe_timeout"],
                    "hidden_probe_reuse": case_stats["hidden_probe_reuse"],
                    "action_async_submitted": case_stats["action_async_submitted"],
                    "action_hit": case_stats["action_hit"],
                    "compute_sync_fallback": case_stats["compute_sync_fallback"],
                    "action_plan_match": case_stats["action_plan_match"],
                    "action_plan_mismatch": case_stats["action_plan_mismatch"],
                    "accept_rate": round(case_stats["accept_rate"], 4)
                }
                results.append(res_entry)
            except Exception as e:
                print(f"Error processing case {i}: {e}")
                import traceback
                traceback.print_exc()
                continue
                
        if results:
            df = pd.DataFrame(results)
            df.to_csv(engine.log_csv, index=False, encoding='utf-8-sig')
            print(f"\n{'#'*20} Evaluation Complete {'#'*20}")
            print(f"Average TPS (wall-clock): {df['tokens_per_sec'].mean():.2f}")
            print(f"Average TPS (excluding tool wait): {df['pure_tokens_per_sec'].mean():.2f}")
            print(f"Total tool calls: {int(df['tool_call_count'].sum())}")
            print(f"Prefetch submissions: {int(df['prefetch_submit_count'].sum())}")
            print(f"Prefetch hits: {int(df['prefetch_hit_count'].sum())}")
            print(f"Prefetch hit rate (per tool call): {(df['prefetch_hit_count'].sum() / df['tool_call_count'].sum()):.2%}" if df['tool_call_count'].sum() > 0 else "Prefetch hit rate (per tool call): 0.00%")
            print(f"Total prefetch wait time: {df['prefetch_wait_time'].sum():.4f}s")
            print(f"Prefetch timeouts: {int(df['prefetch_timeout_count'].sum())}")
            print(f"Prefetch saved wait time: {df['prefetch_saved_wait_ms'].sum():.2f} ms")
            print(f"Hidden probe hits: {int(df['hidden_probe_hit'].sum())}")
            print(f"Hidden probe timeouts: {int(df['hidden_probe_timeout'].sum())}")
            print(f"Hidden probe reuse count: {int(df['hidden_probe_reuse'].sum())}")
            print(f"Action async submissions: {int(df['action_async_submitted'].sum())}")
            print(f"Action hits: {int(df['action_hit'].sum())}")
            print(f"Synchronous fallbacks: {int(df['compute_sync_fallback'].sum())}")
            print(f"Average small-model acceptance rate: {df['accept_rate'].mean():.2%}")
            print(f"Final accuracy: {correct_count / len(results):.2%} ({correct_count}/{len(results)})")
            print(f"Results saved to: {engine.log_csv}")
    finally:
        engine.close()

if __name__ == "__main__":
    main()
