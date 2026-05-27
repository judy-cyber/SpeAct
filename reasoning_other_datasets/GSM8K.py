import argparse
import re
import time
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, TimeoutError

import pandas as pd
import sympy
import torch
import torch.nn.functional as F
from awq import AutoAWQForCausalLM
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache


def extract_boxed_content(text):
    match = re.search(r"\\boxed\{([^}]*)\}", text)
    if match:
        return match.group(1).strip()
    return None


def is_equivalent(pred, ref):
    if pred is None or ref is None:
        return False

    pred = str(pred).strip()
    ref = str(ref).strip()
    if not pred or not ref:
        return False

    import sympy
    from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application

    def normalize_math_str(s):
        s = s.strip().lower()
        s = s.replace("$", "").replace("\\(", "").replace("\\)", "")
        s = s.replace("\\frac{", "(").replace("}{", ")/(").replace("}", ")")
        s = s.replace("\\sqrt{", "sqrt(").replace("}", ")")
        s = s.replace("\\sqrt", "sqrt")
        s = s.replace("\\pi", "pi").replace("\\infty", "oo")
        s = s.replace("^", "**").replace("{", "(").replace("}", ")")
        s = s.replace("\\cdot", "*").replace("\\times", "*")
        s = s.replace("\\", "")
        return s

    pred_norm = normalize_math_str(pred)
    ref_norm = normalize_math_str(ref)
    if pred_norm == ref_norm:
        return True

    def try_parse_to_sympy(raw, normed):
        transformations = standard_transformations + (implicit_multiplication_application,)
        try:
            from sympy.parsing.latex import parse_latex
            return parse_latex(raw)
        except Exception:
            pass
        try:
            return parse_expr(normed, transformations=transformations)
        except Exception:
            return None

    p_expr = try_parse_to_sympy(pred, pred_norm)
    r_expr = try_parse_to_sympy(ref, ref_norm)

    if p_expr is not None and r_expr is not None:
        try:
            if p_expr.equals(r_expr):
                return True
        except Exception:
            pass

    def extract_and_eval(s):
        try:
            return float(sympy.sympify(s).evalf())
        except Exception:
            return None

    p_val = extract_and_eval(pred_norm)
    r_val = extract_and_eval(ref_norm)
    if p_val is not None and r_val is not None:
        if abs(p_val - r_val) < 1e-6:
            return True

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

        print(">>> Loading primary model...")
        self.big_model_wrapper = AutoAWQForCausalLM.from_quantized(
            big_path, fuse_layers=True, trust_remote_code=True, device_map="auto"
        )
        self.big_model = self.big_model_wrapper.model

        print(">>> Loading draft model...")
        self.small_model = AutoModelForCausalLM.from_pretrained(
            small_path, device_map="auto", torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
        )

        self.tokenizer = AutoTokenizer.from_pretrained(big_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.encode_cache = {}
        self.decode_cache = {}
        self.nl_id = self._encode_text("\n")[-1]
        self.eos_id = self.tokenizer.eos_token_id

        self.action_ids = {self._encode_text(w)[-1] for w in ["Action", " Action", "\nAction"]}
        self.obs_ids = {self._encode_text(w)[-1] for w in ["Observation", " Observation", "\nObservation"]}
        self.thought_ids = {self._encode_text(w)[-1] for w in ["Thought", " Thought", "\nThought"]}

        self.base_action_id = self._encode_text("Action")[-1]
        self.base_obs_id = self._encode_text("Observation")[-1]
        self.base_thought_id = self._encode_text("Thought")[-1]

        self.log_csv = "final_integrated_results.csv"
        self.ste_executor = ThreadPoolExecutor(max_workers=ste_max_workers) if self.enable_ste else None
        self.pending_tool_prefetch = None
        self.hidden_probe_expr = None
        self.action_prefetch = None
        self.prefetch_stats = {}
        self.last_run_stats = {}
        self.total_tool_time_sec = 0.0
        self._reset_prefetch_state()

        self.stream_output = True

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
            "action_without_prefetch": 0,
            "action_prefetch_match": 0,
            "action_prefetch_mismatch": 0,
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
        if token_val in self.action_ids:
            return 1
        elif token_val in self.obs_ids:
            return 2
        elif token_val in self.thought_ids:
            return 0
        return current_phase

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

    def _prefetch_min_expr_len(self):
        return max(1, int(self.ste_min_expr_len))

    def _is_prefetch_expr_ready(self, expr_str):
        normalized_expr = self._normalize_tool_expr(expr_str)
        if not normalized_expr:
            return False
        if len(normalized_expr) < self._prefetch_min_expr_len():
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
        min_expr_len = self._prefetch_min_expr_len()
        if not normalized_expr or len(normalized_expr) < min_expr_len:
            return False

        if source_stage == "action" and not self._is_prefetch_expr_ready(expr_str):
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
        self.prefetch_stats["submitted"] += 1
        self.prefetch_stats["compute_submitted"] += 1
        if source_stage == "hidden_probe":
            self.hidden_probe_expr = normalized_expr
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

    def _maybe_prefetch_from_draft(self, token_id, draft_phase, draft_buffer, draft_thought_expr, draft_latest_plan_expr, prefetch_allowed):
        token_text = self._decode_token(token_id)
        draft_phase = self._update_phase(token_id, draft_phase)
        if draft_phase == 0 and "Calculate[" not in draft_buffer and len(draft_buffer) > 256:
            draft_buffer = draft_buffer[-256:]
        draft_buffer = (draft_buffer + token_text)[-4000:]

        if not prefetch_allowed or draft_phase != 1:
            return draft_phase, draft_buffer, draft_thought_expr, draft_latest_plan_expr, False

        if "Calculate[" not in draft_buffer:
            return draft_phase, draft_buffer, draft_thought_expr, draft_latest_plan_expr, False

        draft_expr, is_closed = self._extract_calculate_expr(draft_buffer)
        if draft_expr is None:
            return draft_phase, draft_buffer, draft_thought_expr, draft_latest_plan_expr, False

        cleaned_draft_expr = self._normalize_tool_expr(draft_expr)
        if not cleaned_draft_expr:
            return draft_phase, draft_buffer, draft_thought_expr, draft_latest_plan_expr, False

        if is_closed or self._is_prefetch_expr_ready(cleaned_draft_expr):
            self.prefetch_stats["hidden_probe_seen"] += 1
            self.prefetch_stats["hidden_probe_ready"] += 1
            if cleaned_draft_expr != self.hidden_probe_expr:
                submitted = self._submit_prefetch_candidate(cleaned_draft_expr, "hidden_probe")
                if submitted:
                    draft_thought_expr = cleaned_draft_expr
                    draft_latest_plan_expr = cleaned_draft_expr
                else:
                    self.prefetch_stats["hidden_probe_prefetch_duplicate"] += 1

        return draft_phase, draft_buffer, draft_thought_expr, draft_latest_plan_expr, False

    def _preprocess_tool_expr(self, expr_str):
        expr_str = expr_str.strip()
        expr_str = expr_str.replace('×', '*').replace('÷', '/')
        expr_str = expr_str.replace('−', '-')
        expr_str = expr_str.replace('\\cdot', '*')
        expr_str = expr_str.replace('^', '**')
        expr_str = re.sub(r"\s+", " ", expr_str)
        return expr_str

    def _safe_eval_expr(self, expr_str):
        local_dict = {
            "sqrt": sympy.sqrt,
            "abs": sympy.Abs,
            "pi": sympy.pi,
        }
        transformations = sympy.parsing.sympy_parser.standard_transformations + (
            sympy.parsing.sympy_parser.implicit_multiplication_application,
        )
        return sympy.parsing.sympy_parser.parse_expr(
            expr_str,
            local_dict=local_dict,
            transformations=transformations,
            evaluate=True,
        )

    def _extract_target_variable(self, expr_str):
        match = re.search(r"\bfor\s+([a-zA-Z_]\w*)\s*$", expr_str)
        return match.group(1) if match else None

    def _internal_calculate(self, expr_str):
        normalized_expr = self._preprocess_tool_expr(expr_str)
        lowered_expr = normalized_expr.lower()

        try:
            if lowered_expr.startswith("solve "):
                raw_body = normalized_expr[6:].strip()
                target_var = self._extract_target_variable(raw_body)
                if not target_var:
                    return {"status": "error", "kind": "tool", "answer": "Missing target variable in solve expression."}

                equation_part = re.sub(r"\bfor\s+[a-zA-Z_]\w*\s*$", "", raw_body).strip()
                if "=" not in equation_part:
                    return {"status": "error", "kind": "tool", "answer": "Solve expression must contain '='."}

                left_raw, right_raw = equation_part.split("=", 1)
                target_symbol = sympy.Symbol(target_var)
                left_expr = self._safe_eval_expr(left_raw.strip())
                right_expr = self._safe_eval_expr(right_raw.strip())
                solutions = sympy.solve(sympy.Eq(left_expr, right_expr), target_symbol)

                if not isinstance(solutions, (list, tuple)):
                    solutions = [solutions]
                if len(solutions) == 0:
                    return {"status": "ok", "kind": "solution", "answer": "No solution"}
                if len(solutions) == 1:
                    return {"status": "ok", "kind": "solution", "answer": f"{target_var} = {sympy.sstr(solutions[0])}"}
                joined = ", ".join(sympy.sstr(item) for item in solutions)
                return {"status": "ok", "kind": "solution", "answer": f"{target_var} in {{{joined}}}"}

            parsed_expr = self._safe_eval_expr(normalized_expr)
            simplified_expr = sympy.simplify(parsed_expr)

            if simplified_expr.free_symbols:
                return {"status": "ok", "kind": "expression", "answer": sympy.sstr(simplified_expr)}

            numeric_value = sympy.N(simplified_expr, 16)
            if getattr(numeric_value, "is_Integer", False):
                answer = str(int(numeric_value))
            elif getattr(numeric_value, "is_Rational", False):
                answer = sympy.sstr(simplified_expr)
            else:
                answer = sympy.sstr(sympy.nsimplify(numeric_value))
                if answer in {"zoo", "nan"}:
                    answer = sympy.sstr(numeric_value)
            return {"status": "ok", "kind": "result", "answer": answer}
        except Exception as e:
            return {"status": "error", "kind": "tool", "answer": str(e)}

    def _format_internal_observation(self, expr_str, tool_result):
        status = tool_result.get("status", "ok")
        kind = tool_result.get("kind", "result")
        answer_text = tool_result.get("answer", "")

        if status != "ok":
            return (
                f"Observation: [ToolError]: {answer_text}\n"
                "Thought: Re-check the arithmetic expression. Use plain arithmetic like Calculate[ 24 * 15 ] or a simple solve format like Calculate[ solve 2000 = 400 * d for d ].\n"
            )

        if kind == "solution":
            prefix = "[Solution]"
            thought = "Use the solved value directly and continue only if another arithmetic step is still needed."
        elif kind == "expression":
            prefix = "[Simplified]"
            thought = "Use the simplified expression above to finish the remaining arithmetic."
        else:
            prefix = "[Result]"
            thought = "Use this arithmetic result directly in the final answer."

        return f"Observation: {prefix}: {answer_text}\nThought: {thought}\n"

    def _execute_tool(self, action_buffer, session=None):
        expr_str, is_closed = self._extract_calculate_expr(action_buffer)
        if expr_str is None:
            expr_str = action_buffer.replace("Action:", "").replace("Action", "").replace("\n", "").strip()

        if not expr_str:
            return "Observation: Error: Empty calculation. Think first, then call the calculator with one complete expression.\nThought: "

        if not is_closed and "Calculate[" in action_buffer:
            return "Observation: Error: Incomplete Calculate[...] block. Finish the full expression before using the tool.\nThought: "

        tool_result = self._internal_calculate(expr_str)
        if tool_result.get("status") == "ok":
            self.no_solution_count, self.error_count = 0, 0
            return self._format_internal_observation(expr_str, tool_result)

        self.error_count += 1
        if self.error_count >= 3:
            self.error_count = 0
            return "Observation: FATAL ERROR: Repeated malformed internal calculator calls. Stop using the calculator and reason manually.\nThought: "
        return self._format_internal_observation(expr_str, tool_result)

    def _execute_tool_query(self, expr_str):
        self.prefetch_stats["prefetch_opportunities"] += 1
        prefetched = self._consume_prefetch_if_match(expr_str)
        if prefetched is not None:
            self.prefetch_stats["compute_hit"] += 1
            return prefetched
        self.prefetch_stats["compute_sync_fallback"] += 1
        return self._prepare_tool_payload(expr_str, source="sync")

    def _inject_tool_result(self, tool_obs_str, input_ids, current_len, big_cache, small_cache):
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

    @torch.no_grad()
    def run_speculative(self, question, max_gen=1024):
        self.no_solution_count = 0
        self.error_count = 0
        self.repeat_action_count = 0
        self.repeat_action_interrupt_count = 0
        self.tool_call_count = 0
        self.expr_history = []
        self._reset_prefetch_state()

        few_shot_examples = (
            "Here are examples of solving GSM8K-style math problems with careful reasoning. Use the calculator mainly for arithmetic and short one-variable solves.\n\n"
            "--- Example 1: Count remaining vouchers ---\n"
            "Question: John and Mary each start with 20 vouchers. John uses 3, Mary uses 7, then John uses 4 more and Mary uses 5 more. How many vouchers remain in total?\n"
            "Thought: Total vouchers initially are 20 + 20 = 40. Total used are 3 + 7 + 4 + 5. I will calculate the remaining vouchers.\n"
            "Action: Calculate[ 40 - (3 + 7 + 4 + 5) ]\n"
            "Observation: [Result]: 21\n"
            "Thought: So the total remaining vouchers is 21.\n"
            "Final Answer: \\boxed{21}\n\n"
            "--- Example 2: Price times quantity ---\n"
            "Question: There are 4 boys and 8 girls in a dance class. Each student needs 2 pairs of shoes, and each pair costs 15 dollars. How much money is needed in total?\n"
            "Thought: There are 4 + 8 = 12 students. They need 12 * 2 = 24 pairs of shoes. I will calculate the total cost.\n"
            "Action: Calculate[ 24 * 15 ]\n"
            "Observation: [Result]: 360\n"
            "Thought: The total cost is 360 dollars.\n"
            "Final Answer: \\boxed{360}\n\n"
            "--- Example 3: Daily rate word problem ---\n"
            "Question: A crew installs 50 shingles every hour and works 8 hours a day. If a roof needs 2000 shingles, how many days will the crew need?\n"
            "Thought: The crew installs 50 * 8 = 400 shingles per day. I will solve 2000 = 400 * d for d.\n"
            "Action: Calculate[ solve 2000 = 400 * d for d ]\n"
            "Observation: [Solution]: d = 5\n"
            "Thought: The crew needs 5 days.\n"
            "Final Answer: \\boxed{5}\n\n"
            "--- Example 4: Simple system reduced to one variable ---\n"
            "Question: A basket has apples and oranges. There are 24 more apples than oranges, and there are 60 fruits total. How many oranges are there?\n"
            "Thought: If oranges are o, then apples are o + 24. The total becomes (o + 24) + o = 60. I will solve for o.\n"
            "Action: Calculate[ solve (o + 24) + o = 60 for o ]\n"
            "Observation: [Solution]: o = 18\n"
            "Thought: So there are 18 oranges.\n"
            "Final Answer: \\boxed{18}\n\n"
        )
        system_content = (
            "You are an elite mathematical reasoning engine for GSM8K-style arithmetic word problems. Your primary goal is absolute accuracy. "
            "Use the internal calculator to avoid arithmetic mistakes, especially in multi-step numeric computation.\n\n"
            "### INTERNAL CALCULATOR SYNTAX RULES ###\n"
            "1. For arithmetic expressions, call the tool with plain expressions such as `Calculate[ 24 * 15 ]`.\n"
            "2. For a one-variable equation, use `Calculate[ solve 2000 = 400 * d for d ]`.\n"
            "3. Use `*` for multiplication, `/` for division, and parentheses when needed.\n"
            "4. Keep expressions compact, explicit, and machine-readable.\n"
            "5. Prefer direct arithmetic over symbolic manipulation whenever possible.\n\n"
            "### DECISION POLICY ###\n"
            "1. First identify the quantities in the story and the target being asked.\n"
            "2. Break the problem into short arithmetic steps.\n"
            "3. Use the calculator for the final arithmetic step or for a short one-variable solve when needed.\n"
            "4. Avoid unnecessary symbolic algebra such as factoring or expression rewriting unless it is directly needed.\n"
            "5. Keep thoughts short, concrete, and tied to the numbers in the problem.\n\n"
            "### TOOL RULES ###\n"
            "1. Use the standard ReAct order: Thought -> Action -> Observation.\n"
            "2. Allowed: Calculate[ <expr> ] for arithmetic.\n"
            "3. Allowed: Calculate[ solve <eq> for <var> ] for a short one-variable solve.\n"
            "4. Keep each calculation expression concise, explicit, and machine-readable.\n"
            "5. If the calculator returns [ToolError], rewrite the expression in a simpler valid form and try again.\n\n"
            "### THOUGHT DISCIPLINE ###\n"
            "Before every Action, briefly explain what quantity or value must be computed next.\n"
            "Keep Thought short, concrete, and action-guiding.\n"
            "Do not output any hidden planning line such as ToolPlan.\n"
            "Always end your reasoning with Final Answer: \\boxed{result}. Do not generate any extra text after the final answer.\n\n"
            f"{few_shot_examples}"
        )

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": f"Question: {question}\nSolve this step-by-step using the ReAct format. Begin with 'Thought: ' and think carefully before any Action."}
        ]

        prompt_build_start = time.time()
        prompt_str = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_ids = self.tokenizer.encode(prompt_str, add_special_tokens=False)
        thought_prefix_ids = self.tokenizer.encode("Thought: ", add_special_tokens=False)

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
        prompt_tokens = len(prompt_ids) + len(thought_prefix_ids)
        prompt_build_time = time.time() - prompt_build_start
        current_phase = 0
        high_entropy_steps = 0
        low_entropy_steps = 0
        rejected_tokens_count = 0
        latest_prefetch_expr = None
        current_prefetch_expr = None
        active_buffer = ""

        def process_token(t_val, phase, buffer, thought_prefetch_expr, latest_prefetch_expr, allow_trigger=True):
            text_chunk = self._decode_token(t_val)
            new_phase = self._update_phase(t_val, phase)
            if new_phase == 0 and "Calculate[" not in buffer and len(buffer) > 256:
                buffer = buffer[-256:]
            buffer = (buffer + text_chunk)[-4000:]

            triggered = False

            if "Calculate[" in buffer:
                extracted_expr, is_closed = self._extract_calculate_expr(buffer)
                if extracted_expr is not None and is_closed:
                    self.prefetch_stats["action_seen"] += 1
                    self.prefetch_stats["action_closed"] += 1
                    normalized_action_expr = self._normalize_tool_expr(extracted_expr)
                    if latest_prefetch_expr:
                        if latest_prefetch_expr == normalized_action_expr:
                            self.prefetch_stats["action_prefetch_match"] += 1
                        else:
                            self.prefetch_stats["action_prefetch_mismatch"] += 1
                    else:
                        self.prefetch_stats["action_without_prefetch"] += 1
                    thought_prefetch_expr = normalized_action_expr
                    latest_prefetch_expr = normalized_action_expr
                    if allow_trigger:
                        triggered = True

            return new_phase, buffer, thought_prefetch_expr, latest_prefetch_expr, triggered

        while (current_len - start_len) < max_gen:
            probs = F.softmax(big_logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1).item()
            checkpoint_len = big_cache.get_seq_length()

            if entropy > self.threshold:
                # High-entropy step: let the primary model decode directly.
                high_entropy_steps += 1
                next_token = torch.argmax(big_logits, dim=-1, keepdim=True)
                t_val = next_token.item()
                input_id_chunks.append([t_val])
                current_len += 1

                current_phase, active_buffer, current_prefetch_expr, latest_prefetch_expr, tool_triggered = process_token(
                    t_val, current_phase, active_buffer, current_prefetch_expr, latest_prefetch_expr
                )
                if self.stream_output:
                    print(self._decode_token(t_val), end="", flush=True)

                if t_val == self.eos_id:
                    break

                pos = torch.tensor([[checkpoint_len]], device=next_token.device)
                big_out = self.big_model(next_token, past_key_values=big_cache, position_ids=pos, use_cache=True)
                small_out = self.small_model(next_token.to(self.small_model.device), past_key_values=small_cache, position_ids=pos, use_cache=True)
                big_logits, small_logits = big_out.logits[:, -1, :], small_out.logits[:, -1, :]

                if tool_triggered:
                    input_id_chunks, current_len, big_logits, small_logits = self._handle_tool_trigger(
                        active_buffer, input_id_chunks, current_len, big_cache, small_cache
                    )
                    current_phase, active_buffer, current_prefetch_expr, latest_prefetch_expr = 0, "", None, None
                    continue
            else:
                # Low-entropy step: draft with the small model, then verify with the primary model.
                low_entropy_steps += 1
                draft_tokens = []
                draft_phase = current_phase
                draft_action_buffer = active_buffer
                draft_prefetch_expr = current_prefetch_expr
                draft_latest_prefetch_expr = latest_prefetch_expr
                draft_action_prefetch_expr = None

                hidden_probe_allowed = current_phase == 1
                temp_input = torch.argmax(small_logits, dim=-1, keepdim=True)

                for i in range(self.lookahead):
                    if temp_input.item() == self.nl_id:
                        boost = 7
                        if current_phase == 0:
                            small_logits[0, self.base_action_id] += boost
                        elif current_phase == 1:
                            small_logits[0, self.base_obs_id] += boost
                        elif current_phase == 2:
                            small_logits[0, self.base_thought_id] += boost
                        temp_input = torch.argmax(small_logits, dim=-1, keepdim=True)

                    draft_tokens.append(temp_input)

                    draft_phase, draft_action_buffer, draft_prefetch_expr, draft_latest_prefetch_expr, _ = self._maybe_prefetch_from_draft(
                        temp_input.item(),
                        draft_phase,
                        draft_action_buffer,
                        draft_prefetch_expr,
                        draft_latest_prefetch_expr,
                        hidden_probe_allowed,
                    )

                    if self.enable_ste and draft_phase == 1 and "Calculate[" in draft_action_buffer:
                        extracted_draft_expr, draft_is_closed = self._extract_calculate_expr(draft_action_buffer)
                        if extracted_draft_expr is not None and draft_is_closed:
                            draft_action_prefetch_expr = extracted_draft_expr
                            break

                    if temp_input.item() == self.eos_id:
                        break

                    p_id = torch.tensor([[checkpoint_len + i]], device=temp_input.device)
                    s_out = self.small_model(temp_input, past_key_values=small_cache, position_ids=p_id, use_cache=True)
                    small_logits = s_out.logits[:, -1, :]
                    temp_input = torch.argmax(small_logits, dim=-1, keepdim=True)

                # Trigger action-side prefetch if the drafted span already closes a calculator call.
                if self.enable_ste and draft_action_prefetch_expr:
                    normalized_draft_action_expr = self._normalize_tool_expr(draft_action_prefetch_expr)
                    if normalized_draft_action_expr and normalized_draft_action_expr != self.draft_action_prefetch_expr:
                        submitted = self._submit_prefetch_candidate(draft_action_prefetch_expr, source_stage="action")
                        if submitted:
                            self.draft_action_prefetch_expr = normalized_draft_action_expr

                # Verify the drafted tokens with the primary model.
                draft_seq = torch.cat(draft_tokens, dim=-1).to(self.big_model.device)
                actual_draft_len = draft_seq.shape[1]
                total_draft += actual_draft_len

                v_pos = torch.arange(checkpoint_len, checkpoint_len + actual_draft_len, device=draft_seq.device).unsqueeze(0)
                verify_out = self.big_model(draft_seq, past_key_values=big_cache, position_ids=v_pos, use_cache=True)
                verify_logits = verify_out.logits

                n_matches = 0
                tool_triggered = False
                for i in range(actual_draft_len):
                    target_logits = big_logits if i == 0 else verify_logits[:, i - 1, :]
                    correct_token = torch.argmax(target_logits, dim=-1, keepdim=True)

                    if correct_token.item() == draft_tokens[i].item():
                        n_matches += 1
                        accepted_count += 1
                        t_val = correct_token.item()
                        input_id_chunks.append([t_val])
                        current_len += 1

                        current_phase, active_buffer, current_prefetch_expr, latest_prefetch_expr, tool_triggered = process_token(
                            t_val, current_phase, active_buffer, current_prefetch_expr, latest_prefetch_expr
                        )
                        if self.stream_output:
                            print(self._decode_token(t_val), end="", flush=True)

                        if t_val == self.eos_id or tool_triggered:
                            break
                    else:
                        rejected_tokens_count += 1
                        break

                if tool_triggered:
                    input_id_chunks, current_len, big_logits, small_logits = self._handle_tool_trigger(
                        active_buffer, input_id_chunks, current_len, big_cache, small_cache
                    )
                    current_phase, active_buffer, current_prefetch_expr, latest_prefetch_expr = 0, "", None, None
                    continue

                self._crop_cache(big_cache, checkpoint_len + n_matches)
                self._crop_cache(small_cache, checkpoint_len + n_matches)

                if n_matches < actual_draft_len:
                    corr_token = torch.argmax(verify_logits[:, n_matches, :], dim=-1, keepdim=True)
                    t_val = corr_token.item()
                    input_id_chunks.append([t_val])
                    current_len += 1
                    if self.stream_output:
                        print(self._decode_token(t_val), end="", flush=True)
                    current_phase, active_buffer, current_prefetch_expr, latest_prefetch_expr, tool_triggered = process_token(
                        t_val, current_phase, active_buffer, current_prefetch_expr, latest_prefetch_expr
                    )
                    pos = torch.tensor([[checkpoint_len + n_matches]], device=corr_token.device)
                    big_out = self.big_model(corr_token, past_key_values=big_cache, position_ids=pos, use_cache=True)
                    small_out = self.small_model(corr_token.to(self.small_model.device), past_key_values=small_cache, position_ids=pos, use_cache=True)
                    big_logits, small_logits = big_out.logits[:, -1, :], small_out.logits[:, -1, :]
                    if tool_triggered:
                        input_id_chunks, current_len, big_logits, small_logits = self._handle_tool_trigger(
                            active_buffer, input_id_chunks, current_len, big_cache, small_cache
                        )
                        current_phase, active_buffer, current_prefetch_expr, latest_prefetch_expr = 0, "", None, None
                        continue
                else:
                    big_logits = verify_logits[:, -1, :]
                    if n_matches > 0:
                        accepted_seq = draft_seq[:, :n_matches].to(self.small_model.device)
                        p_pos = torch.arange(checkpoint_len, checkpoint_len + n_matches, device=accepted_seq.device).unsqueeze(0)
                        small_out = self.small_model(accepted_seq, past_key_values=small_cache, position_ids=p_pos, use_cache=True)
                        small_logits = small_out.logits[:, -1, :]

                if t_val == self.eos_id:
                    break

        duration = time.time() - start_time
        generated_tokens = current_len - start_len
        text_out = self.tokenizer.decode(sum(input_id_chunks[1:], []), skip_special_tokens=True)
        pure_tps = generated_tokens / max(duration - self.total_tool_time_sec, 1e-6)
        accept_rate = accepted_count / total_draft if total_draft > 0 else 0.0

        prefetch_hit_count = int(self.prefetch_stats.get("hit", 0))
        prefetch_submit_count = int(self.prefetch_stats.get("submitted", 0))
        prefetch_miss_count = int(self.prefetch_stats.get("miss", 0))
        prefetch_timeout_count = int(self.prefetch_stats.get("timeout", 0))
        prefetch_wait_time = float(self.prefetch_stats.get("prefetch_wait_time_ms", 0.0)) / 1000.0
        prefetch_saved_time = float(self.prefetch_stats.get("saved_wait_ms", 0.0)) / 1000.0
        prefetch_ready_hit_count = int(self.prefetch_stats.get("prefetch_effective_hits", 0))
        prefetch_ready_count = int(self.prefetch_stats.get("hidden_probe_ready", 0))
        avg_tool_exec_time = self.total_tool_time_sec / self.tool_call_count if self.tool_call_count > 0 else 0.0
        avg_prefetch_wait_time = prefetch_wait_time / prefetch_hit_count if prefetch_hit_count > 0 else 0.0
        avg_prefetch_saved_time = prefetch_saved_time / prefetch_hit_count if prefetch_hit_count > 0 else 0.0
        prefetch_hit_rate = prefetch_hit_count / self.tool_call_count if self.tool_call_count > 0 else 0.0

        self.last_run_stats = {
            "duration": duration,
            "prompt_build_time": prompt_build_time,
            "generated_tokens": generated_tokens,
            "prompt_tokens": prompt_tokens,
            "high_entropy_steps": high_entropy_steps,
            "low_entropy_steps": low_entropy_steps,
            "draft_tokens_total": total_draft,
            "accepted_tokens_total": accepted_count,
            "rejected_tokens_total": rejected_tokens_count,
            "overall_tps": generated_tokens / duration if duration > 0 else 0.0,
            "pure_tps": pure_tps,
            "accept_rate": accept_rate,
            "tool_exec_time": self.total_tool_time_sec,
            "avg_tool_exec_time": avg_tool_exec_time,
            "tool_call_count": self.tool_call_count,
            "tool_plan_count": prefetch_ready_count,
            "repeat_action_interrupt_count": self.repeat_action_interrupt_count,
            "prefetch_submit_count": prefetch_submit_count,
            "prefetch_hit_count": prefetch_hit_count,
            "prefetch_ready_hit_count": prefetch_ready_hit_count,
            "prefetch_miss_count": prefetch_miss_count,
            "prefetch_timeout_count": prefetch_timeout_count,
            "prefetch_wait_time": prefetch_wait_time,
            "avg_prefetch_wait_time": avg_prefetch_wait_time,
            "prefetch_saved_time": prefetch_saved_time,
            "avg_prefetch_saved_time": avg_prefetch_saved_time,
            "prefetch_hit_rate": prefetch_hit_rate,
            "generated_text": text_out,
            "prefetch_stats": dict(self.prefetch_stats),
        }
        return self.last_run_stats


def parse_args():
    parser = argparse.ArgumentParser(description="Run speculative ReAct evaluation on GSM8K.")
    parser.add_argument("--big-model-path", required=True, help="Path or model identifier for the primary model.")
    parser.add_argument("--small-model-path", required=True, help="Path or model identifier for the draft model.")
    parser.add_argument("--dataset-path", required=True, help="Path to the GSM8K evaluation dataset file.")
    parser.add_argument("--output-csv", default="final_integrated_results.csv", help="Path to the output CSV file.")
    parser.add_argument("--threshold", type=float, default=0.40, help="Entropy threshold for switching decode modes.")
    parser.add_argument("--lookahead", type=int, default=11, help="Maximum draft lookahead length.")
    parser.add_argument("--max-gen", type=int, default=2048, help="Maximum generated tokens per sample.")
    parser.add_argument("--num-test", type=int, default=10000, help="Maximum number of deduplicated GSM8K samples to evaluate.")
    parser.add_argument("--stream-output", action="store_true", help="Stream generation and tool observations to stdout.")
    parser.add_argument("--enable-ste", action="store_true", default=True, help="Enable speculative tool execution.")
    parser.add_argument("--disable-ste", action="store_false", dest="enable_ste", help="Disable speculative tool execution.")
    parser.add_argument("--ste-max-workers", type=int, default=2, help="Maximum worker threads for tool prefetch.")
    parser.add_argument("--ste-min-expr-len", type=int, default=6, help="Minimum normalized expression length for prefetch.")
    parser.add_argument("--ste-wait-timeout", type=float, default=5.0, help="Maximum wait time for prefetched tool results.")
    parser.add_argument("--ste-ttl-sec", type=float, default=15.0, help="Time-to-live for prefetched tool results.")
    parser.add_argument("--enable-hidden-probe", action="store_true", default=True, help="Enable hidden-probe prefetching.")
    parser.add_argument("--disable-hidden-probe", action="store_false", dest="enable_hidden_probe", help="Disable hidden-probe prefetching.")
    parser.add_argument("--hidden-probe-steps", type=int, default=12, help="Maximum hidden-probe draft steps.")
    return parser.parse_args()


def main():
    args = parse_args()

    engine = ReActSpeculativeEngine(
        big_path=args.big_model_path,
        small_path=args.small_model_path,
        threshold=args.threshold,
        lookahead=args.lookahead,
        enable_ste=args.enable_ste,
        ste_max_workers=args.ste_max_workers,
        ste_min_expr_len=args.ste_min_expr_len,
        ste_wait_timeout=args.ste_wait_timeout,
        ste_ttl_sec=args.ste_ttl_sec,
        enable_hidden_probe=args.enable_hidden_probe,
        hidden_probe_steps=args.hidden_probe_steps,
    )
    engine.stream_output = args.stream_output
    engine.log_csv = args.output_csv

    print(f">>> Loading dataset: {args.dataset_path}")
    full_dataset = Dataset.from_file(args.dataset_path)

    print(">>> Deduplicating GSM8K questions and evaluating them in dataset order.")

    selected_indices = []
    seen_questions = set()

    for idx, entry in enumerate(full_dataset):
        question_text = str(entry.get("question", "")).strip()
        if not question_text or question_text in seen_questions:
            continue
        seen_questions.add(question_text)
        selected_indices.append(idx)

    num_test = min(args.num_test, len(selected_indices))
    test_data = full_dataset.select(selected_indices[:num_test])
    print(f">>> Prepared {num_test} GSM8K samples for evaluation.")

    results = []
    correct_count = 0

    try:
        for i, entry in enumerate(test_data):
            question = str(entry.get("question", "")).strip()
            reference_full = str(entry.get("answer", "")).strip()

            print(f"\n\n[Progress] Evaluating sample {i + 1}/{num_test}")

            try:
                case_stats = engine.run_speculative(question, max_gen=args.max_gen)

                pred_ans = extract_boxed_content(case_stats["generated_text"])
                if not pred_ans:
                    pred_ans = case_stats["generated_text"].strip()

                ref_ans = reference_full.split("####")[-1].strip() if "####" in reference_full else reference_full.strip()

                is_hit = is_equivalent(pred_ans, ref_ans)
                if is_hit:
                    correct_count += 1

                print(f"\n[Result] Predicted: {pred_ans} | Reference: {ref_ans} | Correct: {'yes' if is_hit else 'no'}")
                print(
                    "[Sample metrics] "
                    f"duration={case_stats['duration']:.2f}s | "
                    f"prompt_tokens={case_stats['prompt_tokens']} | "
                    f"total_tokens={case_stats['generated_tokens']} | "
                    f"draft_tokens={case_stats['draft_tokens_total']} | "
                    f"accepted_tokens={case_stats['accepted_tokens_total']} | "
                    f"rejected_tokens={case_stats['rejected_tokens_total']} | "
                    f"high_entropy_steps={case_stats['high_entropy_steps']} | "
                    f"low_entropy_steps={case_stats['low_entropy_steps']} | "
                    f"tps={case_stats['overall_tps']:.2f} | "
                    f"pure_tps={case_stats['pure_tps']:.2f} | "
                    f"tool_exec_time={case_stats['tool_exec_time']:.4f}s | "
                    f"avg_tool_exec_time={case_stats['avg_tool_exec_time']:.4f}s | "
                    f"prefetch_ready={case_stats['tool_plan_count']} | "
                    f"tool_calls={case_stats['tool_call_count']} | "
                    f"repeat_action_interrupts={case_stats['repeat_action_interrupt_count']} | "
                    f"prefetch_submitted={case_stats['prefetch_submit_count']} | "
                    f"prefetch_hit={case_stats['prefetch_hit_count']} | "
                    f"prefetch_ready_hit={case_stats['prefetch_ready_hit_count']} | "
                    f"prefetch_miss={case_stats['prefetch_miss_count']} | "
                    f"prefetch_hit_rate={case_stats['prefetch_hit_rate']:.2%} | "
                    f"prefetch_wait_time={case_stats['prefetch_wait_time']:.4f}s | "
                    f"avg_prefetch_wait={case_stats['avg_prefetch_wait_time']:.4f}s | "
                    f"prefetch_saved_time={case_stats['prefetch_saved_time']:.4f}s | "
                    f"avg_prefetch_saved={case_stats['avg_prefetch_saved_time']:.4f}s | "
                    f"prefetch_timeout={case_stats['prefetch_timeout_count']} | "
                    f"accept_rate={case_stats['accept_rate']:.2%}"
                )

                res_entry = {
                    "id": i,
                    "question": question,
                    "ref_ans": ref_ans,
                    "pred_ans": pred_ans,
                    "is_correct": is_hit,
                    "duration": round(case_stats["duration"], 4),
                    "prompt_tokens": case_stats["prompt_tokens"],
                    "prompt_build_time": round(case_stats["prompt_build_time"], 6),
                    "total_tokens": case_stats["generated_tokens"],
                    "draft_tokens_total": case_stats["draft_tokens_total"],
                    "accepted_tokens_total": case_stats["accepted_tokens_total"],
                    "rejected_tokens_total": case_stats["rejected_tokens_total"],
                    "high_entropy_steps": case_stats["high_entropy_steps"],
                    "low_entropy_steps": case_stats["low_entropy_steps"],
                    "tokens_per_sec": round(case_stats["overall_tps"], 4),
                    "pure_tokens_per_sec": round(case_stats["pure_tps"], 4),
                    "tool_exec_time": round(case_stats["tool_exec_time"], 6),
                    "avg_tool_exec_time": round(case_stats["avg_tool_exec_time"], 6),
                    "prefetch_ready_count": case_stats["tool_plan_count"],
                    "tool_call_count": case_stats["tool_call_count"],
                    "repeat_action_interrupt_count": case_stats["repeat_action_interrupt_count"],
                    "prefetch_submit_count": case_stats["prefetch_submit_count"],
                    "prefetch_hit_count": case_stats["prefetch_hit_count"],
                    "prefetch_ready_hit_count": case_stats["prefetch_ready_hit_count"],
                    "prefetch_miss_count": case_stats["prefetch_miss_count"],
                    "prefetch_hit_rate": round(case_stats["prefetch_hit_rate"], 6),
                    "prefetch_wait_time": round(case_stats["prefetch_wait_time"], 6),
                    "avg_prefetch_wait_time": round(case_stats["avg_prefetch_wait_time"], 6),
                    "prefetch_saved_time": round(case_stats["prefetch_saved_time"], 6),
                    "avg_prefetch_saved_time": round(case_stats["avg_prefetch_saved_time"], 6),
                    "prefetch_timeout_count": case_stats["prefetch_timeout_count"],
                    "accept_rate": round(case_stats["accept_rate"], 6),
                    "tool_used": case_stats["tool_call_count"] > 0,
                }
                results.append(res_entry)
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                import traceback
                traceback.print_exc()
                continue

        if results:
            df = pd.DataFrame(results)
            df.to_csv(engine.log_csv, index=False, encoding="utf-8-sig")
            print(f"\n{'#' * 20} Evaluation complete {'#' * 20}")
            print(f"Average prompt tokens: {df['prompt_tokens'].mean():.2f}")
            print(f"Average prompt build time: {df['prompt_build_time'].mean():.6f}s")
            print(f"Average generated tokens: {df['total_tokens'].mean():.2f}")
            print(f"Average draft tokens: {df['draft_tokens_total'].mean():.2f}")
            print(f"Average accepted tokens: {df['accepted_tokens_total'].mean():.2f}")
            print(f"Average rejected tokens: {df['rejected_tokens_total'].mean():.2f}")
            print(f"Average high-entropy steps: {df['high_entropy_steps'].mean():.2f}")
            print(f"Average low-entropy steps: {df['low_entropy_steps'].mean():.2f}")
            print(f"Average TPS (wall time): {df['tokens_per_sec'].mean():.2f}")
            print(f"Average TPS (excluding tool wait): {df['pure_tokens_per_sec'].mean():.2f}")
            print(f"Total prefetch-ready events: {int(df['prefetch_ready_count'].sum())}")
            print(f"Total tool calls: {int(df['tool_call_count'].sum())}")
            print(f"Average tool execution time: {df['avg_tool_exec_time'].mean():.6f}s")
            print(f"Total repeated-action interrupts: {int(df['repeat_action_interrupt_count'].sum())}")
            print(f"Total prefetch submissions: {int(df['prefetch_submit_count'].sum())}")
            print(f"Total prefetch hits: {int(df['prefetch_hit_count'].sum())}")
            print(f"Total prefetch-ready hits: {int(df['prefetch_ready_hit_count'].sum())}")
            print(f"Total prefetch misses: {int(df['prefetch_miss_count'].sum())}")
            print(
                f"Prefetch hit rate (per tool call): {(df['prefetch_hit_count'].sum() / df['tool_call_count'].sum()):.2%}"
                if df['tool_call_count'].sum() > 0 else "Prefetch hit rate (per tool call): 0.00%"
            )
            print(f"Total prefetch wait time: {df['prefetch_wait_time'].sum():.6f}s")
            print(f"Average prefetch wait time: {df['avg_prefetch_wait_time'].mean():.6f}s")
            print(f"Estimated prefetch time saved: {df['prefetch_saved_time'].sum():.6f}s")
            print(f"Average prefetch time saved: {df['avg_prefetch_saved_time'].mean():.6f}s")
            print(f"Total prefetch timeouts: {int(df['prefetch_timeout_count'].sum())}")
            print(f"Average draft acceptance rate: {df['accept_rate'].mean():.2%}")
            print(f"Share of samples using the tool: {df['tool_used'].mean():.2%}")
            print(
                f"Accuracy on tool-using samples: {df[df['tool_used']]['is_correct'].mean():.2%}"
                if df['tool_used'].any() else "Accuracy on tool-using samples: N/A"
            )
            print(
                f"Accuracy on non-tool samples: {df[~df['tool_used']]['is_correct'].mean():.2%}"
                if (~df['tool_used']).any() else "Accuracy on non-tool samples: N/A"
            )
            print(f"Final accuracy: {correct_count / len(results):.2%} ({correct_count}/{len(results)})")
            print(f"Results saved to: {engine.log_csv}")
    finally:
        engine.close()


if __name__ == "__main__":
    main()
