import os
import re
import time

import pandas as pd
import requests
import sympy
import torch
from awq import AutoAWQForCausalLM
from datasets import Dataset
from transformers import AutoTokenizer, DynamicCache


# =================================================================
# Evaluation helper utilities
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
        if text[i] == '{':
            brace_count += 1
        elif text[i] == '}':
            brace_count -= 1
        if brace_count == 0:
            return text[content_start:i].strip()
    return text[content_start:].strip()


def is_equivalent(pred, ref):
    if not pred or not ref:
        return False

    from sympy.parsing.sympy_parser import (
        implicit_multiplication_application,
        parse_expr,
        standard_transformations,
    )

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


class ReActEngine:
    def __init__(self, model_path):
        torch.cuda.empty_cache()

        print(">>> Loading AWQ model...")
        self.model_wrapper = AutoAWQForCausalLM.from_quantized(
            model_path,
            fuse_layers=True,
            trust_remote_code=True,
            device_map="auto",
        )
        self.model = self.model_wrapper.model

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.is_qwen = "qwen" in model_path.lower() or "qwen" in self.tokenizer.__class__.__name__.lower()
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
        self.action_ids = {self.tokenizer.encode(w, add_special_tokens=False)[-1] for w in ["Action", " Action", "\nAction"]}
        self.obs_ids = {self.tokenizer.encode(w, add_special_tokens=False)[-1] for w in ["Observation", " Observation", "\nObservation"]}
        self.thought_ids = {self.tokenizer.encode(w, add_special_tokens=False)[-1] for w in ["Thought", " Thought", "\nThought"]}
        self.log_csv = os.getenv("RESULT_CSV_PATH", "react_evaluation_results.csv")
        self.tool_call_count = 0
        self.tool_exec_time = 0.0

        self.wolfram_appid = os.getenv("WOLFRAM_APPID", "").strip()
        self.wolfram_timeout = (5, 20)
        self.foreground_http_session = requests.Session()
        self.stream_output = True

        self.no_solution_count = 0
        self.error_count = 0
        self.expr_history = []
        self.repeat_action_count = 0

    def _update_phase(self, token_val, current_phase):
        if token_val in self.action_ids:
            return 1
        if token_val in self.obs_ids:
            return 2
        if token_val in self.thought_ids:
            return 0
        return current_phase

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
        url = "http://api.wolframalpha.com/v1/result"
        params = {
            "appid": self.wolfram_appid,
            "i": expr_str,
        }

        response = session.get(url, params=params, timeout=self.wolfram_timeout)
        if response.status_code == 501:
            return {
                "status": "unparsed",
                "answer": "Wolfram could not understand the query. Try rephrasing with 'solve', 'simplify', or 'factor'.",
            }

        response.raise_for_status()
        answer_text = response.text.strip()
        if not answer_text:
            raise ValueError("Empty response from Wolfram short-answers API.")

        return {
            "status": "ok",
            "answer": answer_text,
        }

    def _format_wolfram_observation(self, expr_str, tool_result):
        lowered_expr = expr_str.lower()
        status = tool_result.get("status", "ok")
        answer_text = tool_result.get("answer", "")
        lowered_answer = answer_text.lower()

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

    def _inject_tool_result(self, tool_obs_str, input_id_chunks, current_len, cache):
        if self.stream_output:
            print(tool_obs_str, end="", flush=True)

        obs_ids = self.tokenizer.encode(tool_obs_str, add_special_tokens=False)
        obs_tensor = torch.tensor([obs_ids], device=self.model.device)
        pos = torch.arange(current_len, current_len + len(obs_ids), device=self.model.device).unsqueeze(0)

        out = self.model(obs_tensor, past_key_values=cache, position_ids=pos, use_cache=True)
        logits = out.logits[:, -1, :]

        input_id_chunks.append(obs_ids)
        current_len += len(obs_ids)
        return input_id_chunks, current_len, logits

    def _handle_tool_trigger(self, action_buffer, tool_obs_str, input_id_chunks, current_len, cache):
        extracted_expr, _ = self._extract_calculate_expr(action_buffer)
        current_expr = self._normalize_tool_expr(extracted_expr if extracted_expr else action_buffer.strip())

        if current_expr in self.expr_history:
            self.repeat_action_count += 1
        else:
            self.expr_history.append(current_expr)
            if len(self.expr_history) > 5:
                self.expr_history.pop(0)
            self.repeat_action_count = 0

        if self.repeat_action_count >= 2:
            tool_obs = (
                "Observation: [System Error] Tool execution is stuck in an infinite loop. You MUST immediately STOP using the Calculate tool.\n"
                "Thought: The calculator is failing continuously. I must stop trying to use it. I will rely entirely on my own manual algebraic reasoning (like completing the square or simple arithmetic) to move forward, or I will output the Final Answer now.\n"
            )
            self.repeat_action_count = 0
        else:
            tool_obs = tool_obs_str

        return self._inject_tool_result(tool_obs, input_id_chunks, current_len, cache)

    @torch.no_grad()
    def run_react(self, question, max_gen=1024):
        self.no_solution_count = 0
        self.error_count = 0
        self.repeat_action_count = 0
        self.expr_history = []
        self.tool_call_count = 0
        self.tool_exec_time = 0.0

        few_shot_examples = (
            "Here are example math solutions with careful reasoning. The calculator is a fallback rather than the default path.\n\n"
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
            "--- Example 6: Universal Quantifier (Find p such that inequality holds for every q>0) ---\n"
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
            "--- Example 8: Recursive Operation (Step by Step) ---\n"
            "Question: Define # by: r # 0 = r, r # s = s # r, (r+1) # s = (r # s) + s + 1. Find 11 # 5.\n"
            "Thought: This is recursive. I will compute 0#5 = 5 (base case). Then I can use (r+1)#5 = (r#5)+5+1 repeatedly.\n"
            "Action: Calculate[ 5 + 5 + 1 ]\n"
            "Observation: 11\n"
            "Thought: So 1#5=11. Next, 2#5 = 1#5 + 5 + 1.\n"
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
            "   - MANDATORY: Use `solve <equation> for <variable>`.\n"
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
            "   - Use `*` for ALL multiplications (for example, `3*x`, `(a+b)*(c+d)`).\n"
            "   - NO ENGLISH instructions (for example, 'round', 'limit', 'value of') inside Calculate[...].\n\n"
            "### DECISION POLICY: ZERO MENTAL ALGEBRA (CRITICAL) ###\n"
            "1. NO MENTAL TRANSPOSITION: Do NOT isolate variables or move terms across the '=' sign in your Thought.\n"
            "   - WRONG: 'Since -2x + y = k and x = -8.4, then k = 16.8 + y.' (This leads to fatal sign errors.)\n"
            "   - RIGHT: 'I will substitute the values into the raw equation and use the solver to find the target.'\n"
            "     -> Action: Calculate[ solve -2*(-8.4) + 9.8 = k for k ]\n"
            "2. EXHAUSTIVE ANALYSIS: If a problem involves multiple cases, pairings, or combinations, you MUST list ALL possible scenarios in Thought before using the tool to compare them.\n"
            "3. DOMAIN VIGILANCE: For rational expressions or inequalities, always identify values where the expression is undefined (for example, denominator = 0) in Thought first.\n\n"
            "### TOOL RULES ###\n"
            "1. When a tool call is needed, first state one concise reasoning line in Thought, then emit the Action line directly.\n"
            "2. Allowed: Calculate[ <expr> ] for exact arithmetic or simplification.\n"
            "3. Allowed: Calculate[ solve <eq> for <var> ] for any algebraic isolation or solving.\n"
            "4. Forbidden: inline substitutions such as Calculate[a*x+b=0, x=2]. Perform substitutions in Thought first.\n"
            "5. If the calculator returns [ToolError], analyze whether you used 'simplify' on an '=' equation. Rewrite using 'solve' or check for missing `*` signs.\n"
            "6. Tip: For complex or multi-step calculations, prefer the Calculate tool over mental algebra.\n"
            "### THOUGHT DISCIPLINE ###\n"
            "Every Action must be justified. Prefer saying: 'I will use the solver to avoid potential sign errors in manual transposition.'\n"
            "When you decide to call the tool, output one concise reasoning line, then the Action line.\n"
            "Prefer exact forms (for example, 20/3, sqrt(5)). Do not convert to decimals unless the question explicitly requests an approximation.\n"
            "Always end your reasoning with Final Answer: \\boxed{result}. Do not generate any extra text after the final answer.\n\n"
            f"{few_shot_examples}"
        )
        user_content = f"Question: {question}\nSolve this step by step using a plain ReAct format with Thought, Action, Observation, and Final Answer. Begin with 'Thought: ' and think carefully before any Action."
        if hasattr(self.tokenizer, "apply_chat_template") and getattr(self.tokenizer, "chat_template", None):
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ]
            prompt_str = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt_str = f"System: {system_content}\n\nUser: {user_content}\nAssistant: "
        prompt_ids = self.tokenizer.encode(prompt_str, add_special_tokens=False)
        thought_prefix_ids = self.tokenizer.encode("Thought: ", add_special_tokens=False)

        input_id_chunks = [prompt_ids, thought_prefix_ids]
        flat_input_ids = prompt_ids + thought_prefix_ids
        input_ids_tensor = torch.tensor([flat_input_ids], device=self.model.device)
        current_len = len(flat_input_ids)
        cache = DynamicCache()

        out = self.model(input_ids_tensor, past_key_values=cache, use_cache=True)
        logits = out.logits[:, -1, :]

        start_time = time.time()
        start_len = current_len
        current_phase = 0
        active_buffer = ""

        def process_token(t_val, phase, buffer):
            new_phase = self._update_phase(t_val, phase)
            text_chunk = self.tokenizer.decode([t_val])
            track_buffer = new_phase in [0, 1]

            if track_buffer:
                if new_phase != phase:
                    buffer = ""
                buffer += text_chunk
            else:
                buffer = ""

            triggered = False
            obs_str = ""

            if new_phase == 1 and "Calculate[" in buffer:
                extracted_expr, is_closed = self._extract_calculate_expr(buffer)
                if extracted_expr is not None and is_closed:
                    tool_exec_start = time.time()
                    obs_str = self._execute_tool(buffer)
                    self.tool_exec_time += time.time() - tool_exec_start
                    self.tool_call_count += 1
                    triggered = True

            return new_phase, buffer, triggered, obs_str

        while (current_len - start_len) < max_gen:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            t_val = next_token.item()
            input_id_chunks.append([t_val])
            current_len += 1

            current_phase, active_buffer, tool_triggered, tool_obs_str = process_token(t_val, current_phase, active_buffer)
            if self.stream_output:
                print(self.tokenizer.decode([t_val]), end="", flush=True)

            if t_val == self.eos_id or t_val == self.eot_id:
                break

            pos = torch.tensor([[cache.get_seq_length()]], device=next_token.device)
            out = self.model(next_token, past_key_values=cache, position_ids=pos, use_cache=True)
            logits = out.logits[:, -1, :]

            if tool_triggered:
                input_id_chunks, current_len, logits = self._handle_tool_trigger(
                    active_buffer,
                    tool_obs_str,
                    input_id_chunks,
                    current_len,
                    cache,
                )
                current_phase, active_buffer = 0, ""
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
        overall_tps = generated_tokens / dur if dur > 0 else 0.0
        pure_generation_time = max(dur - self.tool_exec_time, 1e-8)
        pure_tps = generated_tokens / pure_generation_time if pure_generation_time > 0 else 0.0

        return {
            "duration": dur,
            "generated_tokens": generated_tokens,
            "generated_text": gen_text,
            "overall_tps": overall_tps,
            "pure_tps": pure_tps,
            "tool_exec_time": self.tool_exec_time,
            "tool_call_count": self.tool_call_count,
        }


# ==========================================
# Main evaluation entry point
# ==========================================
def main():
    model_path = os.getenv("MODEL_PATH", "./models/react_model")
    dataset_path = os.getenv("DATASET_PATH", "./data/math_eval.arrow")
    result_csv_path = os.getenv("RESULT_CSV_PATH", "react_evaluation_results.csv")
    stream_output = os.getenv("STREAM_OUTPUT", "false").strip().lower() in {"1", "true", "yes", "y", "on"}
    max_candidates = int(os.getenv("MAX_CANDIDATES", "200"))
    max_samples = int(os.getenv("MAX_SAMPLES", "10000"))
    target_min_level = int(os.getenv("TARGET_MIN_LEVEL", "4"))

    engine = ReActEngine(model_path=model_path)
    engine.stream_output = stream_output
    engine.log_csv = result_csv_path

    print(">>> Loading dataset from configured path...")
    full_dataset = Dataset.from_file(dataset_path)

    print(
        f">>> Prioritizing problems at level {target_min_level} and above; "
        f"backfilling up to {max_candidates} candidates if needed..."
    )

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
        if level_score >= target_min_level:
            candidate_entries.append((idx, level_score, question_text))
        elif level_score >= 0:
            fallback_entries.append((idx, level_score, question_text))

    candidate_entries.sort(key=lambda item: (-item[1], -len(item[2]), item[0]))
    fallback_entries.sort(key=lambda item: (-item[1], -len(item[2]), item[0]))

    selected_indices = [idx for idx, _, _ in candidate_entries[:max_candidates]]

    if len(selected_indices) < max_candidates:
        remain = max_candidates - len(selected_indices)
        print(
            f">>> Fewer than {max_candidates} level-{target_min_level}+ items found; "
            f"backfilling with {remain} lower-level samples."
        )
        selected_indices.extend(idx for idx, _, _ in fallback_entries[:remain])

    if not selected_indices:
        print(">>> No valid level field found; falling back to ranking by problem length...")
        ranked_by_length = sorted(
            enumerate(full_dataset),
            key=lambda item: len(item[1].get('problem', item[1].get('question', ''))),
            reverse=True,
        )
        selected_indices = [idx for idx, _ in ranked_by_length[:max_candidates]]

    num_test = min(max_samples, len(selected_indices))
    test_data = full_dataset.select(selected_indices[:num_test])
    print(f">>> Selection complete. Preparing to evaluate {num_test} samples.")

    results = []
    correct_count = 0

    for i, entry in enumerate(test_data):
        question = entry.get('problem', entry.get('question', ""))
        reference_full = str(entry.get('solution', entry.get('solution', "")))

        print(f"\n\n[Progress] Evaluating sample {i+1}/{num_test}")

        try:
            case_stats = engine.run_react(question, max_gen=2048)

            pred_ans = extract_boxed_content(case_stats["generated_text"])
            ref_ans = extract_boxed_content(reference_full) if "\\boxed" in reference_full else reference_full.strip()

            is_hit = is_equivalent(pred_ans, ref_ans)
            if is_hit:
                correct_count += 1

            print(f"\n[Evaluation] Predicted answer: {pred_ans} | Reference answer: {ref_ans} | Correct: {'yes' if is_hit else 'no'}")
            print(
                "[Case Metrics] "
                f"duration={case_stats['duration']:.2f}s | "
                f"total_tokens={case_stats['generated_tokens']} | "
                f"tps={case_stats['overall_tps']:.2f} | "
                f"pure_tps={case_stats['pure_tps']:.2f} | "
                f"tool_exec_time={case_stats['tool_exec_time']:.2f}s | "
                f"tool_calls={case_stats['tool_call_count']}"
            )

            res_entry = {
                "id": i,
                "question": question,
                "reference_answer": ref_ans,
                "predicted_answer": pred_ans,
                "is_correct": is_hit,
                "duration": round(case_stats["duration"], 2),
                "total_tokens": case_stats["generated_tokens"],
                "tokens_per_second": round(case_stats["overall_tps"], 2),
                "pure_tokens_per_second": round(case_stats["pure_tps"], 2),
                "tool_exec_time": round(case_stats["tool_exec_time"], 2),
                "tool_call_count": case_stats["tool_call_count"],
            }
            results.append(res_entry)
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            import traceback

            traceback.print_exc()
            continue

    if results:
        df = pd.DataFrame(results)
        df.to_csv(engine.log_csv, index=False, encoding='utf-8-sig')
        print(f"\n{'#' * 20} Evaluation Complete {'#' * 20}")
        print(f"Average TPS (wall clock): {df['tokens_per_second'].mean():.2f}")
        print(f"Average TPS (excluding tool wait): {df['pure_tokens_per_second'].mean():.2f}")
        print(f"Total tool calls: {int(df['tool_call_count'].sum())}")
        print(f"Final accuracy: {correct_count / len(results):.2%} ({correct_count}/{len(results)})")
        print(f"Results saved to: {engine.log_csv}")


if __name__ == "__main__":
    main()
