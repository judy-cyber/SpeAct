"""
Open-source evaluation script for speculative decoding with chain-of-thought prompting.

Usage:
    python MATH_SD_COT.py
"""

import os
import re
import time

import pandas as pd
import sympy
import torch
import torch.nn.functional as F
from awq import AutoAWQForCausalLM
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache


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


class CoTSpeculativeEngine:
    """Speculative decoding engine with chain-of-thought prompting and few-shot examples."""

    def __init__(self, big_path, small_path, threshold=0.4, lookahead=9, result_csv_path=None):
        torch.cuda.empty_cache()
        self.threshold = threshold
        self.lookahead = lookahead

        print(">>> Loading large model...")
        self.big_model_wrapper = AutoAWQForCausalLM.from_quantized(
            big_path,
            fuse_layers=True,
            trust_remote_code=True,
            device_map="auto",
        )
        self.big_model = self.big_model_wrapper.model

        print(">>> Loading draft model...")
        self.small_model = AutoModelForCausalLM.from_pretrained(
            small_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(big_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.eos_id = self.tokenizer.eos_token_id
        self.log_csv = result_csv_path or os.getenv("RESULT_CSV_PATH")
        self.stream_output = True

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

    def _build_cot_prompt(self, question):
        """Build a prompt with chain-of-thought instructions and few-shot examples."""
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


        messages = [{"role": "system", "content": system_content}]

        for example in few_shot_examples:
            messages.append({"role": "user", "content": f"Question: {example['question']}\nAnswer:"})
            messages.append({"role": "assistant", "content": example["answer"]})

        messages.append({"role": "user", "content": f"Question: {question}\nAnswer:"})

        prompt_str = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return prompt_str

    @torch.no_grad()
    def run(self, question, max_gen=2048):
        """Run speculative decoding with chain-of-thought prompting."""
        prompt_str = self._build_cot_prompt(question)
        prompt_ids = self.tokenizer.encode(prompt_str, add_special_tokens=False)

        input_id_chunks = [prompt_ids]
        flat_input_ids = prompt_ids
        input_ids_tensor = torch.tensor([flat_input_ids], device=self.big_model.device)
        current_len = len(flat_input_ids)
        big_cache, small_cache = DynamicCache(), DynamicCache()

        big_out = self.big_model(input_ids_tensor, past_key_values=big_cache, use_cache=True)
        small_out = self.small_model(
            input_ids_tensor.to(self.small_model.device),
            past_key_values=small_cache,
            use_cache=True,
        )

        big_logits = big_out.logits[:, -1, :]
        small_logits = small_out.logits[:, -1, :]

        start_time, start_len = time.time(), current_len
        accepted_count, total_draft = 0, 0

        while (current_len - start_len) < max_gen:
            probs = F.softmax(big_logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1).item()
            checkpoint_len = big_cache.get_seq_length()

            if entropy > self.threshold:
                next_token = torch.argmax(big_logits, dim=-1, keepdim=True)
                t_val = next_token.item()
                input_id_chunks.append([t_val])
                current_len += 1

                if self.stream_output:
                    print(self.tokenizer.decode([t_val]), end="", flush=True)

                if t_val == self.eos_id:
                    break

                pos = torch.tensor([[checkpoint_len]], device=next_token.device)
                big_out = self.big_model(next_token, past_key_values=big_cache, position_ids=pos, use_cache=True)
                small_out = self.small_model(
                    next_token.to(self.small_model.device),
                    past_key_values=small_cache,
                    position_ids=pos,
                    use_cache=True,
                )
                big_logits, small_logits = big_out.logits[:, -1, :], small_out.logits[:, -1, :]
            else:
                draft_tokens = []
                temp_input = torch.argmax(small_logits, dim=-1, keepdim=True)

                for i in range(self.lookahead):
                    draft_tokens.append(temp_input)
                    if temp_input.item() == self.eos_id:
                        break

                    p_id = torch.tensor([[checkpoint_len + i]], device=temp_input.device)
                    s_out = self.small_model(temp_input, past_key_values=small_cache, position_ids=p_id, use_cache=True)
                    small_logits = s_out.logits[:, -1, :]
                    temp_input = torch.argmax(small_logits, dim=-1, keepdim=True)

                draft_seq = torch.cat(draft_tokens, dim=-1).to(self.big_model.device)
                actual_draft_len = draft_seq.shape[1]
                total_draft += actual_draft_len

                v_pos = torch.arange(
                    checkpoint_len,
                    checkpoint_len + actual_draft_len,
                    device=draft_seq.device,
                ).unsqueeze(0)
                verify_out = self.big_model(draft_seq, past_key_values=big_cache, position_ids=v_pos, use_cache=True)
                verify_logits = verify_out.logits

                n_matches = 0
                for i in range(actual_draft_len):
                    target_logits = big_logits if i == 0 else verify_logits[:, i - 1, :]
                    correct_token = torch.argmax(target_logits, dim=-1, keepdim=True)

                    if correct_token.item() == draft_tokens[i].item():
                        n_matches += 1
                        accepted_count += 1
                        t_val = correct_token.item()
                        input_id_chunks.append([t_val])
                        current_len += 1

                        if self.stream_output:
                            print(self.tokenizer.decode([t_val]), end="", flush=True)

                        if t_val == self.eos_id:
                            break
                    else:
                        break

                new_len = checkpoint_len + n_matches
                self._crop_cache(big_cache, new_len)
                self._crop_cache(small_cache, new_len)

                if current_len > start_len and input_id_chunks[-1][-1] == self.eos_id:
                    break

                f_logits = big_logits.clone() if n_matches == 0 else verify_logits[:, n_matches - 1, :].clone()

                if n_matches < actual_draft_len:
                    rejected_id = draft_tokens[n_matches].item()
                    rejected_text = self.tokenizer.decode([rejected_id])

                    if any(char.isdigit() for char in rejected_text):
                        f_logits[0, rejected_id] = -float("inf")
                    else:
                        alpha = 4
                        lm_weights = self.big_model.get_output_embeddings().weight
                        sim = F.cosine_similarity(
                            lm_weights,
                            lm_weights[rejected_id].unsqueeze(0),
                            dim=-1,
                        )
                        penalty = F.relu(sim) * alpha
                        f_logits -= penalty.unsqueeze(0)

                final_correct = torch.argmax(f_logits, dim=-1, keepdim=True)
                t_val = final_correct.item()
                input_id_chunks.append([t_val])
                current_len += 1

                if self.stream_output:
                    print(self.tokenizer.decode([t_val]), end="", flush=True)

                if t_val == self.eos_id:
                    break

                sync_pos = torch.tensor([[new_len]], device=final_correct.device)
                big_out = self.big_model(final_correct, past_key_values=big_cache, position_ids=sync_pos, use_cache=True)
                small_out = self.small_model(
                    final_correct.to(self.small_model.device),
                    past_key_values=small_cache,
                    position_ids=sync_pos,
                    use_cache=True,
                )
                big_logits, small_logits = big_out.logits[:, -1, :], small_out.logits[:, -1, :]

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
        gen_text = self.tokenizer.decode(generated_token_ids, skip_special_tokens=True)
        generated_tokens = current_len - start_len
        overall_tps = generated_tokens / dur if dur > 0 else 0.0

        return {
            "duration": dur,
            "generated_tokens": generated_tokens,
            "accept_rate": (accepted_count / total_draft if total_draft > 0 else 0),
            "generated_text": gen_text,
            "overall_tps": overall_tps,
        }


# ==========================================
# Main evaluation entry point
# ==========================================
def main():
    big_model_path = os.getenv("BIG_MODEL_PATH")
    small_model_path = os.getenv("SMALL_MODEL_PATH")
    dataset_path = os.getenv("DATASET_PATH")
    result_csv_path = os.getenv("RESULT_CSV_PATH")
    threshold = float(os.getenv("SPECULATIVE_THRESHOLD", "0.40"))
    lookahead = int(os.getenv("SPECULATIVE_LOOKAHEAD", "11"))
    max_candidates = int(os.getenv("MAX_CANDIDATES", "10000"))
    max_samples = int(os.getenv("MAX_SAMPLES", "10000"))
    target_min_level = int(os.getenv("TARGET_MIN_LEVEL", "4"))
    stream_output = os.getenv("STREAM_OUTPUT", "false").strip().lower() in {"1", "true", "yes", "y", "on"}
    max_generation_tokens = int(os.getenv("MAX_GENERATION_TOKENS", "2048"))

    required_paths = {
        "BIG_MODEL_PATH": big_model_path,
        "SMALL_MODEL_PATH": small_model_path,
        "DATASET_PATH": dataset_path,
        "RESULT_CSV_PATH": result_csv_path,
    }
    missing_paths = [name for name, value in required_paths.items() if not value]
    if missing_paths:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_paths)}")

    print(f">>> Loading dataset from configured path: {dataset_path}")
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

    engine = CoTSpeculativeEngine(
        big_path=big_model_path,
        small_path=small_model_path,
        threshold=threshold,
        lookahead=lookahead,
        result_csv_path=result_csv_path,
    )
    engine.stream_output = stream_output

    results = []
    correct_count = 0

    for i, entry in enumerate(test_data):
        question = entry.get('problem', entry.get('question', ""))
        reference_full = str(entry.get('solution', entry.get('solution', "")))

        print(f"\n\n[Progress] Evaluating sample {i + 1}/{num_test}")

        try:
            case_stats = engine.run(question, max_gen=max_generation_tokens)

            pred_ans = extract_boxed_content(case_stats["generated_text"])
            ref_ans = extract_boxed_content(reference_full) if "\\boxed" in reference_full else reference_full.strip()

            is_hit = is_equivalent(pred_ans, ref_ans)
            if is_hit:
                correct_count += 1

            print(
                f"\n[Evaluation] Predicted answer: {pred_ans} | "
                f"Reference answer: {ref_ans} | Correct: {'yes' if is_hit else 'no'}"
            )
            print(
                "[Case Metrics] "
                f"duration={case_stats['duration']:.2f}s | "
                f"total_tokens={case_stats['generated_tokens']} | "
                f"tps={case_stats['overall_tps']:.2f} | "
                f"accept_rate={case_stats['accept_rate']:.2%}"
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
                "accept_rate": round(case_stats["accept_rate"], 4),
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
        print(f"\n{'#' * 20} Evaluation Complete {'#' * 20}")
        print(f"Average TPS (wall clock): {df['tokens_per_second'].mean():.2f}")
        print(f"Average draft-model acceptance rate: {df['accept_rate'].mean():.2%}")
        print(f"Final accuracy: {correct_count / len(results):.2%} ({correct_count}/{len(results)})")
        print(f"Results saved to: {engine.log_csv}")


if __name__ == "__main__":
    main()
