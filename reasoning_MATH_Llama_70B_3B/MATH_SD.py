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


class PureSpeculativeEngine:
    """Pure speculative decoding engine without tools or explicit chain-of-thought prompting."""

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

    @torch.no_grad()
    def run(self, question, max_gen=1024):
        """Run pure speculative decoding inference and return the final answer text."""
        system_content = (
            "You are a math problem solver. Provide an accurate final answer. "
            "If the task requires reasoning, keep it concise and finish with the answer inside \\boxed{}."
        )

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": f"Question: {question}\nAnswer:"},
        ]

        prompt_str = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
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
    max_samples = int(os.getenv("MAX_SAMPLES", "10000"))
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

    engine = PureSpeculativeEngine(
        big_path=big_model_path,
        small_path=small_model_path,
        threshold=threshold,
        lookahead=lookahead,
        result_csv_path=result_csv_path,
    )
    engine.stream_output = stream_output

    print(f">>> Loading dataset from configured path: {dataset_path}")
    full_dataset = Dataset.from_file(dataset_path)

    print(f">>> Selecting up to {max_samples} samples for speculative decoding evaluation...")
    num_test = min(max_samples, len(full_dataset))
    test_data = full_dataset.select(range(num_test))
    print(f">>> Selection complete. Preparing to evaluate {num_test} samples.")

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
