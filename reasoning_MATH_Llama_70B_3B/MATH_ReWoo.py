"""
MATH_ReWoo.py — ReWOO + Wolfram API evaluation built on lagent.
================================================================
Uses the built-in lagent ReWOO agent, ReWOOProtocol, and
ActionExecutor instead of a handwritten engine.
Tool: Wolfram Alpha Short Answers API wrapped as a lagent BaseAction.
Dataset: configurable math evaluation subset.
"""

import os
import re
import time
import argparse
import requests

import torch
import pandas as pd
import sympy
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
)
from datasets import Dataset
from transformers import AutoTokenizer
from awq import AutoAWQForCausalLM

# ===== lagent framework components =====
from lagent.agents.rewoo import ReWOO, ReWOOProtocol
from lagent.actions import ActionExecutor, BaseAction
from lagent.actions.base_action import tool_api
from lagent.llms.huggingface import HFTransformerCasualLM, HFTransformer
from lagent.schema import ActionReturn, ActionStatusCode, AgentReturn, ModelStatusCode


# =================================================================
# AWQ-compatible LLM wrapper loaded with AutoAWQForCausalLM.from_quantized()
# =================================================================
class HFTransformerAWQ(HFTransformerCasualLM):
    """Fully compatible with HFTransformerCasualLM, but loaded with AWQ.

    This subclass overrides `_load_model` to match the AWQ quantized loading
    path and overrides streaming generation to stay compatible with newer
    transformers versions where `_get_logits_warper` may be absent.
    """

    def _load_model(self, path: str, model_kwargs: dict):
        import torch

        model_kwargs.setdefault("torch_dtype", torch.float16)
        # Remove HF-only arguments that are not supported by the AWQ loader
        model_kwargs.pop("device_map", None)
        model_kwargs.pop("torch_dtype", None)

        awq_model = AutoAWQForCausalLM.from_quantized(
            path,
            fuse_layers=True,
            trust_remote_code=True,
            **model_kwargs,
        )
        # The AWQ wrapper itself does not expose generation_config; lagent needs the inner HF model
        self.model = awq_model.model
        self.model.eval()

        # Fix compatibility between lagent streaming generation and
        # transformers stopping criteria by eagerly populating token tensors.
        gc = self.model.generation_config
        if gc.eos_token_id is not None:
            eos_ids = gc.eos_token_id
            if isinstance(eos_ids, int):
                eos_ids = [eos_ids]
            gc._eos_token_tensor = torch.tensor(eos_ids)
        if gc.bos_token_id is not None:
            bos_ids = gc.bos_token_id
            if isinstance(bos_ids, int):
                bos_ids = [bos_ids]
            gc._bos_token_tensor = torch.tensor(bos_ids)

    def _format_with_chat_template(self, inputs):
        if isinstance(inputs, str):
            messages = [{"role": "user", "content": inputs}]
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        if isinstance(inputs, list):
            if not inputs:
                return inputs
            if isinstance(inputs[0], str):
                return [self._format_with_chat_template(item) for item in inputs]
            if isinstance(inputs[0], dict):
                return self.tokenizer.apply_chat_template(
                    inputs,
                    tokenize=False,
                    add_generation_prompt=True,
                )
        return inputs

    def _tokenize_formatted_inputs(self, formatted_inputs):
        import torch

        batched = True
        if isinstance(formatted_inputs, str):
            formatted_inputs = [formatted_inputs]
            batched = False
        tokenized = self.tokenizer(
            formatted_inputs,
            padding=True,
            return_tensors="pt",
            return_length=True,
            add_special_tokens=False,
        )
        input_length = tokenized["length"]
        for k, v in tokenized.items():
            if isinstance(v, torch.Tensor):
                tokenized[k] = v.cuda()
        return tokenized, input_length, batched

    def _build_streaming_generation_state(self, input_ids, attention_mask, kwargs):
        import copy
        import torch

        input_ids_seq_length = input_ids.shape[-1]
        generation_config = copy.deepcopy(self.model.generation_config)
        new_gen_params = self.update_gen_params(**kwargs)
        generation_config.update(**new_gen_params)
        generation_config.update(**kwargs)
        model_kwargs = generation_config.to_dict()
        model_kwargs = {
            "attention_mask": attention_mask,
            "use_cache": model_kwargs.get("use_cache", True),
            "past_key_values": model_kwargs.get("past_key_values", None),
            "cache_position": model_kwargs.get("cache_position", None),
            "position_ids": model_kwargs.get("position_ids", None),
        }
        model_kwargs = {k: v for k, v in model_kwargs.items() if v is not None}
        _, eos_token_id = (
            generation_config.bos_token_id,
            generation_config.eos_token_id,
        )
        if eos_token_id is None:
            if self.gcfg.eos_token_id is not None:
                eos_token_id = self.gcfg.eos_token_id
            else:
                eos_token_id = []
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        if self.additional_eos_token_id is not None:
            eos_token_id.extend(self.additional_eos_token_id)
        eos_token_id_tensor = (
            torch.tensor(eos_token_id).to(input_ids.device)
            if eos_token_id is not None
            else None
        )
        max_new_tokens = generation_config.max_new_tokens
        if max_new_tokens is None:
            max_new_tokens = kwargs.get("max_new_tokens", 512)
            generation_config.max_new_tokens = max_new_tokens
        generation_config.max_length = max_new_tokens + input_ids_seq_length
        logits_processor = self.logits_processor
        stopping_criteria = self.stopping_criteria

        logits_processor = self.model._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_seq_length,
            encoder_input_ids=input_ids,
            prefix_allowed_tokens_fn=self.prefix_allowed_tokens_fn,
            logits_processor=logits_processor,
        )

        stopping_criteria = self.model._get_stopping_criteria(
            generation_config=generation_config,
            stopping_criteria=stopping_criteria,
        )

        if hasattr(self.model, "_get_logits_warper"):
            logits_warper = self.model._get_logits_warper(generation_config)
        else:
            logits_warper = lambda input_ids, scores: scores

        return {
            "generation_config": generation_config,
            "model_kwargs": model_kwargs,
            "eos_token_id": eos_token_id,
            "eos_token_id_tensor": eos_token_id_tensor,
            "logits_processor": logits_processor,
            "logits_warper": logits_warper,
            "stopping_criteria": stopping_criteria,
            "input_ids_seq_length": input_ids_seq_length,
        }

    def generate(
        self,
        inputs,
        do_sample: bool = False,
        **kwargs,
    ):
        formatted_inputs = self._format_with_chat_template(inputs)
        return super().generate(formatted_inputs, do_sample=do_sample, **kwargs)

    def stream_generate(
        self,
        inputs,
        do_sample: bool = False,
        **kwargs,
    ):
        import torch
        from torch import nn
        from lagent.schema import ModelStatusCode

        formatted_inputs = self._format_with_chat_template(inputs)

        with torch.no_grad():
            inputs, input_length, batched = self._tokenize_formatted_inputs(
                formatted_inputs
            )
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            batch_size = input_ids.shape[0]
            state = self._build_streaming_generation_state(
                input_ids, attention_mask, kwargs
            )
            model_kwargs = state["model_kwargs"]
            eos_token_id = state["eos_token_id"]
            eos_token_id_tensor = state["eos_token_id_tensor"]
            logits_processor = state["logits_processor"]
            logits_warper = state["logits_warper"]
            stopping_criteria = state["stopping_criteria"]

            scores = None
            unfinished_sequences = input_ids.new(batch_size).fill_(1)
            decode_interval = max(1, int(kwargs.get("decode_interval", 8)))
            generated_token_ids = [[] for _ in range(batch_size)]
            last_emitted_response = None
            final_response = ["" for _ in range(batch_size)]
            step_count = 0
            while True:
                model_inputs = self.model.prepare_inputs_for_generation(
                    input_ids, **model_kwargs
                )
                outputs = self.model(
                    **model_inputs,
                    return_dict=True,
                    output_attentions=False,
                    output_hidden_states=False,
                )

                next_token_logits = outputs.logits[:, -1, :]
                next_token_scores = logits_processor(input_ids, next_token_logits)
                next_token_scores = logits_warper(input_ids, next_token_scores)

                probs = nn.functional.softmax(next_token_scores, dim=-1)
                if do_sample:
                    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
                else:
                    next_tokens = torch.argmax(probs, dim=-1)

                input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
                model_kwargs = self.model._update_model_kwargs_for_generation(
                    outputs,
                    model_kwargs,
                    is_encoder_decoder=False,
                )
                unfinished_sequences = unfinished_sequences.mul(
                    next_tokens.tile(eos_token_id_tensor.shape[0], 1)
                    .ne(eos_token_id_tensor.unsqueeze(1))
                    .prod(dim=0)
                )

                next_tokens_list = next_tokens.detach().cpu().tolist()
                for i, token_id in enumerate(next_tokens_list):
                    if token_id in eos_token_id:
                        continue
                    generated_token_ids[i].append(token_id)

                step_count += 1
                should_stop = (
                    unfinished_sequences.max() == 0
                    or stopping_criteria(input_ids, scores)
                )
                should_emit = should_stop or (step_count % decode_interval == 0)

                if should_emit:
                    final_response = self.tokenizer.batch_decode(
                        generated_token_ids, skip_special_tokens=True
                    )
                    current_response = final_response if batched else final_response[0]
                    if current_response != last_emitted_response:
                        yield ModelStatusCode.STREAM_ING, current_response, None
                        last_emitted_response = current_response

                if should_stop:
                    break

            final_response = self.tokenizer.batch_decode(
                generated_token_ids, skip_special_tokens=True
            )
            if not batched:
                final_response = final_response[0]
            yield ModelStatusCode.END, final_response, None


# =================================================================
# Evaluation helper functions
# =================================================================
def extract_boxed_content(text):
    if not isinstance(text, str):
        return ""

    boxed_matches = re.findall(r"\\boxed\{([\s\S]*?)\}", text)
    if boxed_matches:
        return boxed_matches[-1].strip()

    final_answer_match = re.search(
        r"\[Final Answer\]\s*:?\s*(?:\n+)?([\s\S]+?)\s*$", text, re.IGNORECASE
    )
    if final_answer_match:
        candidate = final_answer_match.group(1).strip()
        candidate = re.split(r"\n\s*\[[^\n\]]+\]\s*:?,?", candidate)[0].strip()
        candidate = candidate.splitlines()[0].strip() if candidate else ""
        if candidate:
            return candidate

    answer_patterns = [
        r"(?:final answer|answer)\s*[:：]\s*([^\n]+)",
        r"\\boxed\s*([^\n]+)",
    ]
    for pattern in answer_patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            candidate = matches[-1].strip()
            candidate = re.sub(r"^[\[\(\{]+|[\]\)\}\.]+$", "", candidate).strip()
            if candidate:
                return candidate

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in reversed(lines):
        if line.startswith("[") and line.endswith("]"):
            continue
        if line.lower().startswith("response:"):
            candidate = line.split(":", 1)[1].strip()
            if candidate:
                return candidate
        if re.fullmatch(r"[-+*/().=a-zA-Z0-9_\\\s^]+", line):
            return line

    return ""


def is_equivalent(pred, ref):
    if not pred or not ref:
        return False

    def normalize_math_str(s):
        s = s.strip().lower()
        s = s.replace("$", "").replace("\\(", "").replace("\\)", "")
        s = s.replace("\\left", "").replace("\\right", "")
        s = s.replace("\\frac{", "(").replace("}{", ")/(").replace("}", ")")
        s = s.replace("\\sqrt{", "sqrt(").replace("}", ")")
        s = s.replace("\\sqrt", "sqrt")
        s = s.replace("\\pi", "pi").replace("\\infty", "oo")
        s = s.replace("^", "**").replace("{", "(").replace("}", ")")
        s = s.replace("\\cdot", "*").replace("\\times", "*")
        s = s.replace("−", "-").replace("×", "*").replace("÷", "/")
        s = re.sub(r"\s+", " ", s)
        s = s.replace("\\", "")
        return s.strip()

    def strip_wrappers(s):
        s = s.strip()
        for prefix in ["answer:", "final answer:", "response:"]:
            if s.lower().startswith(prefix):
                return s.split(":", 1)[1].strip()
        return s

    def maybe_extract_numeric_tail(s):
        matches = re.findall(r"[-+]?\d*\.?\d+(?:e[-+]?\d+)?", s, flags=re.IGNORECASE)
        if matches:
            return matches[-1]
        return s

    pred = strip_wrappers(pred)
    ref = strip_wrappers(ref)
    pred_norm = normalize_math_str(pred)
    ref_norm = normalize_math_str(ref)
    if pred_norm == ref_norm:
        return True

    transformations = standard_transformations + (
        implicit_multiplication_application,
    )

    def try_parse_to_sympy(raw, normed):
        try:
            from sympy.parsing.latex import parse_latex

            return parse_latex(raw)
        except Exception:
            pass
        try:
            return parse_expr(normed, transformations=transformations)
        except Exception:
            try:
                return sympy.sympify(normed)
            except Exception:
                return None

    def extract_and_eval(raw, normed):
        expr = try_parse_to_sympy(raw, normed)
        if expr is not None:
            try:
                return float(sympy.N(expr, 30))
            except Exception:
                pass
        try:
            return float(sympy.N(sympy.sympify(normed), 30))
        except Exception:
            pass
        numeric_tail = maybe_extract_numeric_tail(normed)
        if numeric_tail != normed:
            try:
                return float(numeric_tail)
            except Exception:
                pass
        return None

    p_expr = try_parse_to_sympy(pred, pred_norm)
    r_expr = try_parse_to_sympy(ref, ref_norm)

    if p_expr is not None and r_expr is not None:
        try:
            if p_expr.equals(r_expr):
                return True
        except Exception:
            pass
        try:
            diff_val = abs(float(sympy.N(p_expr - r_expr, 30)))
            if diff_val < 1e-6:
                return True
        except Exception:
            pass

    p_val = extract_and_eval(pred, pred_norm)
    r_val = extract_and_eval(ref, ref_norm)
    if p_val is not None and r_val is not None:
        abs_tol = 1e-6
        rel_tol = 1e-6 * max(1.0, abs(r_val), abs(p_val))
        if abs(p_val - r_val) <= max(abs_tol, rel_tol):
            return True

    return False


# =================================================================
# Wolfram Alpha Action wrapped as a lagent BaseAction
# =================================================================
PLANNER_SYSTEM_PROMPT_SUFFIX = """
IMPORTANT FORMAT RULES FOR REWOO PLANNING:
1. Every action must be written exactly as `#E1 = ToolName[input]`, `#E2 = ToolName[input]`.
2. Do NOT write `#E[1]`, `#E[2]`, `expression = ...`, markdown code fences, or natural-language-only plans.
3. Every `Plan:` line must be immediately followed by exactly one `#E<number> = ToolName[input]` line.
4. Use only plain tool input inside brackets, for example:
   Plan: compute x
   #E1 = WolframAlpha[2+2]
   Plan: combine previous results
   #E2 = WolframAlpha[#E1 + 3]
5. If the problem is mathematical, you should usually call `WolframAlpha[...]` at least once before the final answer.
6. Output only the plan/action blocks. Do not add explanations, blank summaries, or final answers.
7. Valid example:
   Plan: simplify the target expression
   #E1 = WolframAlpha[simplify (8/6)]
""".strip()


QWEN_PLANNER_FALLBACK_PROMPT = """
You are formatting a ReWOO worker plan.
Return ONLY lines in the following exact pattern:
Plan: <short reasoning step>
#E1 = WolframAlpha[<math expression or query>]
Plan: <short reasoning step>
#E2 = WolframAlpha[#E1]
Do not answer the math question directly.
Do not output prose, code fences, bullet points, or a final answer.
If the problem is math, produce at least one WolframAlpha action.
""".strip()


REWOO_ACTION_PATTERN = re.compile(
    r"Plan:\s*.+?\n\s*#E\d+\s*=\s*[A-Za-z_][A-Za-z0-9_]*\[.*?\]",
    flags=re.DOTALL,
)


def has_valid_rewoo_actions(text: str) -> bool:
    if not isinstance(text, str) or not text.strip():
        return False
    return bool(REWOO_ACTION_PATTERN.search(text))



def normalize_planner_response(text: str) -> str:
    text = "" if text is None else str(text)
    text = re.sub(r"```(?:[a-zA-Z0-9_+-]*)", "", text)
    text = text.replace("```", "")
    text = re.sub(r"#E\[(\d+)\]", r"#E\1", text)
    text = re.sub(
        r"(#E\d+\s*=\s*[A-Za-z_][A-Za-z0-9_]*\s*)\[\s*expression\s*=\s*",
        r"\1[",
        text,
    )
    text = re.sub(r'\[\s*"([\s\S]*?)"\s*\]', r'[\1]', text)
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()



def sanitize_wolfram_expression(expression) -> str:
    expr = "" if expression is None else str(expression)
    expr = expr.strip()
    expr = expr.replace("×", "*").replace("÷", "/")
    expr = expr.replace("−", "-")
    expr = expr.replace("^", "**")
    expr = expr.replace("\\cdot", "*")
    expr = expr.replace("\\times", "*")
    expr = expr.replace("\\frac", "frac")
    expr = re.sub(r"\s+", " ", expr)
    expr = expr.strip("` \n\t\"")
    if expr.startswith("expression="):
        expr = expr.split("=", 1)[1].strip()
    if expr.startswith("expression ="):
        expr = expr.split("=", 1)[1].strip()
    return expr


def make_action_result_text(text) -> list:
    content = "" if text is None else str(text)
    return [{"type": "text", "content": content}]


def safe_format_action_return(action_return: ActionReturn) -> str:
    if action_return is None:
        return ""

    result = getattr(action_return, "result", None)
    if isinstance(result, list):
        chunks = []
        for item in result:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    chunks.append(str(item.get("content", "")))
                else:
                    chunks.append(str(item.get("content", item)))
            else:
                chunks.append(str(item))
        return "".join(chunks).strip()

    if isinstance(result, dict):
        if "text" in result:
            return str(result.get("text", "")).strip()
        if "content" in result:
            return str(result.get("content", "")).strip()
        return str(result).strip()

    if result is not None:
        return str(result).strip()

    errmsg = getattr(action_return, "errmsg", "")
    return "" if errmsg is None else str(errmsg).strip()



def build_action_input(action_name: str, current_input: str):
    if action_name == "WolframAlpha":
        return {"expression": sanitize_wolfram_expression(current_input)}
    return current_input


class WolframAlpha(BaseAction):
    """Use the Wolfram Alpha Short Answers API for math queries.

    Supports arithmetic, algebra, calculus, equation solving, and more.
    Example expressions:
        - "15 / 3"
        - "solve x^2 + 4*x = 3 for x"
        - "factor x^2 + 5*x + 6"
        - "integrate sin(x) dx"
    """

    def __init__(self, appid: str = "", timeout: tuple = (5, 20)):
        # Build the tool description dictionary required by lagent BaseAction
        description = {
            "name": "WolframAlpha",
            "description": (
                "Wolfram Alpha computational knowledge engine. "
                "Use this tool for mathematical calculations, equation solving, "
                "calculus, algebra, and any symbolic/numeric computation."
            ),
            "parameters": [
                {
                    "name": "expression",
                    "type": "STRING",
                    "description": (
                        "The mathematical expression or query to compute. "
                        "Examples: '15 / 3', 'solve x^2+4*x=3 for x', "
                        "'integrate sin(x) dx', 'factor x^2+5*x+6'"
                    ),
                }
            ],
            "required": ["expression"],  # lagent expects a top-level parameter name list
        }
        super().__init__(description=description)
        self.appid = appid
        self.wolfram_url = "http://api.wolframalpha.com/v1/result"
        self.timeout = timeout
        self.http_session = requests.Session()

    def run(self, expression: str) -> ActionReturn:
        """Execute a Wolfram Alpha query.

        Args:
            expression (str): Mathematical expression or query.

        Returns:
            ActionReturn: Status object containing the query result.
        """
        expr = sanitize_wolfram_expression(expression)
        if not self.appid:
            return ActionReturn(
                args={"expression": expr},
                type=self.name,
                errmsg="No Wolfram AppID configured.",
                state=ActionStatusCode.API_ERROR,
            )
        if not expr:
            return ActionReturn(
                args={"expression": expr},
                type=self.name,
                errmsg="Empty Wolfram expression after sanitization.",
                state=ActionStatusCode.API_ERROR,
            )
        if len(expr) > 512:
            return ActionReturn(
                args={"expression": expr},
                type=self.name,
                errmsg="Wolfram expression too long.",
                state=ActionStatusCode.API_ERROR,
            )

        params = {"appid": self.appid, "i": expr}

        try:
            response = self.http_session.get(
                self.wolfram_url, params=params, timeout=self.timeout
            )
            if response.status_code == 501:
                return ActionReturn(
                    args={"expression": expr},
                    type=self.name,
                    result=make_action_result_text(
                        "Wolfram could not understand the query."
                    ),
                )
            response.raise_for_status()
            answer = response.text.strip() or "(empty response)"
            return ActionReturn(
                args={"expression": expr},
                type=self.name,
                result=make_action_result_text(answer),
            )
        except requests.exceptions.Timeout:
            return ActionReturn(
                args={"expression": expr},
                type=self.name,
                errmsg="Wolfram API timed out.",
                state=ActionStatusCode.API_ERROR,
            )
        except Exception as e:
            return ActionReturn(
                args={"expression": expr},
                type=self.name,
                errmsg=f"Wolfram API error: {e}",
                state=ActionStatusCode.API_ERROR,
            )


# =================================================================
# ReWOO staged execution with streaming planner / solver output
# =================================================================
def stream_rewoo_chat(agent: ReWOO, message, **kwargs) -> AgentReturn:
    if isinstance(message, str):
        inner_history = [dict(role="user", content=message)]
        question_text = message
    elif isinstance(message, dict):
        inner_history = [message]
        question_text = message.get("content", "")
    elif isinstance(message, list):
        inner_history = message[:]
        question_text = message[-1].get("content", "") if message else ""
    else:
        raise TypeError(f"unsupported type: {type(message)}")

    offset = len(inner_history)
    agent_return = AgentReturn()
    stats = {
        "planner_time": 0.0,
        "solver_time": 0.0,
        "worker_time": 0.0,
        "tool_time": 0.0,
        "planner_tokens": 0,
        "solver_tokens": 0,
        "total_generated_tokens": 0,
        "planner_tps": 0.0,
        "solver_tps": 0.0,
        "overall_tps": 0.0,
        "tool_call_count": 0,
    }

    print(f"[Question] {question_text}", flush=True)
    print("[Planner] Generating plan...", flush=True)
    turn_id = 0
    reformat_request = ""
    planner_response = ""
    thoughts = []
    actions = []
    actions_input = []
    planner_start = time.time()

    while turn_id < agent.max_turn:
        planner_prompt = agent._protocol.format_planner(
            chat_history=[],
            inner_step=inner_history,
            action_executor=agent._action_executor,
            reformat_request=reformat_request,
        )
        if planner_prompt and isinstance(planner_prompt, list):
            planner_prompt[0]["content"] += "\n\n" + PLANNER_SYSTEM_PROMPT_SUFFIX
            if reformat_request:
                planner_prompt[0]["content"] += "\n\n" + QWEN_PLANNER_FALLBACK_PROMPT

        planner_response = ""
        for model_state, res, _ in agent._llm.stream_generate(planner_prompt, **kwargs):
            if model_state == ModelStatusCode.STREAM_ING:
                delta = res[len(planner_response):]
                if delta:
                    print(delta, end="", flush=True)
                planner_response = res
            elif model_state == ModelStatusCode.END:
                if len(res) > len(planner_response):
                    print(res[len(planner_response):], end="", flush=True)
                planner_response = res
        print("", flush=True)

        planner_response = normalize_planner_response(planner_response)
        inner_history.append(dict(role="assistant", content=planner_response))
        try:
            thoughts, actions, actions_input = agent._protocol.parse_worker(planner_response)
            if not actions:
                lower_response = planner_response.lower()
                if "could you explain" in lower_response:
                    raise ValueError("Planner misunderstood the prompt and asked for clarification.")
                if not has_valid_rewoo_actions(planner_response):
                    raise ValueError(
                        "Planner did not output any valid `Plan:` + `#E = Tool[...]` action block."
                    )
            break
        except Exception as e:
            turn_id += 1
            reformat_request = (
                str(e)
                + " Please strictly use the format `Plan: ...` followed by `#E1 = Tool[input]`. "
                + "Do not use square brackets in the evidence id like `#E[1]`, do not write `expression = ...`, "
                + "and do not answer directly without at least one valid action block."
            )
            print(f"[Planner] Plan parsing failed on retry {turn_id}: {e}", flush=True)

    stats["planner_time"] = time.time() - planner_start
    stats["planner_tokens"] = len(
        agent._llm.tokenizer.encode(planner_response, add_special_tokens=False)
    )
    stats["planner_tps"] = (
        stats["planner_tokens"] / stats["planner_time"]
        if stats["planner_time"] > 0
        else 0.0
    )

    if turn_id >= agent.max_turn:
        print(
            f"[Planner] Failed to parse a valid plan after {agent.max_turn} attempts; falling back to solver stage.",
            flush=True,
        )
        thoughts = []
        actions = []
        action_responses = []
    else:
        print(f"[Planner] Plan parsed successfully with {len(actions)} actions.", flush=True)
        action_responses = []

    if actions:
        print("[Worker] Executing tool calls...", flush=True)
    worker_start = time.time()
    for action_id in range(len(actions)):
        current_input = actions_input[action_id]
        prev_ptrs = re.findall(r"#E\d+", current_input)
        for prev_ptr in prev_ptrs:
            ptr_num = int(prev_ptr.strip("#E")) - 1
            current_input = current_input.replace(
                prev_ptr, safe_format_action_return(action_responses[ptr_num])
            )

        print(
            f"[Worker {action_id + 1}/{len(actions)}] thought={thoughts[action_id]}",
            flush=True,
        )
        print(
            f"[Worker {action_id + 1}/{len(actions)}] {actions[action_id]}[{current_input}]",
            flush=True,
        )
        tool_start = time.time()
        action_input = build_action_input(actions[action_id], current_input)
        print(
            f"[Worker {action_id + 1}/{len(actions)} Sanitized] {action_input}",
            flush=True,
        )
        action_return = agent._action_executor(actions[action_id], action_input)
        stats["tool_time"] += time.time() - tool_start
        stats["tool_call_count"] += 1
        action_responses.append(action_return)

        if action_return.state == ActionStatusCode.SUCCESS:
            action_text = safe_format_action_return(action_return)
        else:
            action_text = action_return.errmsg
        print(f"[Worker {action_id + 1} Result] {action_text}", flush=True)
    stats["worker_time"] = time.time() - worker_start

    solver_prompt, worker_log = agent._protocol.format_solver(
        question_text, thoughts, action_responses
    )
    inner_history.append(dict(role="system", content=worker_log))

    print("[Solver] Generating final answer...", flush=True)
    final_response = ""
    solver_messages = [{"role": "user", "content": solver_prompt}]
    solver_start = time.time()
    planner_prefix = None
    if planner_prompt and isinstance(planner_prompt, list):
        planner_prefix = planner_prompt[:]
        if planner_prefix and planner_prefix[-1].get("role") == "assistant":
            planner_prefix[-1] = {
                **planner_prefix[-1],
                "content": planner_response,
            }
        else:
            planner_prefix.append({"role": "assistant", "content": planner_response})
    solver_input = solver_messages
    if planner_prefix:
        solver_input = planner_prefix + [{"role": "user", "content": solver_prompt}]
    for model_state, res, _ in agent._llm.stream_generate(solver_input, **kwargs):
        if model_state == ModelStatusCode.STREAM_ING:
            delta = res[len(final_response):]
            if delta:
                print(delta, end="", flush=True)
            final_response = res
        elif model_state == ModelStatusCode.END:
            if len(res) > len(final_response):
                print(res[len(final_response):], end="", flush=True)
            final_response = res
    print("", flush=True)

    stats["solver_time"] = time.time() - solver_start
    stats["solver_tokens"] = len(
        agent._llm.tokenizer.encode(final_response, add_special_tokens=False)
    )
    stats["solver_tps"] = (
        stats["solver_tokens"] / stats["solver_time"]
        if stats["solver_time"] > 0
        else 0.0
    )
    stats["total_generated_tokens"] = (
        stats["planner_tokens"] + stats["solver_tokens"]
    )
    total_generation_time = stats["planner_time"] + stats["solver_time"]
    stats["overall_tps"] = (
        stats["total_generated_tokens"] / total_generation_time
        if total_generation_time > 0
        else 0.0
    )

    inner_history.append(dict(role="assistant", content=final_response))
    agent_return.inner_steps = inner_history[offset:]
    agent_return.response = final_response
    agent_return.extra_info = stats
    return agent_return


# =================================================================
# Main evaluation entry for lagent ReWOO + Wolfram
# =================================================================
def build_arg_parser():
    parser = argparse.ArgumentParser(description="Anonymous ReWOO math evaluation runner.")
    parser.add_argument("--model-path", default=os.getenv("MODEL_PATH", "./models/rewoo_model"), help="Path to the AWQ model directory.")
    parser.add_argument("--dataset-path", default=os.getenv("DATASET_PATH", "./data/math_eval.arrow"), help="Path to the evaluation dataset file.")
    parser.add_argument("--result-csv", default=os.getenv("RESULT_CSV_PATH", "rewoo_evaluation_results.csv"), help="Output CSV path for evaluation results.")
    parser.add_argument("--selection-pool", type=int, default=int(os.getenv("SELECTION_POOL", "10000")), help="Number of ranked candidate samples to keep.")
    parser.add_argument("--sample-count", type=int, default=int(os.getenv("SAMPLE_COUNT", "10000")), help="Number of selected samples to evaluate.")
    parser.add_argument("--max-turn", type=int, default=int(os.getenv("MAX_TURN", "3")), help="Maximum planner retry turns for the ReWOO agent.")
    parser.add_argument("--max-new-tokens", type=int, default=int(os.getenv("MAX_NEW_TOKENS", "512")), help="Maximum new tokens for planner and solver generation.")
    parser.add_argument("--decode-interval", type=int, default=int(os.getenv("DECODE_INTERVAL", "8")), help="Streaming decode interval for partial text emission.")
    parser.add_argument("--planner-system-suffix", default=os.getenv("PLANNER_SYSTEM_SUFFIX", PLANNER_SYSTEM_PROMPT_SUFFIX), help="Planner formatting suffix appended to the planner system prompt.")
    parser.add_argument("--planner-fallback-prompt", default=os.getenv("PLANNER_FALLBACK_PROMPT", QWEN_PLANNER_FALLBACK_PROMPT), help="Fallback planner prompt used after parse failures.")
    parser.add_argument("--stream-output", action="store_true", default=os.getenv("STREAM_OUTPUT", "true").lower() in {"1", "true", "yes", "on"}, help="Stream planner, worker, and solver output to stdout.")
    parser.add_argument("--torch-dtype", default=os.getenv("TORCH_DTYPE", "float16"), choices=["float16", "bfloat16", "float32"], help="Torch dtype used when building the AWQ-backed llm wrapper.")
    return parser


def main():
    args = build_arg_parser().parse_args()
    model_path = args.model_path
    dataset_path = args.dataset_path
    log_csv = args.result_csv

    planner_system_prompt_suffix = args.planner_system_suffix
    planner_fallback_prompt = args.planner_fallback_prompt

    # ---- Wolfram AppID ----
    wolfram_appid = os.getenv("WOLFRAM_APPID", "").strip()
    if not wolfram_appid:
        print("WARNING: WOLFRAM_APPID is not set. The tool backend will not work.")
        print("Set it with: export WOLFRAM_APPID='your-app-id'")

    # ---- Load dataset ----
    print(">>> Loading dataset from configured path...")
    full_dataset = Dataset.from_file(dataset_path)
    print(">>> Prioritizing Level 4+ problems and backfilling with lower levels if needed...")

    def get_level_score(example):
        level_raw = str(example.get("level", "")).strip()
        digits = "".join(ch for ch in level_raw if ch.isdigit())
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
        question_text = entry.get("problem", entry.get("question", ""))
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
        selected_indices.extend(idx for idx, _, _ in fallback_entries[:remain])

    if not selected_indices:
        print(">>> No valid level field found; falling back to ranking by problem length...")
        ranked = sorted(
            enumerate(full_dataset),
            key=lambda item: len(
                item[1].get("problem", item[1].get("question", ""))
            ),
            reverse=True,
        )
        selected_indices = [idx for idx, _ in ranked[:args.selection_pool]]

    num_test = min(args.sample_count, len(selected_indices))
    test_data = full_dataset.select(selected_indices[:num_test])
    print(f">>> Preparing to evaluate {num_test} samples.")

    # ---- Build lagent components ----
    print(f">>> Loading AWQ model from configured path...")

    torch_dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    selected_dtype = torch_dtype_map[args.torch_dtype]

    # 1) LLM wrapper built with the AWQ-compatible subclass
    llm = HFTransformerAWQ(
        path=model_path,
        tokenizer_kwargs={},
        model_kwargs={"device_map": "auto", "torch_dtype": selected_dtype},
    )

    # 2) Wolfram tool
    wolfram_action = WolframAlpha(appid=wolfram_appid)

    # 3) ActionExecutor
    action_executor = ActionExecutor(actions=[wolfram_action])

    # 4) ReWOO protocol
    protocol = ReWOOProtocol()

    # 5) ReWOO agent
    agent = ReWOO(
        llm=llm,
        action_executor=action_executor,
        protocol=protocol,
        max_turn=args.max_turn,
    )

    global PLANNER_SYSTEM_PROMPT_SUFFIX, QWEN_PLANNER_FALLBACK_PROMPT
    PLANNER_SYSTEM_PROMPT_SUFFIX = planner_system_prompt_suffix
    QWEN_PLANNER_FALLBACK_PROMPT = planner_fallback_prompt

    print(">>> lagent ReWOO agent is ready. Starting evaluation...")

    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "decode_interval": args.decode_interval,
    }

    # ---- Evaluation loop ----
    results = []
    correct_count = 0

    for i, entry in enumerate(test_data):
        question = entry.get("problem", entry.get("question", ""))
        reference_full = str(entry.get("solution", entry.get("answer", "")))

        print(f"\n\n[Progress] {i+1}/{num_test}")
        print(f"[Question]: {question[:args.sample_count]}...")

        t_start = time.time()

        try:
            # Three-stage ReWOO inference with streamed Plan -> Work -> Solve output
            agent_return = stream_rewoo_chat(agent, question, **generation_kwargs)

            final_answer_text = agent_return.response
            t_total = time.time() - t_start
            run_stats = getattr(agent_return, "extra_info", {}) or {}

            # Extract staged information from inner_steps
            inner_steps = agent_return.inner_steps
            # inner_steps typically looks like:
            #   [assistant_msg (plan), system_msg (worker_log), assistant_msg (solver)]
            plan_text = ""
            worker_log = ""
            num_actions = 0
            assistant_steps = []

            for step in inner_steps:
                role = step.get("role", "")
                content = step.get("content", "")
                if role == "assistant":
                    assistant_steps.append(content)
                    if not plan_text:
                        plan_text = content
                elif role == "system":
                    worker_log = content

            if plan_text:
                num_actions = len(re.findall(r"#E\d+", plan_text))

            print("\n[Inner Steps]")
            for idx, step in enumerate(inner_steps, 1):
                role = step.get("role", "")
                content = step.get("content", "")
                print(f"--- step {idx} | role={role} ---")
                print(content)

            if not plan_text and assistant_steps:
                plan_text = assistant_steps[0]
            if not final_answer_text and assistant_steps:
                final_answer_text = assistant_steps[-1]

            print(f"\n[Plan]:\n{plan_text}")
            if worker_log:
                print(f"[Worker Log]:\n{worker_log}")
            print(f"[Final Answer]:\n{final_answer_text}")

            # Compare answers
            pred_ans = extract_boxed_content(final_answer_text)
            if not pred_ans:
                pred_ans = final_answer_text.strip()
            ref_ans = extract_boxed_content(reference_full)
            if not ref_ans:
                ref_ans = reference_full.strip()

            is_hit = is_equivalent(pred_ans, ref_ans)
            if is_hit:
                correct_count += 1

            print(
                f"\n[Evaluation] pred: {pred_ans} | ref: {ref_ans} | "
                f"{'correct' if is_hit else 'incorrect'}"
            )
            total_tokens = int(run_stats.get("total_generated_tokens", 0))
            planner_tokens = int(run_stats.get("planner_tokens", 0))
            solver_tokens = int(run_stats.get("solver_tokens", 0))
            overall_tps = float(run_stats.get("overall_tps", 0.0))
            planner_tps = float(run_stats.get("planner_tps", 0.0))
            solver_tps = float(run_stats.get("solver_tps", 0.0))
            planner_time = float(run_stats.get("planner_time", 0.0))
            solver_time = float(run_stats.get("solver_time", 0.0))
            worker_time = float(run_stats.get("worker_time", 0.0))
            tool_time = float(run_stats.get("tool_time", 0.0))
            tool_call_count = int(run_stats.get("tool_call_count", 0))

            print(
                f"[Timing] total={t_total:.2f}s | planner={planner_time:.2f}s | "
                f"worker={worker_time:.2f}s | solver={solver_time:.2f}s"
            )
            print(
                f"[Throughput] total_tokens={total_tokens} | planner_tokens={planner_tokens} | "
                f"solver_tokens={solver_tokens} | overall_tps={overall_tps:.2f} | "
                f"planner_tps={planner_tps:.2f} | solver_tps={solver_tps:.2f}"
            )
            print(
                f"[Tool Stats] actions={num_actions} | tool_calls={tool_call_count} | "
                f"tool_time={tool_time:.2f}s"
            )

            results.append(
                {
                    "id": i,
                    "question": question,
                    "ref_ans": ref_ans,
                    "pred_ans": pred_ans,
                    "is_correct": is_hit,
                    "duration": round(t_total, 2),
                    "planner_time": round(planner_time, 2),
                    "worker_time": round(worker_time, 2),
                    "solver_time": round(solver_time, 2),
                    "tool_exec_time": round(tool_time, 2),
                    "total_tokens": total_tokens,
                    "planner_tokens": planner_tokens,
                    "solver_tokens": solver_tokens,
                    "tokens_per_sec": round(overall_tps, 2),
                    "planner_tps": round(planner_tps, 2),
                    "solver_tps": round(solver_tps, 2),
                    "tool_call_count": tool_call_count,
                    "num_actions": num_actions,
                }
            )

        except Exception as e:
            print(f"Error processing case {i}: {e}")
            import traceback

            traceback.print_exc()
            continue

    # ---- Final summary ----
    if results:
        df = pd.DataFrame(results)
        df.to_csv(log_csv, index=False, encoding="utf-8-sig")
        df.to_csv(log_csv, index=False, encoding="utf-8-sig")
        print(f"\n{'#'*20} Evaluation Complete {'#'*20}")
        print(f"Average total duration: {df['duration'].mean():.2f}s")
        print(f"Average planner duration: {df['planner_time'].mean():.2f}s")
        print(f"Average worker duration: {df['worker_time'].mean():.2f}s")
        print(f"Average solver duration: {df['solver_time'].mean():.2f}s")
        print(f"Average TPS (total generation): {df['tokens_per_sec'].mean():.2f}")
        print(f"Average planner TPS: {df['planner_tps'].mean():.2f}")
        print(f"Average solver TPS: {df['solver_tps'].mean():.2f}")
        print(f"Average tool execution time: {df['tool_exec_time'].mean():.2f}s")
        print(f"Total tool calls: {int(df['tool_call_count'].sum())}")
        print(f"Average duration: {df['duration'].mean():.2f}s")
        print(f"Average tool actions: {df['num_actions'].mean():.1f}")
        print(
            f"Accuracy: {correct_count}/{len(results)} = "
            f"{correct_count/len(results):.2%}"
        )
        print(f"Results saved to: {log_csv}")


if __name__ == "__main__":
    main()