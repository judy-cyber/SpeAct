import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_PROMPT_TEMPLATE = "Question: {question}\nOutput: "
DEFAULT_MAX_SEQUENCE_LENGTH = 512
DEFAULT_DTYPE = "float16"
SUPPORTED_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract teacher hidden-state features and build distillation metadata files."
    )
    parser.add_argument(
        "--teacher_model_path",
        type=str,
        required=True,
        help="Path to the teacher model.",
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to input JSONL data. Each record must contain q and t fields.",
    )
    parser.add_argument(
        "--output_metadata_path",
        type=str,
        required=True,
        help="Path to the output metadata JSONL file.",
    )
    parser.add_argument(
        "--output_features_path",
        type=str,
        required=True,
        help="Path to the output binary feature file.",
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=DEFAULT_MAX_SEQUENCE_LENGTH,
        help="Maximum sequence length used for tokenization.",
    )
    parser.add_argument(
        "--feature_dtype",
        type=str,
        default=DEFAULT_DTYPE,
        choices=sorted(SUPPORTED_DTYPES.keys()),
        help="Data type used when saving teacher features.",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Allow loading custom model code from the pretrained repository.",
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default="auto",
        help="Device map passed to from_pretrained.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> List[Dict]:
    records: List[Dict] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def normalize_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def build_full_text(question_text: str, target_text: str, tokenizer) -> Tuple[str, str]:
    prompt_text = DEFAULT_PROMPT_TEMPLATE.format(question=question_text)
    response_text = f"{target_text}{tokenizer.eos_token}"
    return prompt_text, prompt_text + response_text


def extract_response_hidden_states(
    model,
    tokenizer,
    question_text: str,
    target_text: str,
    max_sequence_length: int,
) -> np.ndarray:
    prompt_text, full_text = build_full_text(question_text, target_text, tokenizer)

    tokenized = tokenizer(
        full_text,
        truncation=True,
        max_length=max_sequence_length,
        return_tensors="pt",
    )
    prompt_token_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
    response_start_index = len(prompt_token_ids)

    tokenized = {key: value.to(model.device) for key, value in tokenized.items()}

    with torch.no_grad():
        outputs = model(**tokenized, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1][0]

    sequence_length = hidden_states.shape[0]
    response_end_index = min(sequence_length, max_sequence_length)

    if response_start_index >= response_end_index:
        return np.zeros((0, hidden_states.shape[-1]), dtype=np.float32)

    response_hidden_states = hidden_states[response_start_index:response_end_index]
    return response_hidden_states.detach().float().cpu().numpy()


def convert_feature_dtype(array: np.ndarray, feature_dtype: str) -> np.ndarray:
    if feature_dtype == "float16":
        return array.astype(np.float16)
    if feature_dtype == "bfloat16":
        tensor = torch.from_numpy(array).to(torch.bfloat16)
        return tensor.view(torch.uint16).cpu().numpy()
    if feature_dtype == "float32":
        return array.astype(np.float32)
    raise ValueError(f"Unsupported feature dtype: {feature_dtype}")


def build_output_record(source_record: Dict, feature_offset: int, feature_length: int) -> Dict:
    output_record = dict(source_record)
    output_record["feat_offset"] = feature_offset
    output_record["feat_len"] = feature_length
    return output_record


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path)
    output_metadata_path = Path(args.output_metadata_path)
    output_features_path = Path(args.output_features_path)

    records = read_jsonl(input_path)

    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model_path)
    if tokenizer.eos_token is None:
        raise ValueError("The tokenizer must provide an eos_token.")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_dtype = SUPPORTED_DTYPES[args.feature_dtype] if args.feature_dtype != "float32" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.teacher_model_path,
        device_map=args.device_map,
        torch_dtype=model_dtype,
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()

    output_metadata_path.parent.mkdir(parents=True, exist_ok=True)
    output_features_path.parent.mkdir(parents=True, exist_ok=True)

    converted_records = []
    current_offset = 0

    with output_features_path.open("wb") as feature_file:
        for index, record in enumerate(records):
            question_text = normalize_text(record.get("q", ""))
            target_text = normalize_text(record.get("t", ""))

            if not question_text or not target_text:
                continue

            response_features = extract_response_hidden_states(
                model=model,
                tokenizer=tokenizer,
                question_text=question_text,
                target_text=target_text,
                max_sequence_length=args.max_sequence_length,
            )

            feature_length = int(response_features.shape[0])
            serialized_features = convert_feature_dtype(response_features, args.feature_dtype)
            feature_bytes = serialized_features.tobytes()
            feature_file.write(feature_bytes)

            converted_record = build_output_record(
                source_record=record,
                feature_offset=current_offset,
                feature_length=feature_length,
            )
            converted_records.append(converted_record)
            current_offset += len(feature_bytes)

            if (index + 1) % 100 == 0:
                print(f"Processed {index + 1} records...")

    write_jsonl(output_metadata_path, converted_records)
    print(f"Saved metadata to: {output_metadata_path}")
    print(f"Saved teacher features to: {output_features_path}")
    print(f"Processed {len(converted_records)} valid records in total.")


if __name__ == "__main__":
    main()
