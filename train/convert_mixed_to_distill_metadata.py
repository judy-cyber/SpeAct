import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List


DEFAULT_PROMPT_TEMPLATE = "Instruction: {instruction}\n\nInput: {input}\n\nOutput: "


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert mixed JSONL samples into metadata format for feature distillation pipelines."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to the mixed JSONL file produced by the dataset mixer.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to the output JSONL metadata file.",
    )
    parser.add_argument(
        "--prompt_template",
        type=str,
        default=DEFAULT_PROMPT_TEMPLATE,
        help="Prompt template used to construct the distillation question field.",
    )
    parser.add_argument(
        "--skip_metadata",
        action="store_true",
        help="Do not copy source metadata into the converted records.",
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
        return value.strip()
    return str(value).strip()


def build_question(record: Dict, prompt_template: str) -> str:
    instruction = normalize_text(record.get("instruction", ""))
    input_text = normalize_text(record.get("input", ""))
    return prompt_template.format(instruction=instruction, input=input_text).strip()


def convert_record(record: Dict, prompt_template: str, skip_metadata: bool) -> Dict:
    question_text = build_question(record, prompt_template)
    target_text = normalize_text(record.get("output", ""))

    converted = {
        "q": question_text,
        "t": target_text,
        "source_dataset": normalize_text(record.get("source_dataset", "unknown")),
        "feat_offset": -1,
        "feat_len": -1,
    }

    if not skip_metadata:
        converted["metadata"] = record.get("metadata", {})

    return converted


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    records = read_jsonl(input_path)
    converted_records = []

    for record in records:
        converted = convert_record(
            record=record,
            prompt_template=args.prompt_template,
            skip_metadata=args.skip_metadata,
        )
        if converted["q"] and converted["t"]:
            converted_records.append(converted)

    write_jsonl(output_path, converted_records)
    print(f"Converted {len(converted_records)} records to distillation metadata format: {output_path}")
    print("The generated metadata file still requires valid teacher feature offsets and lengths before it can be used by the distillation trainer.")


if __name__ == "__main__":
    main()
