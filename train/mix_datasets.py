import argparse
import json
import random
from pathlib import Path
from typing import Callable, Dict, List, Tuple


DEFAULT_OUTPUT_SIZE = 10000
DEFAULT_SEED = 42

DATASET_REGISTRY = {
    "fever": {
        "path": Path("datasets/FEVER/train.jsonl"),
        "format": "jsonl",
    },
    "hotpotqa": {
        "path": Path("datasets/hotpotqa/OpenDataLab___hotpot_qa/default-755683ef7995944c/0.0.0/master/hotpot_qa-train.arrow"),
        "format": "arrow_stream",
    },
    "strategyqa": {
        "path": Path("datasets/StrategyQA/strategyQA_train.json"),
        "format": "json",
    },
    "gsm8k": {
        "path": Path("datasets/AI-ModelScope___gsm8k/main-037a92825d6197fc/0.0.0/master/gsm8k-train.arrow"),
        "format": "arrow_stream",
    },
}

MATH_JSON_CANDIDATES = [
    Path("datasets/Math/train.json"),
    Path("datasets/Math/train.jsonl"),
    Path("datasets/Math/MATH_train.json"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mix two or three datasets into a single JSONL file with a target number of examples."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="Dataset names to mix. Supported: fever, hotpotqa, strategyqa, gsm8k, math",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to the output JSONL file.",
    )
    parser.add_argument(
        "--target_size",
        type=int,
        default=DEFAULT_OUTPUT_SIZE,
        help="Number of examples to generate in the mixed dataset.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed used for shuffling and sampling.",
    )
    parser.add_argument(
        "--allow_repeat",
        action="store_true",
        help="Allow repeated sampling when a dataset has fewer examples than required.",
    )
    return parser.parse_args()


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


# Arrow IPC stream files are used here, so we load them with pyarrow.ipc.open_stream.
def read_arrow_stream(path: Path) -> List[dict]:
    import pyarrow.ipc as ipc

    with path.open("rb") as file:
        reader = ipc.open_stream(file)
        table = reader.read_all()
    return table.to_pylist()


def read_jsonl(path: Path) -> List[dict]:
    records = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def resolve_math_path() -> Path:
    for candidate in MATH_JSON_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find a supported MATH file. Expected one of: "
        + ", ".join(str(path) for path in MATH_JSON_CANDIDATES)
    )


def load_math_records() -> List[dict]:
    math_path = resolve_math_path()
    if math_path.suffix == ".jsonl":
        return read_jsonl(math_path)

    content = read_json(math_path)
    if isinstance(content, list):
        return content
    if isinstance(content, dict):
        for key in ["train", "data", "examples", "records"]:
            if key in content and isinstance(content[key], list):
                return content[key]
    raise ValueError(f"Unsupported MATH file structure in: {math_path}")


def load_dataset_records(dataset_name: str) -> List[dict]:
    normalized_name = dataset_name.lower()

    if normalized_name == "math":
        return load_math_records()

    if normalized_name not in DATASET_REGISTRY:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    config = DATASET_REGISTRY[normalized_name]
    dataset_path = config["path"]
    dataset_format = config["format"]

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    if dataset_format == "jsonl":
        return read_jsonl(dataset_path)
    if dataset_format == "json":
        data = read_json(dataset_path)
        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON list in {dataset_path}")
        return data
    if dataset_format == "arrow_stream":
        return read_arrow_stream(dataset_path)

    raise ValueError(f"Unsupported dataset format: {dataset_format}")


def format_fever_record(record: dict) -> dict:
    answer = record.get("label", "UNKNOWN")
    metadata = {
        "id": record.get("id"),
        "verifiable": record.get("verifiable"),
        "evidence": record.get("evidence", []),
    }
    return {
        "source_dataset": "fever",
        "instruction": "Determine whether the claim is supported, refuted, or lacks enough information.",
        "input": record.get("claim", ""),
        "output": answer,
        "metadata": metadata,
    }


def format_hotpotqa_record(record: dict) -> dict:
    metadata = {
        "id": record.get("_id"),
        "level": record.get("level"),
        "type": record.get("type"),
        "supporting_facts": record.get("supporting_facts", []),
    }
    return {
        "source_dataset": "hotpotqa",
        "instruction": "Answer the multi-hop question based on the dataset annotation.",
        "input": record.get("question", ""),
        "output": str(record.get("answer", "")),
        "metadata": metadata,
    }


def format_strategyqa_record(record: dict) -> dict:
    metadata = {
        "qid": record.get("qid"),
        "term": record.get("term"),
        "description": record.get("description"),
        "facts": record.get("facts", []),
        "decomposition": record.get("decomposition", []),
    }
    return {
        "source_dataset": "strategyqa",
        "instruction": "Answer the question with yes or no.",
        "input": record.get("question", ""),
        "output": "yes" if record.get("answer") else "no",
        "metadata": metadata,
    }


def format_gsm8k_record(record: dict) -> dict:
    return {
        "source_dataset": "gsm8k",
        "instruction": "Solve the math word problem step by step.",
        "input": record.get("question", ""),
        "output": record.get("answer", ""),
        "metadata": {},
    }


def format_math_record(record: dict) -> dict:
    problem = record.get("problem") or record.get("question") or record.get("input") or ""
    solution = record.get("solution") or record.get("answer") or record.get("output") or ""
    metadata = {
        "level": record.get("level"),
        "type": record.get("type") or record.get("subject"),
    }
    return {
        "source_dataset": "math",
        "instruction": "Solve the math problem and provide the final answer.",
        "input": problem,
        "output": solution,
        "metadata": metadata,
    }


FORMATTERS: Dict[str, Callable[[dict], dict]] = {
    "fever": format_fever_record,
    "hotpotqa": format_hotpotqa_record,
    "strategyqa": format_strategyqa_record,
    "gsm8k": format_gsm8k_record,
    "math": format_math_record,
}


def normalize_records(dataset_name: str, records: List[dict]) -> List[dict]:
    formatter = FORMATTERS[dataset_name]
    normalized = []
    for record in records:
        item = formatter(record)
        if item["input"] and item["output"]:
            normalized.append(item)
    return normalized


def allocate_counts(dataset_names: List[str], target_size: int) -> Dict[str, int]:
    base_count = target_size // len(dataset_names)
    remainder = target_size % len(dataset_names)

    counts = {}
    for index, dataset_name in enumerate(dataset_names):
        counts[dataset_name] = base_count + (1 if index < remainder else 0)
    return counts


def sample_records(
    records: List[dict],
    sample_size: int,
    rng: random.Random,
    allow_repeat: bool,
) -> List[dict]:
    if sample_size <= len(records):
        return rng.sample(records, sample_size)

    if not allow_repeat:
        raise ValueError(
            f"Requested {sample_size} samples from a dataset with only {len(records)} examples. "
            "Use --allow_repeat to enable repeated sampling."
        )

    sampled = []
    while len(sampled) < sample_size:
        remaining = sample_size - len(sampled)
        if remaining >= len(records):
            shuffled = records[:]
            rng.shuffle(shuffled)
            sampled.extend(shuffled)
        else:
            sampled.extend(rng.sample(records, remaining))
    return sampled[:sample_size]


def write_jsonl(path: Path, records: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    dataset_names = [name.lower() for name in args.datasets]

    if len(dataset_names) not in {2, 3}:
        raise ValueError("Please provide exactly two or three datasets.")

    for name in dataset_names:
        if name not in FORMATTERS:
            raise ValueError(f"Unsupported dataset name: {name}")

    rng = random.Random(args.seed)
    per_dataset_counts = allocate_counts(dataset_names, args.target_size)

    mixed_records = []
    stats: List[Tuple[str, int, int]] = []

    for dataset_name in dataset_names:
        raw_records = load_dataset_records(dataset_name)
        normalized_records = normalize_records(dataset_name, raw_records)
        sampled_records = sample_records(
            records=normalized_records,
            sample_size=per_dataset_counts[dataset_name],
            rng=rng,
            allow_repeat=args.allow_repeat,
        )
        mixed_records.extend(sampled_records)
        stats.append((dataset_name, len(normalized_records), len(sampled_records)))

    rng.shuffle(mixed_records)
    output_path = Path(args.output_path)
    write_jsonl(output_path, mixed_records)

    print(f"Saved {len(mixed_records)} mixed samples to {output_path}")
    for dataset_name, available_count, sampled_count in stats:
        print(
            f"- {dataset_name}: available={available_count}, sampled={sampled_count}"
        )


if __name__ == "__main__":
    main()

