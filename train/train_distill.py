import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


PROMPT_TEMPLATE = "Question: {query}\nOutput: "
DEFAULT_MAX_SEQUENCE_LENGTH = 512
DEFAULT_TEACHER_FEATURE_DIM = 8192
DEFAULT_LANGUAGE_LOSS_WEIGHT = 1.0
DEFAULT_FEATURE_LOSS_WEIGHT = 0.3
DEFAULT_TRAIN_BATCH_SIZE = 8
DEFAULT_GRADIENT_ACCUMULATION_STEPS = 4
DEFAULT_LEARNING_RATE = 1e-5
DEFAULT_NUM_TRAIN_EPOCHS = 2
DEFAULT_LOGGING_STEPS = 5
DEFAULT_WARMUP_RATIO = 0.1
DEFAULT_OUTPUT_PROJECTION_NAME = "projection_head.bin"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a causal language model with joint language modeling and feature distillation."
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        required=True,
        help="Path to the pretrained base model.",
    )
    parser.add_argument(
        "--metadata_path",
        type=str,
        required=True,
        help="Path to the JSONL metadata file for distillation samples.",
    )
    parser.add_argument(
        "--features_path",
        type=str,
        required=True,
        help="Path to the binary file containing teacher features.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where training artifacts will be saved.",
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=DEFAULT_MAX_SEQUENCE_LENGTH,
        help="Maximum sequence length used for tokenization and feature padding.",
    )
    parser.add_argument(
        "--teacher_feature_dim",
        type=int,
        default=DEFAULT_TEACHER_FEATURE_DIM,
        help="Hidden dimension of teacher feature vectors.",
    )
    parser.add_argument(
        "--language_loss_weight",
        type=float,
        default=DEFAULT_LANGUAGE_LOSS_WEIGHT,
        help="Weight for the language modeling loss.",
    )
    parser.add_argument(
        "--feature_loss_weight",
        type=float,
        default=DEFAULT_FEATURE_LOSS_WEIGHT,
        help="Weight for the feature alignment loss.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=DEFAULT_TRAIN_BATCH_SIZE,
        help="Training batch size per device.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=DEFAULT_GRADIENT_ACCUMULATION_STEPS,
        help="Number of gradient accumulation steps.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help="Optimizer learning rate.",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=float,
        default=DEFAULT_NUM_TRAIN_EPOCHS,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=DEFAULT_LOGGING_STEPS,
        help="Logging frequency in training steps.",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=DEFAULT_WARMUP_RATIO,
        help="Warmup ratio for the learning rate scheduler.",
    )
    parser.add_argument(
        "--disable_compile",
        action="store_true",
        help="Disable torch.compile even when it is available.",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Allow loading custom model code from the pretrained repository.",
    )
    return parser.parse_args()


# ==========================================
# 1. Projection module for feature alignment
# ==========================================
class FeatureProjectionHead(nn.Module):
    def __init__(self, student_hidden_size=3072, teacher_hidden_size=8192):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(student_hidden_size, 4096),
            nn.GELU(),
            nn.Linear(4096, teacher_hidden_size),
        )

    def forward(self, hidden_states):
        return self.projection(hidden_states)


# ==========================================
# 2. Dataset for distillation training
# ==========================================
class DistillationDataset(Dataset):
    def __init__(
        self,
        metadata_path,
        features_path,
        tokenizer,
        max_sequence_length=DEFAULT_MAX_SEQUENCE_LENGTH,
        teacher_feature_dim=DEFAULT_TEACHER_FEATURE_DIM,
    ):
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length
        self.features_path = features_path
        self.teacher_feature_dim = teacher_feature_dim

        with open(metadata_path, "r", encoding="utf-8") as metadata_file:
            self.samples = [json.loads(line) for line in metadata_file]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        prompt_text = PROMPT_TEMPLATE.format(query=sample["q"])
        target_text = f"{sample['t']}{self.tokenizer.eos_token}"
        full_text = prompt_text + target_text

        tokenized = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_sequence_length,
            padding="max_length",
            return_tensors="pt",
        )

        prompt_token_ids = self.tokenizer(prompt_text, add_special_tokens=False).input_ids
        response_start_index = len(prompt_token_ids)

        with open(self.features_path, "rb") as feature_file:
            feature_file.seek(sample["feat_offset"])
            expected_num_bytes = sample["feat_len"] * self.teacher_feature_dim * 2
            feature_bytes = feature_file.read(expected_num_bytes)

        available_num_tokens = len(feature_bytes) // (self.teacher_feature_dim * 2)
        teacher_features = np.frombuffer(feature_bytes, dtype=np.float16).astype(np.float32)
        teacher_features = teacher_features[: available_num_tokens * self.teacher_feature_dim]
        teacher_features = teacher_features.reshape(available_num_tokens, self.teacher_feature_dim)

        padded_teacher_features = np.zeros(
            (self.max_sequence_length, self.teacher_feature_dim), dtype=np.float32
        )
        response_end_index = min(response_start_index + available_num_tokens, self.max_sequence_length)
        aligned_token_count = response_end_index - response_start_index

        if aligned_token_count > 0:
            padded_teacher_features[response_start_index:response_end_index, :] = teacher_features[
                :aligned_token_count, :
            ]

        return {
            "input_ids": tokenized.input_ids.squeeze(0),
            "attention_mask": tokenized.attention_mask.squeeze(0),
            "labels": tokenized.input_ids.squeeze(0),
            "teacher_hidden_states": torch.tensor(padded_teacher_features),
        }


# ==========================================
# 3. Custom trainer for joint language and feature distillation
# ==========================================
class DistillationTrainer(Trainer):
    def __init__(self, projection_head, language_loss_weight=1.0, feature_loss_weight=0.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.projection_head = projection_head.to(self.model.device)
        self.language_loss_weight = language_loss_weight
        self.feature_loss_weight = feature_loss_weight
        self.feature_loss_fn = nn.MSELoss()

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        teacher_hidden_states = inputs.pop("teacher_hidden_states")

        outputs = model(**inputs, output_hidden_states=True)
        student_hidden_states = outputs.hidden_states[-1]

        language_modeling_loss = outputs.loss
        projected_hidden_states = self.projection_head(student_hidden_states.to(torch.bfloat16))

        valid_feature_mask = (
            torch.abs(teacher_hidden_states).sum(dim=-1, keepdim=True) > 0
        ).float()

        feature_alignment_loss = self.feature_loss_fn(
            projected_hidden_states * valid_feature_mask,
            teacher_hidden_states.to(model.device).to(torch.bfloat16) * valid_feature_mask,
        )

        total_loss = (
            self.language_loss_weight * language_modeling_loss
            + self.feature_loss_weight * feature_alignment_loss
        )

        return (total_loss, outputs) if return_outputs else total_loss


# ==========================================
# 4. Training entry point
# ==========================================
def train(args):
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    tokenizer.pad_token = tokenizer.eos_token

    print(">>> Loading base model weights...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=args.trust_remote_code,
    )

    training_model = base_model
    if hasattr(torch, "compile") and not args.disable_compile:
        print(">>> Enabling torch.compile for faster execution...")
        training_model = torch.compile(base_model)

    base_model.gradient_checkpointing_enable()

    projection_head = FeatureProjectionHead(
        student_hidden_size=base_model.config.hidden_size,
        teacher_hidden_size=args.teacher_feature_dim,
    ).to(torch.bfloat16)

    train_dataset = DistillationDataset(
        metadata_path=args.metadata_path,
        features_path=args.features_path,
        tokenizer=tokenizer,
        max_sequence_length=args.max_sequence_length,
        teacher_feature_dim=args.teacher_feature_dim,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        bf16=True,
        logging_steps=args.logging_steps,
        save_strategy="epoch",
        gradient_checkpointing=True,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = DistillationTrainer(
        model=training_model,
        projection_head=projection_head,
        args=training_args,
        train_dataset=train_dataset,
        language_loss_weight=args.language_loss_weight,
        feature_loss_weight=args.feature_loss_weight,
    )

    print(">>> Starting joint distillation training (language modeling + feature alignment)...")
    trainer.train()

    print(">>> Saving export-friendly model artifacts...")
    tokenizer.save_pretrained(args.output_dir)
    base_model.save_pretrained(args.output_dir, safe_serialization=True)
    torch.save(
        projection_head.state_dict(),
        os.path.join(args.output_dir, DEFAULT_OUTPUT_PROJECTION_NAME),
    )
    print(f">>> Training finished. Artifacts saved to: {args.output_dir}")


if __name__ == "__main__":
    train(parse_args())
