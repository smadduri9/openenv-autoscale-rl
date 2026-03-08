from __future__ import annotations

import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train SFT policy model.")
    p.add_argument("--dataset-path", type=Path, default=Path("sft_data/train.jsonl"))
    p.add_argument("--output-dir", type=Path, default=Path("sft_model"))
    p.add_argument("--model-name", type=str, default="unsloth/mistral-7b-instruct-v0.2-bnb-4bit")
    p.add_argument("--hf-fallback-model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    p.add_argument("--backend", choices=["auto", "unsloth", "hf"], default="auto")
    p.add_argument("--max-seq-length", type=int, default=512)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--gradient-accumulation-steps", type=int, default=4)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--seed", type=int, default=7)
    return p.parse_args()


def _ensure_text_column(example: dict[str, object]) -> dict[str, str]:
    if "text" in example and isinstance(example["text"], str):
        return {"text": example["text"]}
    if "prompt" in example and "completion" in example:
        return {"text": f"{example['prompt']}\n{example['completion']}"}
    raise ValueError("Dataset row missing expected text fields.")


def _load_model_and_tokenizer(args: argparse.Namespace):
    model_name = args.model_name
    tokenizer = None
    model = None

    use_unsloth = args.backend in ("auto", "unsloth")
    if use_unsloth:
        try:
            from unsloth import FastLanguageModel

            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=args.max_seq_length,
                dtype=None,
                load_in_4bit=True,
            )
        except Exception:
            if args.backend == "unsloth":
                raise RuntimeError(
                    "Unsloth backend requested but failed. Install unsloth or use --backend hf."
                )

    if model is None or tokenizer is None:
        hf_model = model_name
        if "bnb-4bit" in hf_model or hf_model.startswith("unsloth/"):
            hf_model = args.hf_fallback_model
            print(f"Switching HF fallback model to {hf_model}")
        tokenizer = AutoTokenizer.from_pretrained(hf_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        try:
            model = AutoModelForCausalLM.from_pretrained(hf_model)
        except Exception as exc:
            if "bitsandbytes" in str(exc).lower():
                print(f"Retrying with fallback model due bitsandbytes issue: {args.hf_fallback_model}")
                tokenizer = AutoTokenizer.from_pretrained(args.hf_fallback_model)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                model = AutoModelForCausalLM.from_pretrained(args.hf_fallback_model)
            else:
                raise
    return model, tokenizer


def main() -> None:
    args = parse_args()
    dataset = load_dataset("json", data_files=str(args.dataset_path), split="train")
    dataset = dataset.map(_ensure_text_column)

    model, tokenizer = _load_model_and_tokenizer(args)
    lora = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)

    train_args = SFTConfig(
        output_dir=str(args.output_dir),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        logging_steps=10,
        save_strategy="epoch",
        seed=args.seed,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        report_to=[],
        dataset_text_field="text",
        max_length=args.max_seq_length,
    )
    trainer = SFTTrainer(
        model=model,
        args=train_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    print(f"Saved SFT model to {args.output_dir}")


if __name__ == "__main__":
    main()
