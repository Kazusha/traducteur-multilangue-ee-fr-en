from __future__ import annotations

import argparse
from pathlib import Path

from datasets import Dataset
import pandas as pd
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

LANG_CODE_MAP = {
    "fr": "fra_Latn",
    "en": "eng_Latn",
    "ewe": "ewe_Latn",
}


def tokenize_fn(batch, tokenizer, src_col: str, tgt_col: str, src_lang: str, tgt_lang: str):
    tokenizer.src_lang = LANG_CODE_MAP[src_lang]
    model_inputs = tokenizer(batch[src_col], max_length=256, truncation=True)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(batch[tgt_col], max_length=256, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tuning NLLB pour une paire de langues")
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--eval-csv", required=True)
    parser.add_argument("--src-col", required=True)
    parser.add_argument("--tgt-col", required=True)
    parser.add_argument("--src-lang", choices=["fr", "en", "ewe"], required=True)
    parser.add_argument("--tgt-lang", choices=["fr", "en", "ewe"], required=True)
    parser.add_argument("--model-name", default="facebook/nllb-200-distilled-600M")
    parser.add_argument("--output-dir", default="outputs/nllb-finetuned")
    args = parser.parse_args()

    train_df = pd.read_csv(Path(args.train_csv))
    eval_df = pd.read_csv(Path(args.eval_csv))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    train_ds = Dataset.from_pandas(train_df)
    eval_ds = Dataset.from_pandas(eval_df)

    train_ds = train_ds.map(
        tokenize_fn,
        batched=True,
        fn_kwargs={
            "tokenizer": tokenizer,
            "src_col": args.src_col,
            "tgt_col": args.tgt_col,
            "src_lang": args.src_lang,
            "tgt_lang": args.tgt_lang,
        },
    )
    eval_ds = eval_ds.map(
        tokenize_fn,
        batched=True,
        fn_kwargs={
            "tokenizer": tokenizer,
            "src_col": args.src_col,
            "tgt_col": args.tgt_col,
            "src_lang": args.src_lang,
            "tgt_lang": args.tgt_lang,
        },
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        learning_rate=2e-5,
        num_train_epochs=3,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        predict_with_generate=True,
        fp16=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
