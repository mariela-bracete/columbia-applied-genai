from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

MODEL_NAME = "openai-community/gpt2"
OUTPUT_DIR = Path("app/models/gpt2_qa")

# Keep the first run small enough to be practical on a local Mac.
NUM_EXAMPLES = 1_000
MAX_LENGTH = 256
TEST_SIZE = 0.1
SEED = 42


def select_best_response(answers: list[dict]) -> str:
    """
    Return the highest-ranked response from a Nectar example.

    Nectar normally stores responses in ranked order. The rank field is
    checked as a safeguard in case the order changes.
    """
    if not answers:
        return ""

    ranked_answers = sorted(
        answers,
        key=lambda answer: answer.get("rank", float("inf")),
    )

    best_answer = ranked_answers[0]

    return (
        best_answer.get("answer")
        or best_answer.get("response")
        or best_answer.get("text")
        or ""
    ).strip()


def clean_prompt(prompt: str) -> str:
    """
    Remove common conversation markers from Nectar prompts.
    """
    prompt = prompt.strip()

    if prompt.startswith("Human:"):
        prompt = prompt[len("Human:"):].strip()

    if prompt.endswith("Assistant:"):
        prompt = prompt[:-len("Assistant:")].strip()

    return prompt


def format_example(example: dict) -> dict:
    """
    Convert one Nectar prompt and its highest-ranked response into a
    question-and-answer causal-language-modeling example.
    """
    prompt = clean_prompt(str(example.get("prompt", "")))
    answer = select_best_response(example.get("answers", []))

    return {
        "text": (
            f"Question: {prompt}\n"
            f"Answer: {answer}"
        )
    }


def main() -> None:
    print(f"Loading {NUM_EXAMPLES} examples from Nectar...")

    dataset = load_dataset(
        "berkeley-nest/Nectar",
        split=f"train[:{NUM_EXAMPLES}]",
    )

    print("Original dataset columns:", dataset.column_names)
    print("First raw example:", dataset[0])

    dataset = dataset.map(
        format_example,
        remove_columns=dataset.column_names,
    )

    # Remove malformed or empty examples before tokenization.
    dataset = dataset.filter(
        lambda example: (
            example["text"].startswith("Question:")
            and "\nAnswer: " in example["text"]
            and len(example["text"].strip()) > len("Question:\nAnswer:")
        )
    )

    dataset = dataset.train_test_split(
        test_size=TEST_SIZE,
        seed=SEED,
    )

    print("Training examples:", len(dataset["train"]))
    print("Evaluation examples:", len(dataset["test"]))
    print("Formatted example:", dataset["train"][0]["text"])

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # GPT-2 does not define a padding token by default.
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    def tokenize_batch(batch: dict) -> dict:
        tokenized = tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_LENGTH,
        )

        # Add EOS so the model learns where each response should end.
        tokenized["input_ids"] = [
            input_ids + [tokenizer.eos_token_id]
            if input_ids[-1] != tokenizer.eos_token_id
            else input_ids
            for input_ids in tokenized["input_ids"]
        ]

        tokenized["attention_mask"] = [
            attention_mask + [1]
            if len(attention_mask) < len(input_ids)
            else attention_mask
            for attention_mask, input_ids in zip(
                tokenized["attention_mask"],
                tokenized["input_ids"],
            )
        ]

        return tokenized

    tokenized_dataset = dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=["text"],
    )

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.config.pad_token_id = tokenizer.pad_token_id

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    use_mps = torch.backends.mps.is_available()

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / "training_output"),
        num_train_epochs=1,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=25,
        save_total_limit=1,
        report_to="none",
        seed=SEED,
        dataloader_pin_memory=not use_mps,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        data_collator=data_collator,
    )

    print("Starting GPT-2 fine-tuning...")
    trainer.train()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    print(f"Saved fine-tuned model and tokenizer to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()