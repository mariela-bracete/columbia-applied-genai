from pathlib import Path
import random
import re

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_MODEL_DIR = Path("app/models/gpt2_qa")
OUTPUT_DIR = Path("app/models/gpt2_rl")
EPOCH_CHECKPOINT_DIR = Path("app/models/gpt2_rl_checkpoints")
REWARD_PLOT_PATH = Path("rl_training_rewards.png")

REQUIRED_PREFIX = "Answer:"
REQUIRED_SUFFIX = "Done."

NUM_EXAMPLES = 128
WARMUP_EPOCHS = 8
RL_EPOCHS = 3

# Used only for the supervised format warm-up.
BATCH_SIZE = 4

# RL keeps a differentiable GPT-2 graph for each sampled response, so use
# one episode per update to avoid exhausting Apple Silicon shared memory.
RL_BATCH_SIZE = 1

MAX_SEQUENCE_LENGTH = 256
MAX_NEW_TOKENS = 48
MIN_NEW_TOKENS = 8

WARMUP_LEARNING_RATE = 5e-5
RL_LEARNING_RATE = 5e-6

TEMPERATURE = 0.8
TOP_K = 40
TOP_P = 0.9
REPETITION_PENALTY = 1.1

SEED = 42


# ---------------------------------------------------------------------------
# Reproducibility and device
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Nectar data preparation
# ---------------------------------------------------------------------------

def clean_prompt(prompt: str) -> str:
    """
    Remove the conversation wrappers commonly found in Nectar prompts.
    """
    prompt = prompt.strip()

    if prompt.startswith("Human:"):
        prompt = prompt[len("Human:"):].strip()

    if prompt.endswith("Assistant:"):
        prompt = prompt[:-len("Assistant:")].strip()

    return prompt


def select_best_response(answers: list[dict]) -> str:
    """
    Select the highest-ranked response from a Nectar example.
    """
    if not answers:
        return ""

    ranked_answers = sorted(
        answers,
        key=lambda answer: answer.get("rank", float("inf")),
    )

    return str(ranked_answers[0].get("answer", "")).strip()


def load_training_examples(num_examples: int) -> list[dict[str, str]]:
    """
    Load short, one-turn Nectar examples for warm-up and RL training.
    """
    dataset = load_dataset(
        "berkeley-nest/Nectar",
        split=f"train[:{num_examples * 8}]",
    )

    examples: list[dict[str, str]] = []

    for example in dataset:
        if example.get("turns") != 1:
            continue

        if not example.get("good_natured", True):
            continue

        question = clean_prompt(str(example.get("prompt", "")))
        answer = select_best_response(example.get("answers", []))

        # Avoid very long or malformed examples for local training.
        if not question or not answer:
            continue

        if len(question.split()) > 80:
            continue

        if len(answer.split()) > 100:
            continue

        examples.append(
            {
                "question": question,
                "answer": answer,
            }
        )

        if len(examples) >= num_examples:
            break

    if len(examples) < num_examples:
        raise RuntimeError(
            f"Only found {len(examples)} usable examples; "
            f"expected {num_examples}."
        )

    return examples


def format_prompt(question: str) -> str:
    return f"Question: {question}\n"


def format_target(answer: str) -> str:
    answer = " ".join(answer.strip().split())

    # Keep the response short enough that the closing marker is present in
    # nearly every warm-up target.
    answer_words = answer.split()[:40]
    shortened_answer = " ".join(answer_words)

    return (
        f"{REQUIRED_PREFIX} "
        f"{shortened_answer} "
        f"{REQUIRED_SUFFIX}"
    )

# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------

def freeze_lower_layers(model) -> None:
    """
    Train only the final two GPT-2 blocks, final layer norm, and LM head.
    """
    for parameter in model.parameters():
        parameter.requires_grad = False

    for block in model.transformer.h[-2:]:
        for parameter in block.parameters():
            parameter.requires_grad = True

    for parameter in model.transformer.ln_f.parameters():
        parameter.requires_grad = True

    for parameter in model.lm_head.parameters():
        parameter.requires_grad = True


def count_trainable_parameters(model) -> tuple[int, int]:
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    total = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    return trainable, total


# ---------------------------------------------------------------------------
# Short supervised format warm-up
# ---------------------------------------------------------------------------

def build_warmup_items(
    examples: list[dict[str, str]],
    tokenizer,
) -> list[dict[str, torch.Tensor]]:
    """
    Tokenize formatted question-answer demonstrations.

    Labels for prompt tokens are set to -100, so the loss is calculated only
    on the formatted answer.
    """
    items: list[dict[str, torch.Tensor]] = []

    for example in examples:
        prompt = format_prompt(example["question"])
        target = format_target(example["answer"])

        prompt_ids = tokenizer(
            prompt,
            add_special_tokens=False,
        )["input_ids"]

        target_ids = tokenizer(
            " " + target + tokenizer.eos_token,
            add_special_tokens=False,
        )["input_ids"]

        available_target_length = (
            MAX_SEQUENCE_LENGTH - len(prompt_ids)
        )

        if available_target_length <= 0:
            continue

        target_ids = target_ids[:available_target_length]

        input_ids = prompt_ids + target_ids
        labels = [-100] * len(prompt_ids) + target_ids
        attention_mask = [1] * len(input_ids)

        items.append(
            {
                "input_ids": torch.tensor(
                    input_ids,
                    dtype=torch.long,
                ),
                "attention_mask": torch.tensor(
                    attention_mask,
                    dtype=torch.long,
                ),
                "labels": torch.tensor(
                    labels,
                    dtype=torch.long,
                ),
            }
        )

    return items


def make_collate_function(tokenizer):
    """
    Pad a batch of causal language-modeling examples.
    """
    def collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        max_length = max(
            item["input_ids"].shape[0]
            for item in batch
        )

        padded_input_ids = []
        padded_attention_masks = []
        padded_labels = []

        for item in batch:
            padding_length = (
                max_length - item["input_ids"].shape[0]
            )

            padded_input_ids.append(
                F.pad(
                    item["input_ids"],
                    (0, padding_length),
                    value=tokenizer.pad_token_id,
                )
            )

            padded_attention_masks.append(
                F.pad(
                    item["attention_mask"],
                    (0, padding_length),
                    value=0,
                )
            )

            padded_labels.append(
                F.pad(
                    item["labels"],
                    (0, padding_length),
                    value=-100,
                )
            )

        return {
            "input_ids": torch.stack(padded_input_ids),
            "attention_mask": torch.stack(
                padded_attention_masks
            ),
            "labels": torch.stack(padded_labels),
        }

    return collate

def run_format_warmup(
    model,
    tokenizer,
    examples: list[dict[str, str]],
    device: torch.device,
) -> None:
    """
    Give the model a small number of supervised demonstrations of the format.

    This does not replace the RL stage. It gives the policy enough initial
    probability of producing the desired phrases for reward-based exploration
    to become possible.
    """
    items = build_warmup_items(
        examples=examples,
        tokenizer=tokenizer,
    )

    loader = DataLoader(
        items,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=make_collate_function(tokenizer),
    )

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=WARMUP_LEARNING_RATE,
    )

    print("\nStarting format warm-up...")

    model.train()

    for epoch in range(WARMUP_EPOCHS):
        losses: list[float] = []

        for batch in loader:
            batch = {
                name: tensor.to(device)
                for name, tensor in batch.items()
            }

            optimizer.zero_grad()

            outputs = model(**batch)
            loss = outputs.loss

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                trainable_parameters,
                max_norm=1.0,
            )

            optimizer.step()
            losses.append(loss.item())

        print(
            f"Warm-up epoch {epoch + 1:02d}/{WARMUP_EPOCHS} | "
            f"average loss: {np.mean(losses):.4f}"
        )

# ---------------------------------------------------------------------------
# Reward shaping
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def normalize_words(text: str) -> list[str]:
    """
    Convert text into lowercase alphanumeric words while ignoring punctuation.

    This treats forms such as "Answer:", "Answer,", and "Answer -" as the
    same marker, and likewise treats "Done." and "Done!" equally.
    """
    return re.findall(r"[a-z0-9]+", text.lower())


def matching_prefix_fraction(text: str, target: str) -> float:
    """
    Measure partial word-level agreement at the start while ignoring
    punctuation differences.
    """
    text_words = normalize_words(text)
    target_words = normalize_words(target)

    matches = 0

    for generated_word, target_word in zip(
        text_words,
        target_words,
    ):
        if generated_word != target_word:
            break

        matches += 1

    return matches / max(len(target_words), 1)


def matching_suffix_fraction(text: str, target: str) -> float:
    """
    Measure partial word-level agreement at the end while ignoring
    punctuation differences.
    """
    text_words = normalize_words(text)
    target_words = normalize_words(target)

    matches = 0

    for generated_word, target_word in zip(
        reversed(text_words),
        reversed(target_words),
    ):
        if generated_word != target_word:
            break

        matches += 1

    return matches / max(len(target_words), 1)


def format_reward(response: str) -> float:
    """
    Reward responses that begin with Answer and end with Done, while allowing
    harmless punctuation variants.
    """
    response_words = normalize_words(response)
    prefix_words = normalize_words(REQUIRED_PREFIX)
    suffix_words = normalize_words(REQUIRED_SUFFIX)

    prefix_fraction = matching_prefix_fraction(
        response,
        REQUIRED_PREFIX,
    )

    suffix_fraction = matching_suffix_fraction(
        response,
        REQUIRED_SUFFIX,
    )

    reward = 0.0

    # Partial phrase rewards.
    reward += 4.0 * prefix_fraction
    reward += 4.0 * suffix_fraction

    starts_correctly = (
        len(response_words) >= len(prefix_words)
        and response_words[:len(prefix_words)] == prefix_words
    )

    ends_correctly = (
        len(response_words) >= len(suffix_words)
        and response_words[-len(suffix_words):] == suffix_words
    )

    # Strong exact-position rewards.
    if starts_correctly:
        reward += 6.0
    else:
        reward -= 2.0

    if ends_correctly:
        reward += 6.0
    else:
        reward -= 2.0

    if len(response_words) < 8:
        reward -= 2.0

    if not response.strip():
        reward -= 5.0

    return reward


def follows_exact_format(response: str) -> bool:
    """
    Check the chosen Answer ... Done format while allowing punctuation variants.
    """
    response_words = normalize_words(response)
    prefix_words = normalize_words(REQUIRED_PREFIX)
    suffix_words = normalize_words(REQUIRED_SUFFIX)

    if len(response_words) < (
        len(prefix_words) + len(suffix_words)
    ):
        return False

    starts_correctly = (
        response_words[:len(prefix_words)] == prefix_words
    )

    ends_correctly = (
        response_words[-len(suffix_words):] == suffix_words
    )

    return starts_correctly and ends_correctly


# ---------------------------------------------------------------------------
# Sampling and differentiable log probabilities
# ---------------------------------------------------------------------------

def sample_response(
    model,
    tokenizer,
    question: str,
    device: torch.device,
) -> tuple[str, torch.Tensor, torch.Tensor]:
    """
    Sample an answer from the current policy.
    """
    prompt = format_prompt(question)

    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=384,
    )

    prompt_ids = encoded["input_ids"].to(device)
    prompt_attention_mask = encoded[
        "attention_mask"
    ].to(device)

    model.eval()

    with torch.no_grad():
        generated = model.generate(
            input_ids=prompt_ids,
            attention_mask=prompt_attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            min_new_tokens=MIN_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_k=TOP_K,
            top_p=TOP_P,
            repetition_penalty=REPETITION_PENALTY,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    prompt_length = prompt_ids.shape[1]
    generated_ids = generated[:, prompt_length:]

    response = tokenizer.decode(
        generated_ids[0],
        skip_special_tokens=True,
    ).strip()

    return response, prompt_ids, generated_ids


def calculate_action_log_probability(
    model,
    prompt_ids: torch.Tensor,
    generated_ids: torch.Tensor,
) -> torch.Tensor:
    """
    Calculate mean log probability of the generated token actions.
    """
    if generated_ids.numel() == 0:
        raise ValueError("The model generated no tokens.")

    full_sequence = torch.cat(
        [prompt_ids, generated_ids],
        dim=1,
    )

    attention_mask = torch.ones_like(
        full_sequence,
        dtype=torch.long,
    )

    model.train()

    outputs = model(
        input_ids=full_sequence[:, :-1],
        attention_mask=attention_mask[:, :-1],
    )

    logits = outputs.logits

    prompt_length = prompt_ids.shape[1]
    generated_length = generated_ids.shape[1]

    generated_logits = logits[
        :,
        prompt_length - 1:
        prompt_length - 1 + generated_length,
        :,
    ]

    log_probabilities = F.log_softmax(
        generated_logits,
        dim=-1,
    )

    selected_log_probabilities = log_probabilities.gather(
        dim=-1,
        index=generated_ids.unsqueeze(-1),
    ).squeeze(-1)

    return selected_log_probabilities.mean()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model,
    tokenizer,
    examples: list[dict[str, str]],
    device: torch.device,
    label: str,
    num_samples: int = 5,
) -> tuple[float, float]:
    """
    Report average reward and exact format success.
    """
    print(f"\n{label}")
    print("=" * 80)

    rewards: list[float] = []
    successes = 0

    for example in examples[:num_samples]:
        response, _, _ = sample_response(
            model=model,
            tokenizer=tokenizer,
            question=example["question"],
            device=device,
        )

        reward = format_reward(response)
        success = follows_exact_format(response)

        rewards.append(reward)
        successes += int(success)

        print(f"\nQuestion: {example['question']}")
        print(f"Response: {response}")
        print(f"Reward: {reward:.2f}")
        print(f"Follows exact format: {success}")

    average_reward = float(np.mean(rewards))
    success_rate = successes / len(rewards)

    print(
        f"\nAverage reward: {average_reward:.3f}"
    )
    print(
        f"Exact format success: {successes}/{len(rewards)} "
        f"({100 * success_rate:.1f}%)"
    )

    return average_reward, success_rate


# ---------------------------------------------------------------------------
# REINFORCE post-training
# ---------------------------------------------------------------------------

def run_rl_training(
    model,
    tokenizer,
    examples: list[dict[str, str]],
    device: torch.device,
) -> list[float]:
    """
    Post-train the model using a simplified REINFORCE update.
    """
    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=RL_LEARNING_RATE,
    )

    reward_history: list[float] = []
    baseline = 0.0
    baseline_initialized = False

    print("\nStarting reinforcement-learning post-training...")

    for epoch in range(RL_EPOCHS):
        random.shuffle(examples)

        epoch_rewards: list[float] = []
        epoch_losses: list[float] = []
        epoch_successes = 0

        for batch_start in range(
            0,
            len(examples),
            RL_BATCH_SIZE,
        ):
            batch_examples = examples[
                batch_start:batch_start + RL_BATCH_SIZE
            ]

            batch_log_probabilities: list[torch.Tensor] = []
            batch_rewards: list[float] = []

            last_response = ""

            for example in batch_examples:
                response, prompt_ids, generated_ids = (
                    sample_response(
                        model=model,
                        tokenizer=tokenizer,
                        question=example["question"],
                        device=device,
                    )
                )

                reward = format_reward(response)

                log_probability = (
                    calculate_action_log_probability(
                        model=model,
                        prompt_ids=prompt_ids,
                        generated_ids=generated_ids,
                    )
                )

                batch_log_probabilities.append(
                    log_probability
                )
                batch_rewards.append(reward)

                epoch_successes += int(
                    follows_exact_format(response)
                )

                last_response = response

            rewards_tensor = torch.tensor(
                batch_rewards,
                dtype=torch.float32,
                device=device,
            )

            batch_mean_reward = rewards_tensor.mean().item()

            if not baseline_initialized:
                baseline = batch_mean_reward
                baseline_initialized = True

            advantages = rewards_tensor - baseline

            # If a batch happens to have identical rewards, using the raw
            # rewards preserves a learning signal instead of producing zeros.
            if torch.allclose(
                advantages,
                torch.zeros_like(advantages),
            ):
                advantages = rewards_tensor

            log_probability_tensor = torch.stack(
                batch_log_probabilities
            )

            policy_loss = -(
                log_probability_tensor
                * advantages.detach()
            ).mean()

            optimizer.zero_grad(set_to_none=True)
            policy_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                trainable_parameters,
                max_norm=1.0,
            )

            optimizer.step()

            loss_value = policy_loss.item()

            # Release the token-level computation graph before the next
            # episode. This is important on Apple Silicon shared memory.
            del policy_loss
            del log_probability_tensor
            del rewards_tensor
            del advantages
            del batch_log_probabilities

            if device.type == "mps":
                torch.mps.empty_cache()

            baseline = (
                0.9 * baseline
                + 0.1 * batch_mean_reward
            )

            epoch_rewards.extend(batch_rewards)
            epoch_losses.append(loss_value)

        average_reward = float(
            np.mean(epoch_rewards)
        )

        average_loss = float(
            np.mean(epoch_losses)
        )

        total_episodes = len(examples)
        success_rate = (
            epoch_successes / total_episodes
        )

        reward_history.append(average_reward)

        print(
            f"RL epoch {epoch + 1:02d}/{RL_EPOCHS} | "
            f"average reward: {average_reward:.3f} | "
            f"average loss: {average_loss:.4f} | "
            f"format success: "
            f"{epoch_successes}/{total_episodes} "
            f"({100 * success_rate:.1f}%) | "
            f"baseline: {baseline:.3f}"
        )

        print("Sample response:")
        print(last_response)
        print("-" * 80)

        # Save after every completed RL epoch so progress is not lost if MPS
        # memory is exhausted later.
        epoch_output_dir = EPOCH_CHECKPOINT_DIR / f"epoch_{epoch + 1}"
        epoch_output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(epoch_output_dir)
        tokenizer.save_pretrained(epoch_output_dir)

        print(
            f"Saved epoch {epoch + 1} checkpoint to: "
            f"{epoch_output_dir.resolve()}"
        )

        if device.type == "mps":
            torch.mps.empty_cache()

    return reward_history


# ---------------------------------------------------------------------------
# Plot and main
# ---------------------------------------------------------------------------

def save_reward_plot(
    reward_history: list[float],
) -> None:
    plt.figure(figsize=(8, 4))

    plt.plot(
        range(1, len(reward_history) + 1),
        reward_history,
        marker="o",
    )

    plt.xlabel("RL epoch")
    plt.ylabel("Average reward")
    plt.title("RL Post-Training Reward")
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(REWARD_PLOT_PATH)
    plt.close()

    print(
        f"Saved reward plot to: "
        f"{REWARD_PLOT_PATH.resolve()}"
    )


def main() -> None:
    set_seed(SEED)

    if not BASE_MODEL_DIR.exists():
        raise FileNotFoundError(
            f"Base checkpoint not found at: "
            f"{BASE_MODEL_DIR.resolve()}"
        )

    device = get_device()

    print(f"Using device: {device}")
    print(
        f"Loading supervised checkpoint from: "
        f"{BASE_MODEL_DIR}"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_DIR
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_DIR
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.config.pad_token_id = (
        tokenizer.pad_token_id
    )

    model.to(device)

    examples = load_training_examples(
        NUM_EXAMPLES
    )

    print(
        f"Loaded {len(examples)} training examples."
    )
    print(f"Required prefix: {REQUIRED_PREFIX}")
    print(f"Required suffix: {REQUIRED_SUFFIX}")

    # Train all GPT-2 parameters during the short supervised format warm-up.
    for parameter in model.parameters():
        parameter.requires_grad = True

    warmup_trainable, total = count_trainable_parameters(model)

    print(
        f"Warm-up trainable parameters: {warmup_trainable:,} / "
        f"{total:,} "
        f"({100 * warmup_trainable / total:.2f}%)"
    )

    run_format_warmup(
        model=model,
        tokenizer=tokenizer,
        examples=examples,
        device=device,
    )

    # Freeze most layers again before the lower-learning-rate RL stage.
    freeze_lower_layers(model)

    rl_trainable, total = count_trainable_parameters(model)

    print(
        f"RL trainable parameters: {rl_trainable:,} / "
        f"{total:,} "
        f"({100 * rl_trainable / total:.2f}%)"
    )

    warmup_reward, warmup_success = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        examples=examples,
        device=device,
        label="Evaluation after format warm-up",
    )

    if warmup_success == 0:
        raise RuntimeError(
            "The format warm-up produced 0% success. "
            "Increase WARMUP_EPOCHS before running RL."
        )

    if device.type == "mps":
        torch.mps.empty_cache()

    reward_history = run_rl_training(
        model=model,
        tokenizer=tokenizer,
        examples=examples,
        device=device,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    save_reward_plot(reward_history)

    print(
        f"\nSaved RL-post-trained model and tokenizer to: "
        f"{OUTPUT_DIR.resolve()}"
    )

    evaluate_model(
        model=model,
        tokenizer=tokenizer,
        examples=examples,
        device=device,
        label="Final post-training evaluation",
    )


if __name__ == "__main__":
    main()