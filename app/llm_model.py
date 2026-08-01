from functools import lru_cache
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)


MODEL_DIR = Path(__file__).resolve().parent / "models" / "gpt2_rl"

STOP_MARKERS = (
    "Done.",
    "Done!",
    "Done",
)


def get_device() -> torch.device:
    """Select the best available device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


@lru_cache(maxsize=1)
def load_llm():
    """
    Load and cache the RL-post-trained GPT-2 model and tokenizer.

    Caching prevents the model from being reloaded for every API request.
    """
    if not MODEL_DIR.exists():
        raise FileNotFoundError(
            f"RL-post-trained LLM checkpoint was not found at: {MODEL_DIR}"
        )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.config.pad_token_id = tokenizer.pad_token_id

    device = get_device()
    model.to(device)
    model.eval()

    return tokenizer, model, device


class StopOnFormatMarker(StoppingCriteria):
    """
    Stop generation when the RL formatting marker appears, but only after
    enough response tokens have been generated.
    """

    def __init__(
        self,
        tokenizer,
        prompt_length: int,
        stop_markers: tuple[str, ...],
        min_generated_tokens: int = 30,
    ):
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length
        self.stop_markers = stop_markers
        self.min_generated_tokens = min_generated_tokens

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
        **kwargs,
    ) -> bool:
        generated_ids = input_ids[0, self.prompt_length:]

        # Do not allow the learned marker to end an extremely short response.
        if generated_ids.shape[0] < self.min_generated_tokens:
            return False

        generated_text = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        ).rstrip()

        return any(
            generated_text.endswith(marker)
            for marker in self.stop_markers
        )


def remove_format_marker(text: str) -> str:
    """
    Remove the RL formatting markers from the user-facing response.
    """
    cleaned_text = text.strip()

    # Remove learned opening variants.
    opening_markers = (
        "Answer:",
        "Answer,",
        "Answer -",
        "Answer!",
        "Answer",
    )

    for marker in opening_markers:
        if cleaned_text.startswith(marker):
            cleaned_text = cleaned_text[len(marker):].lstrip()
            break

    # Remove learned closing variants.
    for marker in STOP_MARKERS:
        if cleaned_text.endswith(marker):
            cleaned_text = cleaned_text[:-len(marker)].rstrip()
            break

    return cleaned_text


def generate_with_llm(
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.7,
) -> str:
    """
    Generate an answer using the RL-post-trained GPT-2 model.

    Generation stops when the model produces its learned ending marker.
    The marker is removed before the response is returned to the user.
    """
    prompt = prompt.strip()

    if not prompt:
        raise ValueError("The prompt cannot be empty.")

    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be at least 1.")

    if temperature <= 0:
        raise ValueError("temperature must be greater than 0.")

    # The RL model learned to generate the Answer marker itself.
    formatted_prompt = f"Question: {prompt}\n"

    tokenizer, model, device = load_llm()

    encoded_prompt = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    encoded_prompt = {
        name: tensor.to(device)
        for name, tensor in encoded_prompt.items()
    }

    prompt_length = encoded_prompt["input_ids"].shape[1]

    stopping_criteria = StoppingCriteriaList(
        [
            StopOnFormatMarker(
                tokenizer=tokenizer,
                prompt_length=prompt_length,
                stop_markers=STOP_MARKERS,
                min_generated_tokens=30,
            )
        ]
    )

    with torch.no_grad():
        output_ids = model.generate(
            **encoded_prompt,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.1,
            stopping_criteria=stopping_criteria,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output_ids[0, prompt_length:]

    generated_answer = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    ).strip()

    return remove_format_marker(generated_answer)