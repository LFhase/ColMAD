"""Batched chat completions over an OpenAI-compatible endpoint (OpenRouter by default).

Every model call in this repository goes through :func:`chat_batch`. It sends one single-turn
user message per prompt, retries failed requests with exponential back-off, and returns the
responses in the order they complete.
"""

from __future__ import annotations

import asyncio
import os
import random

#: Short names accepted on the command line -> the model names used in result file names.
MODEL_ALIASES = {
    "mini": "gpt-4o-mini-2024-07-18",
    "l70b": "Llama-3.1-70B",
    "r1": "deepseek-r1",
    "ms26": "mistral-small-2603",
    "qwen": "Qwen2.5-72B-Instruct-Turbo",
}

#: Model name -> OpenRouter model id. Any other name is sent to the endpoint unchanged.
OPENROUTER_IDS = {
    "gpt-4o-mini-2024-07-18": "openai/gpt-4o-mini-2024-07-18",
    "Llama-3.1-70B": "meta-llama/llama-3.1-70b-instruct",
    "deepseek-r1": "deepseek/deepseek-r1",
    "mistral-small-2603": "mistralai/mistral-small-2603",
    "Qwen2.5-72B-Instruct-Turbo": "qwen/qwen-2.5-72b-instruct",
}

#: Reasoning models get a larger completion budget, so that the answer is not cut off.
REASONING_MODELS = {"deepseek/deepseek-r1"}
MAX_TOKENS = 4096
REASONING_MAX_TOKENS = 40960

BASE_URL = os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1")
API_KEY_ENV = os.environ.get("LLM_API_KEY_ENV", "OPENROUTER_API_KEY")


def model_name(name):
    """Resolve a short name (``mini``) to the model name used in file names."""
    return MODEL_ALIASES.get(name, name)


def api_model_id(name):
    return OPENROUTER_IDS.get(model_name(name), model_name(name))


def strip_think(text):
    """Drop a ``<think>...</think>`` block that some reasoning models return inline."""
    if "<think>" in text and "</think>" in text:
        return text.split("<think>")[-1].split("</think>")[-1].strip()
    return text


async def _one(client, sem, model_id, prompt, temperature, max_tokens, seed, max_retries):
    data = dict(model=model_id, messages=[{"role": "user", "content": prompt["content"]}],
                temperature=temperature, max_tokens=max_tokens)
    if temperature > 0 and seed is not None and seed > 0:
        data["seed"] = seed
    async with sem:
        for attempt in range(max_retries):
            try:
                out = await client.chat.completions.create(**data)
                break
            except Exception as e:  # rate limits, timeouts, provider errors
                if attempt == max_retries - 1:
                    print(f"  [warn] idx={prompt['idx']}: giving up after {max_retries} attempts ({e})")
                    return {"idx": prompt["idx"], "response": ""}
                await asyncio.sleep(random.uniform(0, min(60, 2 ** attempt)))
    choices = getattr(out, "choices", None)
    if not choices or choices[0].message.content is None:
        print(f"  [warn] empty response for idx={prompt['idx']}; recording ''")
        return {"idx": prompt["idx"], "response": ""}
    if getattr(choices[0], "finish_reason", None) == "length":
        print(f"  [warn] truncated response (finish_reason=length) for idx={prompt['idx']}")
    return {"idx": prompt["idx"], "response": strip_think(choices[0].message.content)}


async def _batch(prompts, model, temperature, seed, concurrency, max_retries):
    from openai import AsyncOpenAI

    key = os.environ.get(API_KEY_ENV)
    if not key:
        raise RuntimeError(f"set {API_KEY_ENV} (see .env.example)")
    model_id = api_model_id(model)
    max_tokens = REASONING_MAX_TOKENS if model_id in REASONING_MODELS else MAX_TOKENS
    client = AsyncOpenAI(base_url=BASE_URL, api_key=key)
    sem = asyncio.Semaphore(concurrency)
    tasks = [asyncio.ensure_future(_one(client, sem, model_id, p, temperature, max_tokens, seed, max_retries))
             for p in prompts]
    out = []
    for fut in asyncio.as_completed(tasks):
        out.append(await fut)
    await client.close()
    return out


def chat_batch(prompts, model, temperature=0.0, seed=None, concurrency=10, max_retries=10):
    """Send ``prompts`` (a list of ``{"idx": int, "content": str}``) to ``model``.

    Returns ``[{"idx": int, "response": str}, ...]`` in completion order. A request that still
    fails after ``max_retries`` attempts is recorded as an empty string.
    """
    return asyncio.run(_batch(prompts, model, temperature, seed, concurrency, max_retries))
