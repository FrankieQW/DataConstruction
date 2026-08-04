from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Iterable

from .config import CompositionConfig
from .contracts import Classification, ObjectAsset, write_json_atomic
from .discovery import normalized_metadata


def _extract_json(text: str) -> dict[str, object]:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("model output contains no JSON object")
    value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("model output is not a JSON object")
    return value


def classify_with_local_llm(
    assets: Iterable[ObjectAsset], config: CompositionConfig, cache_path: Path
) -> dict[str, Classification]:
    """Classify unresolved metadata with one process-local Transformers model."""
    if not config.llm.enabled:
        return {}
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError("LLM classification requires torch and transformers") from error

    config_digest = config.digest()
    cache = _read_cache(cache_path, config_digest)
    rules = {rule.canonical_class: rule for rule in config.catalog.class_rules}
    pending = [asset for asset in assets if asset.uid not in cache]
    dtype = getattr(torch, config.llm.dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        config.llm.model_path,
        trust_remote_code=config.llm.trust_remote_code,
        local_files_only=config.llm.local_files_only,
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.llm.model_path,
        torch_dtype=dtype,
        trust_remote_code=config.llm.trust_remote_code,
        local_files_only=config.llm.local_files_only,
    ).to(config.llm.device).eval()
    allowed = ", ".join(sorted(rules))
    for start in range(0, len(pending), config.llm.batch_size):
        batch = pending[start:start + config.llm.batch_size]
        prompts = []
        for asset in batch:
            metadata = normalized_metadata(asset.annotation)
            prompts.append(
                "Classify one 3D asset for physical scene placement. "
                f"Allowed classes: {allowed}. Return JSON only with keys "
                'decision (accepted or rejected), canonical_class, confidence, reason. '
                f"Metadata: {json.dumps(metadata, ensure_ascii=False)} /no_think"
            )
        texts = []
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            try:
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            texts.append(text)
        encoded = tokenizer(texts, return_tensors="pt", padding=True).to(config.llm.device)
        generation = model.generate(
            **encoded,
            max_new_tokens=config.llm.max_new_tokens,
            do_sample=config.llm.temperature > 0,
            temperature=max(config.llm.temperature, 1e-5),
        )
        input_length = int(encoded["input_ids"].shape[1])
        for asset, tokens in zip(batch, generation):
            raw = tokenizer.decode(tokens[input_length:], skip_special_tokens=True)
            classification = _validate(raw, rules, config)
            cache[asset.uid] = classification
            write_json_atomic(cache_path, {
                "schema_version": 1,
                "config_digest": config_digest,
                "records": {uid: value.to_dict() for uid, value in sorted(cache.items())},
            })
    return cache


def _validate(raw: str, rules: dict[str, object], config: CompositionConfig) -> Classification:
    try:
        value = _extract_json(raw)
        name = str(value.get("canonical_class") or "")
        confidence = float(value.get("confidence", 0.0))
        reason = str(value.get("reason") or "Local LLM classification")[:500]
        if value.get("decision") != "accepted" or name not in rules:
            return Classification("rejected", None, None, (), None, None, None,
                                  confidence, "local_llm", reason)
        if confidence < config.llm.acceptance_threshold:
            return Classification("needs_review", None, None, (), None, None, None,
                                  confidence, "local_llm", "Below acceptance threshold")
        rule = rules[name]
        return Classification(
            "accepted", name, rule.placement_type, rule.support_classes,
            rule.target_dimension, rule.target_min_m, rule.target_max_m,
            confidence, "local_llm", reason,
        )
    except (ValueError, TypeError, json.JSONDecodeError) as error:
        return Classification("needs_review", None, None, (), None, None, None,
                              0.0, "local_llm", f"Invalid model output: {error}")


def _read_cache(path: Path, config_digest: str) -> dict[str, Classification]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or payload.get("config_digest") != config_digest:
        return {}
    raw = payload.get("records", {})
    result = {}
    for uid, value in raw.items():
        value = dict(value)
        value["support_classes"] = tuple(value.get("support_classes", ()))
        result[str(uid)] = Classification(**value)
    return result
