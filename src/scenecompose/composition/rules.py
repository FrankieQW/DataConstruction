from __future__ import annotations

import re

from .config import CompositionConfig
from .contracts import Classification, ObjectAsset
from .discovery import normalized_metadata


def _tokens(values: list[str]) -> set[str]:
    result: set[str] = set()
    for value in values:
        normalized = re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()
        result.add(normalized)
        result.update(normalized.split())
    return result


def classify_with_rules(asset: ObjectAsset, config: CompositionConfig) -> Classification:
    metadata = normalized_metadata(asset.annotation)
    license_name = str(metadata["license"])
    if license_name not in {value.casefold() for value in config.catalog.allowed_licenses}:
        return _rejected("license", f"License is not allowed: {license_name or 'missing'}")
    categories = {value.casefold() for value in metadata["categories"]}
    rejected_categories = {
        value.casefold() for value in config.catalog.reject_metadata_categories
    }
    blocked = sorted(categories & rejected_categories)
    if blocked:
        return _rejected("metadata_rule", f"Rejected metadata category: {blocked[0]}")
    lvis = [str(value).replace("_", " ") for value in metadata["lvis_categories"]]
    all_text = [
        *lvis, str(metadata["name"]), *metadata["tags"], *metadata["categories"]
    ]
    terms = _tokens(all_text)
    matches = []
    for rule in config.catalog.class_rules:
        matched_terms = [
            term for term in rule.terms
            if re.sub(r"[^a-z0-9]+", " ", term.casefold()).strip() in terms
        ]
        if matched_terms:
            lvis_match = any(
                re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()
                in {re.sub(r"[^a-z0-9]+", " ", item.casefold()).strip() for item in lvis}
                for value in rule.terms
            )
            matches.append((rule, matched_terms, lvis_match))
    if len(matches) == 1:
        rule, matched_terms, lvis_match = matches[0]
        return Classification(
            "accepted", rule.canonical_class, rule.placement_type,
            rule.support_classes, rule.target_dimension, rule.target_min_m,
            rule.target_max_m, 0.98 if lvis_match else 0.90, "metadata_rule",
            f"Matched {'LVIS' if lvis_match else 'metadata'} term: {matched_terms[0]}",
        )
    if len(matches) > 1:
        names = ", ".join(sorted(match[0].canonical_class for match in matches))
        return _review("metadata_rule", f"Conflicting canonical classes: {names}")
    return _review("metadata_rule", "No high-confidence canonical class rule matched")


def _rejected(source: str, reason: str) -> Classification:
    return Classification("rejected", None, None, (), None, None, None, 1.0, source, reason)


def _review(source: str, reason: str) -> Classification:
    return Classification("needs_review", None, None, (), None, None, None, 0.0, source, reason)

