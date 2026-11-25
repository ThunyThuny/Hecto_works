"""Label-free, fail-closed inputs for the first fixed-policy B classifier.

The candidate is bracketed by IDs of two added tokens, never by searching for
the candidate's value. Unknown metadata is excluded. The MVP requires the
*whole reviewed document* to fit: silently dropping a remote ownership sentence
would change the decision's meaning. Long documents must first be split into
independently reviewed context units or use a larger supported input length.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any


FIXED_POLICY_ID = "career_external_share_v1"
CAND_START, CAND_END = "[CAND_START]", "[CAND_END]"
MARKERS = (CAND_START, CAND_END)
LABEL2ID = {"MASK": 0, "KEEP": 1}
DEFAULT_INPUT_VERSION = "whole_document_v1"
FOCUSED_INPUT_VERSION = "target_focus_v2"
INPUT_VERSIONS = (DEFAULT_INPUT_VERSION, FOCUSED_INPUT_VERSION)

TYPE_ALIASES = {
    "NAME": "NAME", "PS_NAME": "NAME",
    "RESIDENT_NUMBER": "RESIDENT_NUMBER", "QT_RESIDENT_NUMBER": "RESIDENT_NUMBER",
    "PHONE": "PHONE", "QT_PHONE": "PHONE", "QT_MOBILE": "PHONE",
    "EMAIL": "EMAIL", "TMI_EMAIL": "EMAIL",
    "ADDRESS": "ADDRESS", "LC_ADDRESS": "ADDRESS",
    "WORKPLACE": "WORKPLACE", "OG_WORKPLACE": "WORKPLACE",
    "SCHOOL": "SCHOOL", "OGG_EDUCATION": "SCHOOL",
    "DEPARTMENT": "DEPARTMENT", "OG_DEPARTMENT": "DEPARTMENT",
    "POSITION": "POSITION", "CV_POSITION": "POSITION",
}
CONTEXTUAL_TYPES = frozenset({"PHONE", "EMAIL", "ADDRESS", "WORKPLACE", "SCHOOL", "DEPARTMENT", "POSITION"})
CONTACT_TYPES = frozenset({"PHONE", "EMAIL", "ADDRESS"})
DOCUMENT_TYPES = frozenset({"career_record", "resume", "cover_letter", "job_posting", "general_notice"})
TRUSTED_KEYS = frozenset({"approved_public_contacts", "approved_public_orgs", "approved_public_positions", "document_type"})
POLICY_CONFIG = {
    "policy_id": FIXED_POLICY_ID,
    "gate_version": "exact-approved-values-v1",
    "trusted_allowlists": {
        "PHONE": "approved_public_contacts", "EMAIL": "approved_public_contacts",
        "ADDRESS": "approved_public_contacts", "WORKPLACE": "approved_public_orgs",
        "SCHOOL": "approved_public_orgs", "DEPARTMENT": "approved_public_orgs",
        "POSITION": "approved_public_positions",
    },
    "type_aliases": dict(TYPE_ALIASES),
    "always_mask_canonical_types": ["NAME", "RESIDENT_NUMBER", "OTHER"],
    "unsupported_type_default": "MASK",
    "document_types": sorted(DOCUMENT_TYPES),
    "approval_match": "exact_source_value; membership_required_but_not_sufficient",
    "context_policy": "whole_reviewed_document_required",
}
POLICY_SPEC = POLICY_CONFIG


def policy_sha256() -> str:
    return hashlib.sha256(json.dumps(POLICY_SPEC, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode("utf-8")).hexdigest()


def input_spec(input_version: str = DEFAULT_INPUT_VERSION) -> dict[str, Any]:
    """Version model inputs separately from the unchanged masking policy.

    Existing checkpoints use v1. A focused checkpoint must be trained and
    calibrated again; changing only its scoring prefix is not compatible.
    """
    if input_version not in INPUT_VERSIONS:
        raise ValueError(f"Unsupported B input_version: {input_version!r}")
    return {
        "version": input_version,
        "whole_document_required": True,
        "candidate_markers": list(MARKERS),
        "target_binding": "source_start_end_offsets; exactly_one_marker_pair",
        "metadata": "allowlisted_trusted_boolean_features_and_document_type_only",
        "focus": ("none" if input_version == DEFAULT_INPUT_VERSION else
                  "target_value_and_containing_sentence_or_line_before_whole_document; offset_derived_v1"),
        "truncation": "reject_and_MASK; never_drop_whole_document_context",
    }


def input_sha256(input_version: str = DEFAULT_INPUT_VERSION) -> str:
    return hashlib.sha256(json.dumps(input_spec(input_version), sort_keys=True,
        ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def _target_focus(text: str, start: int, end: int) -> tuple[int, int]:
    """Locate a target's sentence/line without gold labels or value searches.

    Punctuation is a boundary only when followed by whitespace, so periods
    inside email addresses do not detach the target. A mention crossing a
    boundary remains intact. The entire source is also retained downstream.
    """
    boundaries = [match.span() for match in re.finditer(r"\n|(?<=[.!?。！？])[^\S\n]+", text)]
    focus_start = max((right for _, right in boundaries if right <= start), default=0)
    focus_end = min((left for left, _ in boundaries if left >= end), default=len(text))
    return focus_start, focus_end


def canonical_type(entity_type: str) -> str:
    return TYPE_ALIASES.get(entity_type, "OTHER")


def keep_eligible(entity_type: str) -> bool:
    """Only explicit policy-supported types can be released by B."""
    return canonical_type(entity_type) in CONTEXTUAL_TYPES


def trusted_features(document: Any, mention: Any) -> dict[str, Any]:
    """Convert an allowlist of externally supplied facts to fixed features.

    These fields are a provenance boundary: the caller/reviewer must supply
    them from a trusted source. Words inside ``document.text`` never set them.
    Unknown metadata (including gold actions/reasons) never reaches the model.
    """
    context = document.trusted_context
    if not isinstance(context, dict):
        raise ValueError("trusted_context must be a dictionary")
    flags: dict[str, Any] = {}
    for key, flag in (("approved_public_contacts", "approved_contact_match"),
                      ("approved_public_orgs", "approved_org_match"),
                      ("approved_public_positions", "approved_position_match")):
        values = context.get(key, [])
        if not isinstance(values, list) or any(not isinstance(value, str) or not value for value in values):
            raise ValueError(f"{key} must be a list of nonempty strings")
        flags[flag] = mention.text in values
    doc_type = context.get("document_type", "unknown")
    if not isinstance(doc_type, str):
        raise ValueError("document_type must be a string")
    flags["document_type"] = doc_type if doc_type in DOCUMENT_TYPES else "unknown"
    return flags


def allowed_to_keep(document: Any, mention: Any,
                    policy_id: str = FIXED_POLICY_ID) -> bool:
    """Hard policy gate independent of a classifier's probability."""
    if policy_id != FIXED_POLICY_ID or document.policy_id != policy_id:
        return False
    kind = canonical_type(mention.entity_type)
    if kind not in CONTEXTUAL_TYPES:
        return False
    facts = trusted_features(document, mention)
    if kind in CONTACT_TYPES:
        return facts["approved_contact_match"]
    if kind == "POSITION":
        return facts["approved_position_match"]
    return facts["approved_org_match"]


def _encoded(tokenizer: Any, text: str) -> list[int]:
    ids = tokenizer.encode(text, add_special_tokens=False, truncation=False)
    if not isinstance(ids, list) or any(type(value) is not int for value in ids):
        raise ValueError("tokenizer.encode must return integer token IDs")
    return ids


def prepare_candidate_input(tokenizer: Any, document: Any, mention: Any,
                            max_length: int = 512,
                            policy_id: str = FIXED_POLICY_ID,
                            input_version: str = DEFAULT_INPUT_VERSION) -> dict[str, Any]:
    """Return a model input or an explicit unavailable decision.

    Labels, review flags, reasons and arbitrary trusted metadata are never read.
    Sequence components are encoded separately to insert exactly one target
    marker pair even when the same candidate value appears repeatedly.
    """
    if type(max_length) is not int or max_length < 8:
        raise ValueError("max_length must be an integer >= 8")
    input_spec(input_version)
    if policy_id != FIXED_POLICY_ID:
        raise ValueError("B implements only career_external_share_v1")
    if document.policy_id != policy_id:
        raise ValueError("Document policy differs from the B checkpoint policy")
    mention.validate(document.text)
    base = {"available": False, "reason": "", "model_inputs": None,
            "context": {"start": 0, "end": len(document.text), "complete": True}}
    if not keep_eligible(mention.entity_type):
        return {**base, "reason": "policy_always_mask"}
    if any(marker in document.text for marker in MARKERS):
        return {**base, "reason": "reserved_marker_in_source"}
    facts = trusted_features(document, mention)
    prefix = (
        f"policy={policy_id};type={canonical_type(mention.entity_type)};"
        f"document_type={facts['document_type']};"
        f"approved_contact={int(facts['approved_contact_match'])};"
        f"approved_org={int(facts['approved_org_match'])};"
        f"approved_position={int(facts['approved_position_match'])}\n"
    )
    if input_version == FOCUSED_INPUT_VERSION:
        focus_start, focus_end = _target_focus(document.text, mention.start, mention.end)
        # This duplicates only source text. Gold labels, reason codes and
        # generator/reviewer metadata must never become explanatory prompts.
        prefix += (f"target_value={mention.text}\n"
                   f"target_focus={document.text[focus_start:focus_end]}\n"
                   "whole_document=\n")
        base["context"]["target_focus"] = {"start": focus_start, "end": focus_end}
    marker_ids = [tokenizer.convert_tokens_to_ids(marker) for marker in MARKERS]
    if (any(type(value) is not int for value in marker_ids)
            or len(set(marker_ids)) != 2
            or getattr(tokenizer, "unk_token_id", None) in marker_ids):
        raise ValueError("Tokenizer must contain two distinct registered candidate markers")
    before, candidate, after = (_encoded(tokenizer, document.text[:mention.start]),
                                _encoded(tokenizer, document.text[mention.start:mention.end]),
                                _encoded(tokenizer, document.text[mention.end:]))
    if not candidate:
        return {**base, "reason": "candidate_has_no_tokens"}
    raw_ids = _encoded(tokenizer, prefix) + before + [marker_ids[0]] + candidate + [marker_ids[1]] + after
    # prepare_for_model adds architecture-specific BOS/EOS but never truncates.
    inputs = dict(tokenizer.prepare_for_model(raw_ids, add_special_tokens=True,
                  truncation=False, return_attention_mask=True, return_token_type_ids=False))
    inputs = {key: value for key, value in inputs.items()
              if key in {"input_ids", "attention_mask", "token_type_ids"}}
    actual_ids = inputs["input_ids"]
    if any(actual_ids.count(value) != 1 for value in marker_ids):
        return {**base, "reason": "ambiguous_candidate_markers"}
    if len(actual_ids) > max_length:
        return {**base, "reason": "context_exceeds_max_length"}
    return {**base, "available": True, "reason": "scorable", "model_inputs": inputs}
