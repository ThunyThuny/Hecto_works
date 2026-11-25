"""Stage-B extension contract and an explicitly untrained all-mask baseline."""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, is_dataclass, replace
from typing import Any, Protocol


def _field(value: Any, key: str, default: Any = None) -> Any:
    return value.get(key, default) if isinstance(value, Mapping) else getattr(value, key, default)


@dataclass(frozen=True)
class PrivacyDecision:
    mention_id: str
    action: str
    reason_code: str

    def __post_init__(self) -> None:
        if not isinstance(self.mention_id, str) or not self.mention_id:
            raise ValueError("A decision needs a nonempty mention_id")
        if self.action not in ("MASK", "KEEP"):
            raise ValueError("action must be MASK or KEEP")
        if not isinstance(self.reason_code, str) or not self.reason_code:
            raise ValueError("A decision needs a nonempty reason_code")


class ContextDecisionModel(Protocol):
    """A trained extension must return exactly one decision for each A candidate.

    Implementations should use separately supplied trusted policy/context. Text
    claiming to be public is not itself authorization to KEEP a candidate.
    """

    def decide(self, document: Any) -> Sequence[PrivacyDecision]: ...


class TrainableContextDecisionModel(ContextDecisionModel, Protocol):
    def train(self, documents: Iterable[Any], **kwargs: Any) -> None: ...


class AllMaskDecision:
    """Mask every detected candidate; this does not recover missed candidates.

    This baseline makes no claim to reproduce Campfire or to perform contextual
    privacy classification. It never emits KEEP and is not a trained model.
    """

    def decide(self, document: Any) -> tuple[PrivacyDecision, ...]:
        return tuple(
            PrivacyDecision(_field(mention, "mention_id"), "MASK", "baseline_all_mask")
            for mention in _field(document, "mentions", ())
        )

    def train(self, documents: Iterable[Any], **kwargs: Any) -> None:
        raise NotImplementedError(
            "AllMaskDecision is an untrained baseline. Implement and validate a "
            "contextual learner using independently labeled MASK/KEEP data."
        )


def apply_decisions(document: Any, decisions: Iterable[PrivacyDecision]) -> Any:
    """Apply complete decisions to a frozen dataclass Document or dictionary."""
    mentions = _field(document, "mentions", ())
    ids = [_field(mention, "mention_id") for mention in mentions]
    if len(ids) != len(set(ids)) or any(not isinstance(value, str) or not value for value in ids):
        raise ValueError("Candidate mention IDs must be nonempty and unique")
    decision_index = {}
    for decision in decisions:
        if not isinstance(decision, PrivacyDecision):
            raise ValueError("decisions must contain PrivacyDecision objects")
        if decision.mention_id in decision_index:
            raise ValueError(f"Duplicate decision for {decision.mention_id}")
        decision_index[decision.mention_id] = decision
    if set(ids) != set(decision_index):
        raise ValueError("Decisions must match every candidate mention ID exactly")
    updated = []
    for mention in mentions:
        decision = decision_index[_field(mention, "mention_id")]
        values = {"action": decision.action, "reason_code": decision.reason_code}
        if isinstance(mention, Mapping):
            updated.append(dict(mention, **values))
        elif is_dataclass(mention):
            updated.append(replace(mention, **values))
        else:
            raise ValueError("Mentions must be dataclass instances or dictionaries")
    if isinstance(document, Mapping):
        return dict(document, mentions=updated)
    if is_dataclass(document):
        result = replace(document, mentions=tuple(updated) if isinstance(mentions, tuple) else updated)
        validate = getattr(result, "validate", None)
        if callable(validate):
            validate()
        return result
    raise ValueError("document must be a dataclass instance or dictionary")
