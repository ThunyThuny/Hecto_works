"""Conservative NLST prognosis labels, without calendar-date inference.

Source definitions: NCI NLST Participant Dictionary (02/17/2026), sections 6,
14, 16 and 17; NLST User Guide, event conditions. Trial and extended mortality
data are never mixed. Source URLs are recorded in LABEL_SOURCES below.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DAYS_PER_MONTH = 365.25 / 12.0
HORIZON_YEARS = (1, 3, 5)
SUBTYPE_NAMES = {0: "adenocarcinoma", 1: "squamous_cell_carcinoma"}
# NCI-provided categories; never guessed ICD-O morphology code ranges.
HISTOLOGY_CAT_TO_SUBTYPE = {3: 0, 2: 1}
STAGE7_TO_CLASS = {110: 0, 120: 0, 210: 0, 220: 0, 310: 1, 320: 1, 400: 2}
DFS_UNAVAILABLE_REASON = "no disease-free-origin recurrence endpoint established"
LABEL_SOURCES = {
    "participant_dictionary": "https://cdas.cancer.gov/files/download/vi4qsrq5mc/participant.dictionary.d021326.pdf",
    "survival_user_guide": "https://cdas.cancer.gov/files/download/2f4hld9231/userguide.d032421_accessible.pdf",
}


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    # SAS .M/.N/.F codes and blanks stay unknown, never zero.
    result = pd.to_numeric(frame[column], errors="coerce").astype(float)
    return result.where(np.isfinite(result), np.nan)


def horizon_targets(time, event, validmask, horizons=(12.0, 36.0, 60.0)):
    """Return event-by-horizon targets and masks, using months throughout.

    Known event at/before h is positive, observation through h is negative,
    censoring before h is unknown. Masked target placeholders are zero and MUST
    be paired with the returned mask during loss and evaluation.
    """
    time = np.asarray(time, dtype=float).reshape(-1)
    event = np.asarray(event, dtype=float).reshape(-1)
    valid = np.asarray(validmask, dtype=float).reshape(-1)
    if not (time.shape == event.shape == valid.shape):
        raise ValueError("time, event, and validmask must have identical lengths")
    if not np.isfinite(valid).all() or not np.isin(valid, [0, 1]).all():
        raise ValueError("validmask must contain only 0 or 1")
    observed = valid.astype(bool)
    if not np.isfinite(time[observed]).all() or (time[observed] < 0).any():
        raise ValueError("Observed survival times must be finite and nonnegative")
    if not np.isfinite(event[observed]).all() or not np.isin(event[observed], [0, 1]).all():
        raise ValueError("Observed event indicators must contain only 0 or 1")
    horizons = np.asarray(horizons, dtype=float)
    if horizons.ndim != 1 or not len(horizons) or not np.isfinite(horizons).all() or (horizons <= 0).any():
        raise ValueError("horizons must be nonempty and positive")
    positives = (event[:, None] == 1) & (time[:, None] <= horizons[None, :])
    evaluable = observed[:, None] & (positives | (time[:, None] >= horizons[None, :]))
    return (positives & evaluable).astype(np.float32), evaluable.astype(np.float32)


def derive_participant_labels(
    frame: pd.DataFrame,
    stage_edition: str = "7",
    os_policy: str = "confirmed_death",
) -> pd.DataFrame:
    """Create independently masked labels keyed by unique participant pid.

    Cohort: conflc=1, finite nonnegative candx_days. Subtype: histology_cat 3/2
    maps to adeno/squamous 0/1; all other categories stay unsupported/masked.
    Stage: ONLY de_stag_7thed, without fallback to another AJCC edition.

    OS starts at first diagnosis. deathstat 1/2/3 with known death_days is an
    event; deathstat 0 with known fup_days is censored. Incomplete statuses
    4..7, missing values, or inconsistent chronology are masked. The raw trial
    follow-up is NOT adjusted to the published trial calendar cutoff; reporting
    must state this policy. Extended 2015 fields are ignored.

    DFS stays unavailable: progression does not establish disease-free origin
    or recurrence-only DFS. Unknown survival values remain NaN with mask=0;
    unknown classification values use -100 with mask=0. Placeholders never
    provide supervision without their matching mask.
    """
    if str(stage_edition) != "7":
        raise ValueError("Only explicit AJCC 7 labels supported; editions must not be mixed")
    if os_policy != "confirmed_death":
        raise ValueError("Unsupported os_policy; expected confirmed_death")
    if not frame.columns.is_unique:
        raise ValueError("Participant table has duplicate column names")
    if "pid" not in frame:
        raise ValueError("Participant table requires pid")
    identifiers = frame["pid"].astype("string").str.strip()
    if identifiers.isna().any() or identifiers.eq("").any():
        raise ValueError("Participant table has missing pid values")
    if identifiers.duplicated().any():
        raise ValueError("Participant table must contain exactly one row per pid")

    result = pd.DataFrame({"pid": identifiers}, index=frame.index)
    confirmed = _numeric(frame, "conflc").eq(1)
    diagnosis = _numeric(frame, "candx_days")
    origin_valid = diagnosis.notna() & diagnosis.ge(0)
    eligible = confirmed & origin_valid
    result["cohort_eligible"] = eligible.astype(np.int8)
    result["label_index_days"] = diagnosis.where(eligible)
    result["cohort_reason"] = np.where(
        ~confirmed, "not_confirmed_lung_cancer",
        np.where(~origin_valid, "missing_or_negative_diagnosis_offset", "eligible"),
    )

    subtype = _numeric(frame, "histology_cat").map(HISTOLOGY_CAT_TO_SUBTYPE)
    result["subtype_mask"] = (eligible & subtype.notna()).astype(np.int8)
    result["subtype"] = subtype.where(result["subtype_mask"].eq(1), -100).astype(np.int64)
    result["subtype_source"] = "histology_cat:3=adenocarcinoma(0);2=squamous(1);others=masked"

    stage = _numeric(frame, "de_stag_7thed").map(STAGE7_TO_CLASS)
    result["stage_mask"] = (eligible & stage.notna()).astype(np.int8)
    result["stage"] = stage.where(result["stage_mask"].eq(1), -100).astype(np.int64)
    result["stage_source"] = "de_stag_7thed:AJCC7;I-II=0;III=1;IV=2"

    death_status = _numeric(frame, "deathstat")
    death = _numeric(frame, "death_days")
    last_alive = _numeric(frame, "fup_days")
    confirmed_death = death_status.isin([1, 2, 3])
    alive = death_status.eq(0)
    contradictory_alive = alive & death.notna()
    if "contactstatus" in frame:
        contradictory_alive |= alive & _numeric(frame, "contactstatus").eq(2)
    endpoint = death.where(confirmed_death, last_alive.where(alive))
    valid_endpoint = endpoint.notna() & endpoint.ge(diagnosis)
    os_valid = eligible & (confirmed_death | alive) & ~contradictory_alive & valid_endpoint
    result["os_mask"] = os_valid.astype(np.int8)
    result["os_time"] = ((endpoint - diagnosis) / DAYS_PER_MONTH).where(os_valid)
    result["os_event"] = confirmed_death.astype(float).where(os_valid)
    result["os_reason"] = "valid"
    result.loc[~eligible, "os_reason"] = result.loc[~eligible, "cohort_reason"]
    result.loc[eligible & ~(confirmed_death | alive), "os_reason"] = "unknown_or_unconfirmed_death_status"
    result.loc[eligible & (confirmed_death | alive) & endpoint.isna(), "os_reason"] = "missing_endpoint_offset"
    result.loc[eligible & endpoint.notna() & ~valid_endpoint, "os_reason"] = "endpoint_before_diagnosis"
    result.loc[eligible & contradictory_alive, "os_reason"] = "alive_status_conflicts_with_death_record"
    result["os_policy"] = "confirmed_death:deathstat=1,2,3;alive=0;raw_trial_followup_no_cutoff_adjustment"
    result["os_origin"] = "first_confirmed_lung_cancer_diagnosis"
    result["time_unit"] = "months=days/30.4375"

    result["dfs_time"] = np.nan
    result["dfs_event"] = np.nan
    result["dfs_mask"] = np.int8(0)
    result["dfs_reason"] = DFS_UNAVAILABLE_REASON
    for endpoint_name in ("os", "dfs"):
        targets, masks = horizon_targets(
            result[f"{endpoint_name}_time"],
            result[f"{endpoint_name}_event"],
            result[f"{endpoint_name}_mask"],
        )
        for index, year in enumerate(HORIZON_YEARS):
            result[f"{endpoint_name}_y{year}"] = targets[:, index]
            result[f"{endpoint_name}_m{year}"] = masks[:, index]
    result.attrs.update({"label_sources": LABEL_SOURCES, "stage_edition": "7", "os_policy": os_policy})
    return result


