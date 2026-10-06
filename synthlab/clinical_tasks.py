"""Source-record clinical tasks with separate inputs and gold metadata.

Natural Synthea labels describe recorded codes, not clinically verified disease
absence or sufficiency of peripheral facts. Controlled clue-deletion cases have
separate, generator-defined evidence and abstention labels.
"""

from __future__ import annotations

import argparse
import calendar
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path

from .ehr_benchmark import CONDITIONS
from .synthea_benchmark import (
    CLINICAL_TABLES,
    SOURCE_SHA256,
    SOURCE_URL,
    histories_from_csv,
    read_csv_archive,
)

SCHEMA_VERSION = "clinical-tasks-v1"
RESOLUTIONS = ("1d", "1mo", "1y")
EXPLICITNESS = ("explicit", "implicit")
DEFAULT_INCIDENT_CUTOFFS = tuple(f"{year}-01-01" for year in range(2010, 2020))
# These are evaluation candidate groups, not mutually exclusive diagnoses.
DIFFERENTIAL_FAMILIES = {
    "respiratory": (
        "SNOMED:195967001",
        "SNOMED:13645005",
        "SNOMED:233604007",
        "SNOMED:84114007",
    ),
    "arthritis": ("SNOMED:69896004", "SNOMED:396275006"),
    "fatigue": ("SNOMED:40930008", "SNOMED:35489007", "SNOMED:78275009"),
}


def patient_split(patient_id, seed=42):
    """A patient never changes split when panels, tasks or resolutions change."""
    digest = hashlib.sha256(f"{seed}:{patient_id}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") / 2**64
    return "train" if bucket < 0.6 else "validation" if bucket < 0.8 else "test"


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _panel(conditions, concept_sets, extra_aliases):
    ids = [condition.concept_id for condition in conditions]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("candidate panel must contain unique conditions")
    missing = set(ids) - set(concept_sets)
    if missing:
        raise ValueError(f"missing verified concept sets: {sorted(missing)}")
    normalized = {
        key: {str(code).split(":")[-1] for code in concept_sets[key]} for key in ids
    }
    if any(
        not codes or key.split(":")[-1] not in codes
        for key, codes in normalized.items()
    ):
        raise ValueError("every descendant set must include its SNOMED root")
    aliases = {
        value.strip()
        for condition in conditions
        for value in (
            condition.name,
            *condition.aliases,
            *extra_aliases.get(condition.concept_id, []),
        )
        if value.strip()
    }
    pattern = re.compile(
        r"(?<!\w)(?:"
        + "|".join(re.escape(a) for a in sorted(aliases, key=lambda a: (-len(a), a)))
        + r")(?!\w)",
        re.IGNORECASE,
    )
    return ids, normalized, pattern


def _families(families, ids):
    result = {}
    for key, values in sorted(families.items()):
        selected = list(dict.fromkeys(values))
        if not set(selected) <= set(ids):
            raise ValueError(
                f"differential family {key} contains unvalidated panel candidates"
            )
        if len(selected) < 2:
            raise ValueError("differential families need at least two candidates")
        result[key] = selected
    return result


def _clean_identifier_text(text, patient, encounter_ids, candidate_ids):
    # Codes and identifiers are metadata. Numeric clinical measurements remain.
    identifiers = [
        patient,
        *encounter_ids,
        *candidate_ids,
        *(c.split(":")[-1] for c in candidate_ids),
    ]
    for identifier in identifiers:
        if identifier:
            text = re.sub(r"(?<!\w)" + re.escape(identifier) + r"(?!\w)", "", text)
    text = re.sub(
        r"\b[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\b",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return " ".join(text.split()).strip(" ;")


def _events(tables, ids, code_sets, alias_pattern, seed):
    histories = histories_from_csv(tables)
    demographics = {row["Id"]: row for row in tables["patients"]}
    if len(demographics) != len(tables["patients"]):
        raise ValueError("duplicate patient records")
    unknown = set(histories) - set(demographics)
    if unknown:
        raise ValueError("clinical entries reference unknown source patients")
    result = {}
    for patient in sorted(demographics):
        result[patient] = []
        for source in histories.get(patient, []):
            facts = [kind.capitalize() + ": " + fact for kind, fact in source["facts"]]
            clean = lambda value, patient=patient, source=source: (
                _clean_identifier_text(value, patient, source["encounter_ids"], ids)
            )
            implicit = clean(alias_pattern.sub("", "; ".join(facts)))
            explicit = clean(
                "; ".join(
                    [
                        *facts,
                        *(
                            "Recorded condition/indication: " + d
                            for d in sorted(set(source["diagnoses"]))
                            if d
                        ),
                    ]
                )
            )
            labels = [key for key in ids if source["codes"] & code_sets[key]]
            non_diagnosis = [
                clean(alias_pattern.sub("", fact))
                for kind, fact in source["facts"]
                if kind != "encounter"
            ]
            result[patient].append(
                {
                    "event_id": source["event_id"],
                    "patient_id": patient,
                    "split": patient_split(patient, seed),
                    "date": source["date"],
                    "explicit_text": explicit or "Clinical encounter recorded.",
                    "implicit_text": implicit or "Clinical encounter recorded.",
                    "recorded_codes": sorted(source["codes"]),
                    "recorded_labels": labels,
                    "gold_diagnosis_assertions": [
                        {"concept_id": key, "assertion": "affirmed_record"}
                        for key in labels
                    ],
                    "has_non_diagnosis_context": any(non_diagnosis),
                    "encounter_ids": source["encounter_ids"],
                }
            )
    return result, demographics


def _calendar_window(day, resolution):
    day = date.fromisoformat(day)
    if resolution == "1d":
        return day, day
    if resolution == "1mo":
        return date(day.year, day.month, 1), date(
            day.year, day.month, calendar.monthrange(day.year, day.month)[1]
        )
    if resolution == "1y":
        return date(day.year, 1, 1), date(day.year, 12, 31)
    raise ValueError("unsupported resolution")


def _record_labels(events, candidate_ids, version):
    evidence = {
        key: [e["event_id"] for e in events if key in e["recorded_labels"]]
        for key in candidate_ids
    }
    positives = [key for key in candidate_ids if evidence[key]]
    status = {}
    for key in candidate_ids:
        if key not in positives:
            status[key] = "no_recorded_target"
        elif version == "explicit":
            status[key] = "recorded_visible"
        elif any(
            e["has_non_diagnosis_context"]
            for e in events
            if key in e["recorded_labels"]
        ):
            status[key] = "recorded_hidden_with_context"
        else:
            status[key] = "recorded_hidden_no_context"
    return {
        "candidate_ids": candidate_ids,
        "gold_labels": positives,
        "eligible_ids": candidate_ids,
        "eligible_mask": [True] * len(candidate_ids),
        "evidence_event_ids": evidence,
        "information_status": status,
        "evidence_eligible_ids": positives if version == "explicit" else [],
        "first_recorded_dates": {
            key: next((e["date"] for e in events if key in e["recorded_labels"]), None)
            for key in candidate_ids
        },
        "negative_label_meaning": "no_matching_recorded_code_in_observed_scope_not_clinical_absence",
    }


def _add(
    documents, labels, base, events, candidate_ids, version, *, label_override=None
):
    # Full-history text is represented losslessly by dated windows. Avoid copying
    # the same ignored long string into every task/resolution/cutoff/horizon row.
    digest = hashlib.sha256()
    for index, event in enumerate(events):
        if index:
            digest.update(b"\n")
        digest.update((event["date"] + ": " + event[version + "_text"]).encode())
    referenced = bool(base.get("window_document_ids"))
    text = (
        "Clinical history is represented by the referenced dated windows."
        if referenced
        else "\n".join(e["date"] + ": " + e[version + "_text"] for e in events)
    )
    document = {
        **base,
        "explicitness": version,
        "text": text or "No clinical records available in this interval.",
        "event_ids": [e["event_id"] for e in events],
    }
    document["text_sha256"] = hashlib.sha256(document["text"].encode()).hexdigest()
    document["input_text_sha256"] = (
        digest.hexdigest() if events else document["text_sha256"]
    )
    if referenced:
        document["text_reference_policy"] = "all_referenced_dated_windows"
    gold = (
        _record_labels(events, candidate_ids, version)
        if label_override is None
        else label_override
    )
    labels.append(
        {
            **{
                key: document[key]
                for key in (
                    "document_id",
                    "patient_id",
                    "split",
                    "task",
                    "level",
                    "resolution",
                    "explicitness",
                )
            },
            **{
                key: document[key]
                for key in ("horizon_days", "family_id", "cutoff_date")
                if key in document
            },
            "label_scope": "window_recorded"
            if base["level"] == "event"
            else "full_history_recorded",
            **gold,
        }
    )
    documents.append(document)
    return document["document_id"]


def _history_documents(
    documents,
    labels,
    patient,
    events,
    candidate_ids,
    *,
    task,
    seed,
    suffix="",
    cutoff=None,
    extra=None,
    label_factory=None,
    window_registry=None,
    window_candidate_ids=None,
):
    extra = extra or {}
    split = patient_split(patient, seed)
    first = events[0]["date"] if events else None
    last = events[-1]["date"] if events else None
    for resolution in RESOLUTIONS:
        groups = defaultdict(list)
        for event in events:
            start, end = _calendar_window(event["date"], resolution)
            if cutoff:
                end = min(end, date.fromisoformat(cutoff))
            groups[(start.isoformat(), end.isoformat())].append(event)
        for version in EXPLICITNESS:
            windows = []
            for (start, end), entries in sorted(groups.items()):
                if window_registry is not None:
                    key = (patient, resolution, start, end, version)
                    if key not in window_registry:
                        base = {
                            "document_id": f"{patient}:history:{resolution}:{start}:{end}:{version}",
                            "patient_id": patient,
                            "split": split,
                            "task": "history",
                            "level": "event",
                            "resolution": resolution,
                            "window_start": start,
                            "window_end": end,
                            "cutoff_date": end,
                        }
                        window_registry[key] = _add(
                            documents,
                            labels,
                            base,
                            entries,
                            window_candidate_ids or candidate_ids,
                            version,
                        )
                    windows.append(window_registry[key])
                    continue
                base = {
                    "document_id": f"{patient}:{task}:{suffix}:{resolution}:{start}:{version}",
                    "patient_id": patient,
                    "split": split,
                    "task": task,
                    "level": "event",
                    "resolution": resolution,
                    "window_start": start,
                    "window_end": end,
                    "cutoff_date": cutoff or end,
                    **extra,
                }
                gold = label_factory(version) if label_factory else None
                windows.append(
                    _add(
                        documents,
                        labels,
                        base,
                        entries,
                        candidate_ids,
                        version,
                        label_override=gold,
                    )
                )
            base = {
                "document_id": f"{patient}:{task}:{suffix}:{resolution}:full-history:{version}",
                "patient_id": patient,
                "split": split,
                "task": task,
                "level": "patient",
                "resolution": resolution,
                "window_start": first,
                "window_end": cutoff or last,
                "cutoff_date": cutoff or last,
                "window_document_ids": windows,
                **extra,
            }
            gold = label_factory(version) if label_factory else None
            _add(
                documents,
                labels,
                base,
                events,
                candidate_ids,
                version,
                label_override=gold,
            )


def _incident_labels(
    all_events,
    past,
    candidate_ids,
    cutoff,
    horizon,
    followup_end,
    source_end,
    born,
    version,
):
    end = cutoff + timedelta(days=horizon)
    first_dates = {
        key: next((e["date"] for e in all_events if key in e["recorded_labels"]), None)
        for key in candidate_ids
    }
    evidence, positives, eligible, status = {}, [], [], {}
    at_risk = bool(past) and (not born or born <= cutoff) and followup_end >= cutoff
    for key in candidate_ids:
        first = date.fromisoformat(first_dates[key]) if first_dates[key] else None
        incident = first is not None and cutoff < first <= end
        evidence[key] = [
            e["event_id"]
            for e in all_events
            if incident
            and e["date"] == first.isoformat()
            and key in e["recorded_labels"]
        ]
        if not at_risk:
            status[key] = "not_at_risk"
        elif first and first <= cutoff:
            status[key] = "prevalent"
        elif incident and first <= followup_end:
            positives.append(key)
            eligible.append(key)
            status[key] = "future_recorded_target"
        elif followup_end < end:
            status[key] = "censored"
        else:
            eligible.append(key)
            status[key] = "no_recorded_target"
    return {
        "label_scope": "patient_risk_at_cutoff",
        "candidate_ids": candidate_ids,
        "gold_labels": positives,
        "eligible_ids": eligible,
        "eligible_mask": [key in eligible for key in candidate_ids],
        "evidence_event_ids": evidence,
        "evidence_eligible_ids": [],
        "information_status": status,
        "first_recorded_dates": first_dates,
        "prediction_end_date": end.isoformat(),
        "followup_end": followup_end.isoformat(),
        "source_end_date": source_end.isoformat(),
        "last_event_date": all_events[-1]["date"] if all_events else None,
        "negative_label_meaning": "no_first_recorded_target_in_complete_observed_horizon_not_clinical_absence",
    }


def _incident_documents(
    documents, labels, registry, patient, past, ids, cutoff, horizon, gold, seed
):
    """Share past-only event windows across cutoffs and prediction horizons."""
    first = past[0]["date"] if past else None
    split = patient_split(patient, seed)
    for resolution in RESOLUTIONS:
        groups = defaultdict(list)
        for event in past:
            start, end = _calendar_window(event["date"], resolution)
            groups[(start.isoformat(), min(end, cutoff).isoformat())].append(event)
        for version in EXPLICITNESS:
            windows = []
            for (start, end), entries in sorted(groups.items()):
                key = (patient, resolution, start, end, version)
                if key not in registry:
                    base = {
                        "document_id": f"{patient}:history:{resolution}:{start}:{end}:{version}",
                        "patient_id": patient,
                        "split": split,
                        "task": "history",
                        "level": "event",
                        "resolution": resolution,
                        "window_start": start,
                        "window_end": end,
                        "cutoff_date": end,
                    }
                    registry[key] = _add(documents, labels, base, entries, ids, version)
                windows.append(registry[key])
            base = {
                "document_id": f"{patient}:incident:{cutoff}:{horizon}:{resolution}:full-history:{version}",
                "patient_id": patient,
                "split": split,
                "task": "incident",
                "level": "patient",
                "resolution": resolution,
                "window_start": first,
                "window_end": cutoff.isoformat(),
                "cutoff_date": cutoff.isoformat(),
                "horizon_days": horizon,
                "window_document_ids": windows,
            }
            _add(documents, labels, base, past, ids, version, label_override=gold)


def _risk_summary_update(summary, patient, split, born, cutoff, horizon, gold):
    key = f"{cutoff}:{horizon}"
    age = (
        (
            (cutoff.year - born.year)
            - ((cutoff.month, cutoff.day) < (born.month, born.day))
        )
        if born
        else None
    )
    age_bin = (
        "unknown"
        if age is None
        else "not_born"
        if age < 0
        else "0-17"
        if age < 18
        else "18-39"
        if age < 40
        else "40-64"
        if age < 65
        else "65+"
    )
    entry = summary.setdefault(
        key,
        {
            "cutoff_date": cutoff.isoformat(),
            "horizon_days": horizon,
            "patients": 0,
            "age_counts": {},
            "conditions": {},
            "by_split": {},
        },
    )
    for current in (
        entry,
        entry["by_split"].setdefault(
            split, {"patients": 0, "age_counts": {}, "conditions": {}}
        ),
    ):
        current["patients"] += 1
        current["age_counts"][age_bin] = current["age_counts"].get(age_bin, 0) + 1
        for concept in gold["candidate_ids"]:
            values = current["conditions"].setdefault(
                concept,
                {
                    "eligible": 0,
                    "positive": 0,
                    "negative": 0,
                    "censored": 0,
                    "prevalent": 0,
                    "not_at_risk": 0,
                },
            )
            if concept in gold["eligible_ids"]:
                values["eligible"] += 1
                values[
                    "positive" if concept in gold["gold_labels"] else "negative"
                ] += 1
            else:
                values[gold["information_status"][concept]] += 1


def generate_clinical_tasks(
    tables,
    concept_sets,
    *,
    conditions=CONDITIONS,
    extra_aliases=None,
    differential_families=None,
    cutoff_dates=(),
    horizons=(90, 365),
    source_end_date=None,
    seed=42,
    tasks=("multilabel", "differential", "incident", "evidence"),
):
    """Generate natural-prevalence tasks without cohort sampling or label balancing.

    ``source_end_date`` is required for incident tasks with cutoffs. Follow-up ends
    at the earliest of last source event, death and export end (conservative).
    First diagnosis *or indication* defines recorded incidence; no claim is made
    about biological onset. Full histories at every resolution link all windows.
    """
    conditions = tuple(conditions)
    ids, code_sets, aliases = _panel(conditions, concept_sets, extra_aliases or {})
    if not set(tasks) <= {"multilabel", "differential", "incident", "evidence"}:
        raise ValueError("unknown clinical task")
    if differential_families is None:
        differential_families = {
            key: [c for c in values if c in ids]
            for key, values in DIFFERENTIAL_FAMILIES.items()
        }
        differential_families = {
            key: values
            for key, values in differential_families.items()
            if len(values) >= 2
        }
    families = _families(differential_families, ids)
    cutoffs = sorted({date.fromisoformat(value) for value in cutoff_dates})
    if cutoffs and (
        source_end_date is None
        or not horizons
        or any(h <= 0 or not isinstance(h, int) for h in horizons)
    ):
        raise ValueError(
            "incident cutoffs require source_end_date and positive integer horizons"
        )
    source_end = date.fromisoformat(source_end_date) if source_end_date else None
    if source_end and any(cutoff >= source_end for cutoff in cutoffs):
        raise ValueError("incident cutoffs must precede source export end")
    by_patient, demographics = _events(tables, ids, code_sets, aliases, seed)
    if source_end and any(
        e["date"] > source_end.isoformat()
        for events in by_patient.values()
        for e in events
    ):
        raise ValueError("source contains events after declared source_end_date")
    documents, labels, history_registry, risk_sets = [], [], {}, {}
    prevalence = {
        key: {
            "positive_patients": 0,
            "total_patients": len(by_patient),
            "by_split": {
                split: {"positive_patients": 0, "total_patients": 0}
                for split in ("train", "validation", "test")
            },
        }
        for key in ids
    }
    for patient, events in by_patient.items():
        split = patient_split(patient, seed)
        observed = {key for event in events for key in event["recorded_labels"]}
        for key in ids:
            prevalence[key]["positive_patients"] += key in observed
            prevalence[key]["by_split"][split]["positive_patients"] += key in observed
            prevalence[key]["by_split"][split]["total_patients"] += 1
        for task in ("multilabel", "evidence"):
            if task in tasks:
                _history_documents(
                    documents,
                    labels,
                    patient,
                    events,
                    ids,
                    task=task,
                    seed=seed,
                    window_registry=history_registry,
                    window_candidate_ids=ids,
                )
        if "differential" in tasks:
            for family, candidates in families.items():
                _history_documents(
                    documents,
                    labels,
                    patient,
                    events,
                    candidates,
                    task="differential",
                    seed=seed,
                    suffix=family,
                    extra={"family_id": family},
                    window_registry=history_registry,
                    window_candidate_ids=ids,
                )
        if "incident" in tasks:
            death_value = demographics[patient].get("DEATHDATE", "")
            birth_value = demographics[patient].get("BIRTHDATE", "")
            death = date.fromisoformat(death_value[:10]) if death_value else None
            born = date.fromisoformat(birth_value[:10]) if birth_value else None
            for cutoff in cutoffs:
                past = [e for e in events if e["date"] <= cutoff.isoformat()]
                followup_end = min(
                    [
                        source_end,
                        date.fromisoformat(events[-1]["date"]) if events else cutoff,
                        *([death] if death else []),
                    ]
                )
                for horizon in sorted(set(horizons)):
                    gold = _incident_labels(
                        events,
                        past,
                        ids,
                        cutoff,
                        horizon,
                        followup_end,
                        source_end,
                        born,
                        "implicit",
                    )
                    _risk_summary_update(
                        risk_sets, patient, split, born, cutoff, horizon, gold
                    )
                    _incident_documents(
                        documents,
                        labels,
                        history_registry,
                        patient,
                        past,
                        ids,
                        cutoff,
                        horizon,
                        gold,
                        seed,
                    )
    flat_events = [event for events in by_patient.values() for event in events]
    unique_inputs = {
        doc["text"] for doc in documents if not doc.get("window_document_ids")
    } | {event[version + "_text"] for event in flat_events for version in EXPLICITNESS}
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "corpus": "natural_synthea",
        "seed": seed,
        "candidate_ids": ids,
        "candidates": [
            {"concept_id": c.concept_id, "name": c.name} for c in conditions
        ],
        "source_url": SOURCE_URL,
        "reference_source_sha256": SOURCE_SHA256,
        "source_sha256": None,
        "source_archive_verified": False,
        "source_tables_sha256": _digest(tables),
        "mapping_sha256": _digest(
            {key: sorted(codes) for key, codes in code_sets.items()}
        ),
        "masking_aliases_sha256": hashlib.sha256(aliases.pattern.encode()).hexdigest(),
        "source_patients": len(by_patient),
        "patients_without_events": sum(not events for events in by_patient.values()),
        "source_events": len(flat_events),
        "included_clinical_tables": CLINICAL_TABLES,
        "split_policy": "sha256(seed:source_patient_id); 60/20/20; no stratification or resampling",
        "split_counts": dict(
            Counter(patient_split(patient, seed) for patient in by_patient)
        ),
        "prevalence": prevalence,
        "differential_families": families,
        "rare_conditions": [
            {
                "concept_id": key,
                "reason": "zero_recorded_positives"
                if value["positive_patients"] == 0
                else "some_split_has_zero_positives_or_zero_negatives",
            }
            for key, value in prevalence.items()
            if value["positive_patients"] == 0
            or any(
                count["positive_patients"] in (0, count["total_patients"])
                for count in value["by_split"].values()
            )
        ],
        "clinical_date_policy": "each_record_own_START_or_DATE; bundle_patient_day; inclusive_calendar_windows",
        "implicit_policy": "omit_all_diagnosis_and_indication_fields; remove_entire_panel_aliases; no_target_specific_text",
        "storage_policy": "canonical_full_panel_history_windows_shared_all_natural_tasks; nonempty_patient_text_is_reference_placeholder; input_text_sha256_hashes_exact_full_dated_history; source_events_and_patient_gold_unchanged",
        "label_policy": "source_recorded_SNOMED_root_or_verified_descendant_diagnosis_or_indication",
        "negative_policy": "no_recorded_code_in_scope; not_validated_disease_absence",
        "natural_evidence_policy": "attribution_to_recorded_source_event; explicit_only_gradable; peripheral_context_is_not_proof_of_sufficiency",
        "incident_policy": "fixed_calendar_cutoffs; text_at_or_before_cutoff; first_recorded_code_in_(cutoff,cutoff+horizon]; prevalent_excluded; positive_before_followup_end_or_complete_negative_followup_required",
        "followup_policy": "min(last_source_clinical_event,deathdate,source_export_end); death_before_horizon_censored_without_competing_risk_model",
        "source_end_date": source_end_date,
        "source_end_date_policy": "caller_declared_export_bound; reject_later_captured_clinical_events",
        "max_captured_clinical_date": max(
            (event["date"] for event in flat_events), default=None
        ),
        "cutoff_dates": [c.isoformat() for c in cutoffs],
        "horizons": sorted(set(horizons)),
        "risk_sets": risk_sets,
        "primary_metrics": [
            "micro_AUPRC",
            "macro_AUPRC_with_undefined_conditions_reported",
        ],
        "counts": {
            "documents": len(documents),
            "labels": len(labels),
            "events": len(flat_events),
            "unique_texts": len({d["text_sha256"] for d in documents}),
            "unique_input_texts": len(unique_inputs),
            "documents_by_task_level": dict(
                Counter(d["task"] + ":" + d["level"] for d in documents)
            ),
            "documents_by_task_resolution_level_explicitness": dict(
                Counter(
                    ":".join(
                        d[key]
                        for key in ("task", "resolution", "level", "explicitness")
                    )
                    for d in documents
                )
            ),
        },
    }
    return {
        "documents": documents,
        "labels": labels,
        "events": flat_events,
        "metadata": metadata,
    }


def generate_controlled_stress(
    *,
    conditions=CONDITIONS,
    differential_families=None,
    repetitions=12,
    seed=42,
    extra_aliases=None,
):
    """Paired clue deletion and family comorbidity stress cases, never natural data.

    A marked template finding is sufficient only by the generator's definition.
    Deleting every marked finding and recorded name makes different latent cases
    text-identical; abstention is then required by construction. These labels do
    not validate the diagnostic adequacy of any real clinical finding.
    """
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    conditions = tuple(conditions)
    ids, _, aliases = _panel(
        conditions,
        {c.concept_id: [c.concept_id.split(":")[-1]] for c in conditions},
        extra_aliases or {},
    )
    if differential_families is None:
        differential_families = {
            key: [c for c in values if c in ids]
            for key, values in DIFFERENTIAL_FAMILIES.items()
        }
        differential_families = {
            key: values
            for key, values in differential_families.items()
            if len(values) > 1
        }
    families = _families(differential_families, ids)
    lookup = {c.concept_id: c for c in conditions}
    documents, labels, events = [], [], []
    family_context = {
        "respiratory": "Reports shortness of breath and cough. Further characterization is unavailable.",
        "arthritis": "Reports joint pain and stiffness. Further characterization is unavailable.",
        "fatigue": "Reports fatigue and reduced daytime activity. Further characterization is unavailable.",
    }
    for family, candidates in families.items():
        combinations = [(key,) for key in candidates] + [tuple(candidates[:2])]
        for repeat in range(repetitions):
            for combination in combinations:
                patient = "stress-" + _digest([family, repeat, combination])[:18]
                split = patient_split(patient, seed)
                background = aliases.sub(
                    "",
                    family_context.get(
                        family,
                        "Reports ongoing symptoms; detailed characterization is unavailable.",
                    ),
                )
                day = (date(2018, 1, 10) + timedelta(days=repeat * 35)).isoformat()
                for variant in ("sufficient", "clue_deleted", "no_information"):
                    source_events = []
                    if variant != "no_information":
                        source_events.append(
                            {
                                "event_id": patient + ":context:" + variant,
                                "patient_id": patient,
                                "split": split,
                                "date": day,
                                "explicit_text": background,
                                "implicit_text": background,
                                "recorded_labels": [],
                                "has_non_diagnosis_context": True,
                                "gold_diagnosis_assertions": [],
                                "assertion": "context",
                            }
                        )
                    if variant == "sufficient":
                        for index, key in enumerate(combination):
                            finding = aliases.sub(
                                "", "; ".join(lookup[key].findings)
                            ).strip()
                            if not finding:
                                raise ValueError(
                                    "controlled finding disappeared under candidate alias masking"
                                )
                            event = {
                                "event_id": patient + f":clue:{index}",
                                "patient_id": patient,
                                "split": split,
                                "date": day,
                                "explicit_text": finding
                                + "; Recorded condition: "
                                + lookup[key].name,
                                "implicit_text": finding,
                                "recorded_labels": [key],
                                "has_non_diagnosis_context": True,
                                "gold_diagnosis_assertions": [
                                    {"concept_id": key, "assertion": "affirmed"}
                                ],
                                "assertion": "affirmed",
                                "template_evidence_role": "diagnostic_clue",
                            }
                            source_events.append(event)
                        # Same-family distractors require assertion handling.
                        for key in candidates:
                            if key not in combination:
                                finding = aliases.sub(
                                    "", "; ".join(lookup[key].findings)
                                ).strip()
                                source_events.append(
                                    {
                                        "event_id": patient + ":negated:" + key,
                                        "patient_id": patient,
                                        "split": split,
                                        "date": day,
                                        "explicit_text": "Current assessment does not demonstrate: "
                                        + finding,
                                        "implicit_text": "Current assessment does not demonstrate: "
                                        + finding,
                                        "recorded_labels": [],
                                        "has_non_diagnosis_context": True,
                                        "gold_diagnosis_assertions": [
                                            {"concept_id": key, "assertion": "negated"}
                                        ],
                                        "assertion": "negated",
                                        "template_evidence_role": "negated_distractor",
                                    }
                                )
                        for key in candidates:
                            finding = aliases.sub(
                                "", "; ".join(lookup[key].findings)
                            ).strip()
                            source_events.append(
                                {
                                    "event_id": patient + ":family-history:" + key,
                                    "patient_id": patient,
                                    "split": split,
                                    "date": day,
                                    "explicit_text": "Family history only; a sibling previously had: "
                                    + finding,
                                    "implicit_text": "Family history only; a sibling previously had: "
                                    + finding,
                                    "recorded_labels": [],
                                    "has_non_diagnosis_context": True,
                                    "gold_diagnosis_assertions": [
                                        {
                                            "concept_id": key,
                                            "assertion": "family_history",
                                        }
                                    ],
                                    "assertion": "family_history",
                                    "template_evidence_role": "family_history_distractor",
                                }
                            )
                    events.extend(source_events)

                    def factory(
                        version,
                        source_events=source_events,
                        variant=variant,
                        combination=combination,
                        candidates=candidates,
                    ):
                        sufficient = variant == "sufficient"
                        gold = list(combination) if sufficient else []
                        evidence = {
                            key: [
                                e["event_id"]
                                for e in source_events
                                if key in e["recorded_labels"]
                            ]
                            for key in candidates
                        }
                        return {
                            "candidate_ids": candidates,
                            "gold_labels": gold,
                            "latent_gold_labels": list(combination),
                            "eligible_ids": candidates,
                            "eligible_mask": [True] * len(candidates),
                            "evidence_event_ids": evidence,
                            "evidence_eligible_ids": gold,
                            "information_status": {
                                key: "sufficient_by_construction"
                                if sufficient
                                else "insufficient_by_construction"
                                for key in candidates
                            },
                            "gold_abstain": not sufficient,
                            "acceptable_differential_ids": list(combination)
                            if sufficient
                            else candidates,
                            "label_scope": "controlled_template_rule",
                            "stress_variant": variant,
                            "negative_label_meaning": "generator_defined_unasserted_candidate",
                        }

                    for task in ("differential", "multilabel", "evidence"):
                        _history_documents(
                            documents,
                            labels,
                            patient,
                            source_events,
                            candidates,
                            task=task,
                            seed=seed,
                            suffix=family + ":" + variant,
                            extra={"family_id": family, "stress_variant": variant},
                            label_factory=factory,
                        )
    return {
        "documents": documents,
        "labels": labels,
        "events": events,
        "metadata": {
            "schema_version": SCHEMA_VERSION,
            "corpus": "controlled_clue_deletion_stress",
            "candidate_ids": ids,
            "candidates": [
                {"concept_id": c.concept_id, "name": c.name} for c in conditions
            ],
            "differential_families": families,
            "repetitions": repetitions,
            "seed": seed,
            "split_policy": "same patient across clue deletion, explicitness, resolution and task; independent hash source namespace",
            "label_policy": "generator_defined_marked_finding_and_assertion; not_clinically_validated_evidence",
            "abstention_policy": "all_affirmed_template_evidence_and_names_deleted; shared_family_context_or_empty_input; gold_abstain_true",
            "prevalence_policy": "constructed_stress_cases_only; do_not_pool_with_natural_prevalence",
            "counts": {
                "documents": len(documents),
                "labels": len(labels),
                "events": len(events),
                "patients": len({d["patient_id"] for d in documents}),
                "unique_texts": len({d["text_sha256"] for d in documents}),
            },
        },
    }


def write_clinical_bundle(bundle, output_dir):
    """Write an immutable JSONL bundle; never overwrite an existing artifact."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    checksums = {}
    for name in ("documents", "labels", "events"):
        path = output / (name + ".jsonl")
        digest = hashlib.sha256()
        with path.open("xb") as handle:
            for row in bundle[name]:
                content = (json.dumps(row, sort_keys=True) + "\n").encode()
                handle.write(content)
                digest.update(content)
        checksums[path.name] = digest.hexdigest()
    path = output / "metadata.json"
    path.write_text(json.dumps(bundle["metadata"], sort_keys=True, indent=2) + "\n")
    checksums[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (output / "SHA256SUMS.json").write_text(
        json.dumps(checksums, sort_keys=True, indent=2) + "\n"
    )
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--mapping-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-end-date", required=True)
    parser.add_argument("--cutoff-date", action="append", default=[])
    parser.add_argument("--horizon-days", type=int, action="append")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--task",
        choices=("multilabel", "differential", "incident", "evidence"),
        action="append",
    )
    parser.add_argument("--controlled-output-dir", type=Path)
    parser.add_argument("--controlled-repetitions", type=int, default=12)
    args = parser.parse_args(argv)
    if hashlib.sha256(args.source_archive.read_bytes()).hexdigest() != SOURCE_SHA256:
        parser.error("source archive does not match pinned official April 2020 release")
    mapping = json.loads(args.mapping_json.read_text())
    available = set(mapping["concept_sets"])
    conditions = [
        condition for condition in CONDITIONS if condition.concept_id in available
    ]
    tables = read_csv_archive(args.source_archive)
    source_end = args.source_end_date
    if source_end == "auto":
        source_end = max(
            event["date"]
            for events in histories_from_csv(tables).values()
            for event in events
        )
    bundle = generate_clinical_tasks(
        tables,
        mapping["concept_sets"],
        conditions=conditions,
        extra_aliases=mapping.get("target_aliases", {}),
        cutoff_dates=args.cutoff_date or DEFAULT_INCIDENT_CUTOFFS,
        horizons=args.horizon_days or (90, 365),
        source_end_date=source_end,
        seed=args.seed,
        tasks=args.task or ("multilabel", "differential", "incident", "evidence"),
    )
    bundle["metadata"]["source_archive_verified"] = True
    bundle["metadata"]["source_sha256"] = SOURCE_SHA256
    if args.source_end_date == "auto":
        bundle["metadata"]["source_end_date_policy"] = (
            "max_captured_clinical_record_day_as_export_bound_proxy; not_continuous_observation"
        )
    bundle["metadata"]["mapping_file_sha256"] = hashlib.sha256(
        args.mapping_json.read_bytes()
    ).hexdigest()
    bundle["metadata"]["mapping_provenance"] = {
        key: value
        for key, value in mapping.items()
        if key not in ("concept_sets", "target_aliases")
    }
    write_clinical_bundle(bundle, args.output_dir)
    if args.controlled_output_dir:
        write_clinical_bundle(
            generate_controlled_stress(
                conditions=conditions,
                extra_aliases=mapping.get("target_aliases", {}),
                repetitions=args.controlled_repetitions,
                seed=args.seed,
            ),
            args.controlled_output_dir,
        )
    print(json.dumps(bundle["metadata"]["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
