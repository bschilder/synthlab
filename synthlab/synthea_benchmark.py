"""Paired diagnosis-visible/hidden benchmarks from complete Synthea CSV histories.

Ground truth is recorded diagnosis/indication membership in verified SNOMED
descendant sets. It is never inferred from text or filled from disease labels.
All source events are retained, including events older than the recent windows.
"""

from __future__ import annotations

import calendar
import csv
import io
import random
import re
import zipfile
from collections import defaultdict
from datetime import date, timedelta

from .ehr_benchmark import CONDITIONS

SOURCE_URL = "https://synthetichealth.github.io/synthea-sample-data/downloads/synthea_sample_data_csv_apr2020.zip"
SOURCE_SHA256 = "4194b18c11eaedcf0d5d5dd448d8ac9661f14381e2ef9f109215dc42266cd38a"


def read_csv_archive(path):
    with zipfile.ZipFile(path) as archive:
        return {
            name: list(
                csv.DictReader(io.TextIOWrapper(archive.open("csv/" + name + ".csv")))
            )
            for name in [
                "patients",
                "encounters",
                "conditions",
                "observations",
                "medications",
                "procedures",
                "careplans",
            ]
        }


def _hidden(text, pattern):
    # Empty replacements prevent a diagnosis-redaction marker from becoming a label cue.
    return " ".join(pattern.sub("", text).split())


def histories_from_csv(tables):
    """Bundle dated encounters and preserve orphan clinical entries as dated events."""
    histories, encounters = defaultdict(dict), {}
    for row in tables["encounters"]:
        patient, identifier = row["PATIENT"], row["Id"]
        event = {
            "event_id": identifier,
            "date": row["START"][:10],
            "facts": [],
            "diagnoses": [],
            "codes": set(),
        }
        event["facts"].append(("encounter", row["DESCRIPTION"]))
        if row.get("REASONCODE"):
            event["codes"].add(row["REASONCODE"])
            event["diagnoses"].append(row.get("REASONDESCRIPTION", ""))
        histories[patient][identifier] = event
        encounters[(patient, identifier)] = event
    for table in [
        "conditions",
        "observations",
        "medications",
        "procedures",
        "careplans",
    ]:
        for index, row in enumerate(tables[table]):
            patient, identifier = row["PATIENT"], row.get("ENCOUNTER", "")
            event = encounters.get((patient, identifier))
            if event is None:
                identifier = f"{table}-{index}"
                event = {
                    "event_id": identifier,
                    "date": row.get("DATE", row.get("START"))[:10],
                    "facts": [],
                    "diagnoses": [],
                    "codes": set(),
                }
                histories[patient][identifier] = event
            if table == "conditions":
                event["codes"].add(row["CODE"])
                event["diagnoses"].append(row["DESCRIPTION"])
            else:
                fact = row["DESCRIPTION"]
                if table == "observations":
                    fact += ": " + row.get("VALUE", "") + " " + row.get("UNITS", "")
                event["facts"].append((table, fact.strip()))
                if row.get("REASONCODE"):
                    event["codes"].add(row["REASONCODE"])
                    event["diagnoses"].append(row.get("REASONDESCRIPTION", ""))
    return {
        patient: sorted(events.values(), key=lambda e: (e["date"], e["event_id"]))
        for patient, events in histories.items()
    }


def _window_key(day, resolution):
    if resolution == "1d":
        return day.isoformat(), day - timedelta(days=1), day
    if resolution == "1mo":
        return (
            f"{day.year:04d}-{day.month:02d}",
            date(day.year, day.month, 1) - timedelta(days=1),
            date(day.year, day.month, calendar.monthrange(day.year, day.month)[1]),
        )
    return (
        str(day.year),
        date(day.year, 1, 1) - timedelta(days=1),
        date(day.year, 12, 31),
    )


def generate_synthea_benchmark(
    tables,
    concept_sets,
    *,
    extra_aliases=None,
    patients_per_condition=120,
    minimum_cases=20,
    seed=42,
    conditions=CONDITIONS,
):
    """Balanced case/control cohorts from a shared, globally split source population.

    Cohorts may share a source patient; patient_id and split remain identical across
    every target. Conditions with fewer than minimum_cases are explicitly excluded.
    Implicit text removes diagnosis/indication fields and every supplied target alias;
    medication, procedure and observation facts remain. This is a Synthea simulation
    benchmark, not a clinical validation of absence of disease in real records.
    """
    if patients_per_condition < 12 or patients_per_condition % 2 or minimum_cases < 3:
        raise ValueError("even cohort size >=12 and minimum_cases >=3 required")
    histories = histories_from_csv(tables)
    demographics = {p["Id"]: p for p in tables["patients"]}
    source_patients = sorted(set(histories) & set(demographics))
    positives, eligible, excluded = {}, [], []
    for condition in conditions:
        codes = set(concept_sets[condition.concept_id])
        positives[condition.concept_id] = {
            p for p in source_patients if any(e["codes"] & codes for e in histories[p])
        }
        count = len(positives[condition.concept_id])
        (
            eligible
            if count >= minimum_cases and len(source_patients) - count >= minimum_cases
            else excluded
        ).append(condition)
    if not eligible:
        raise ValueError("no conditions have sufficient recorded cases and controls")
    # Fixed source-patient split precedes target-specific cohort selection.
    assignment = None
    for attempt in range(1000):
        shuffled = source_patients.copy()
        random.Random(seed + attempt).shuffle(shuffled)
        trial = {
            p: "train"
            if i < int(0.6 * len(shuffled))
            else "validation"
            if i < int(0.8 * len(shuffled))
            else "test"
            for i, p in enumerate(shuffled)
        }
        if all(
            sum(trial[p] == split for p in positives[c.concept_id]) >= 3
            and sum(
                trial[p] == split
                for p in source_patients
                if p not in positives[c.concept_id]
            )
            >= 3
            for c in eligible
            for split in ["train", "validation", "test"]
        ):
            assignment = trial
            break
    if assignment is None:
        raise ValueError("could not create patient-disjoint splits with both classes")
    extra_aliases = extra_aliases or {}
    aliases = [
        a
        for c in conditions
        for a in (c.name, *c.aliases, *extra_aliases.get(c.concept_id, []))
    ]
    alias_pattern = re.compile(
        r"(?<!\w)(?:"
        + "|".join(
            re.escape(a.strip())
            for a in sorted(set(aliases), key=len, reverse=True)
            if a.strip()
        )
        + r")(?!\w)",
        flags=re.IGNORECASE,
    )
    windows, patient_rows, atomic, counts = [], [], [], {}
    rng = random.Random(seed)
    for condition in eligible:
        codes = set(concept_sets[condition.concept_id])
        case_set = positives[condition.concept_id]
        selected = []
        for split, fraction in [("train", 0.6), ("validation", 0.2), ("test", 0.2)]:
            cases = sorted(p for p in case_set if assignment[p] == split)
            controls = sorted(
                p
                for p in source_patients
                if p not in case_set and assignment[p] == split
            )
            rng.shuffle(cases)
            rng.shuffle(controls)
            count = min(
                len(cases),
                len(controls),
                max(3, int(patients_per_condition / 2 * fraction)),
            )
            selected.extend(cases[:count] + controls[:count])
        counts[condition.concept_id] = {
            "available_positive_patients": len(case_set),
            "selected_patients": len(selected),
            "splits": {
                s: sum(assignment[p] == s for p in selected)
                for s in ["train", "validation", "test"]
            },
        }
        for patient in selected:
            positive = patient in case_set
            stratum = "case" if positive else "control"
            cohort = condition.concept_id + "-" + patient
            variants = {"explicit": [], "implicit": []}
            for event in histories[patient]:
                supports = bool(event["codes"] & codes)
                facts = [
                    kind.capitalize() + ": " + fact for kind, fact in event["facts"]
                ]
                explicit = "; ".join(
                    [
                        *facts,
                        *[
                            "Recorded condition/indication: " + d
                            for d in sorted(set(event["diagnoses"]))
                            if d
                        ],
                    ]
                )
                implicit = (
                    _hidden("; ".join(facts), alias_pattern)
                    or "Clinical encounter recorded."
                )
                if alias_pattern.search(implicit):
                    raise ValueError("target alias leaked into diagnosis-hidden text")
                for version, text in [("explicit", explicit), ("implicit", implicit)]:
                    entry = {
                        "event_id": cohort + "-" + event["event_id"] + "-" + version,
                        "date": event["date"],
                        "positive": supports,
                        "text": text,
                        "has_peripheral_facts": any(
                            kind != "encounter" for kind, _ in event["facts"]
                        ),
                    }
                    variants[version].append(entry)
                    atomic.append(
                        {
                            **entry,
                            "patient_id": patient,
                            "concept_id": condition.concept_id,
                            "condition": condition.name,
                            "explicitness": version,
                            "split": assignment[patient],
                            "patient_ever_positive": positive,
                        }
                    )
            for resolution in ["1d", "1mo", "1y"]:
                for version, events in variants.items():
                    groups = defaultdict(list)
                    boundaries = {}
                    for event in events:
                        key, start, end = _window_key(
                            date.fromisoformat(event["date"]), resolution
                        )
                        groups[key].append(event)
                        boundaries[key] = (start, end)
                    window_ids, covered = [], []
                    base = {
                        "patient_id": patient,
                        "condition": condition.name,
                        "concept_id": condition.concept_id,
                        "patient_ever_positive": positive,
                        "patient_stratum": stratum,
                        "split": assignment[patient],
                        "resolution": resolution,
                        "explicitness": version,
                        "target_aliases": [condition.name, *condition.aliases],
                        "generator": "synthea-csv-apr2020-v1",
                    }
                    for key, entries in sorted(groups.items()):
                        start, end = boundaries[key]
                        identifier = (
                            cohort + "-" + resolution + "-" + key + "-" + version
                        )
                        window_ids.append(identifier)
                        covered.extend(e["event_id"] for e in entries)
                        supported = any(e["positive"] for e in entries)
                        text = "\n".join(e["date"] + ": " + e["text"] for e in entries)
                        windows.append(
                            {
                                **base,
                                "document_id": identifier,
                                "level": "event",
                                "positive": supported,
                                "negative_type": "case"
                                if supported
                                else "history_outside_window"
                                if positive
                                else "synthea_other_conditions",
                                "window_start": start.isoformat(),
                                "window_end": end.isoformat(),
                                "event_ids": [e["event_id"] for e in entries],
                                "n_events": len(entries),
                                "text": text,
                                "has_peripheral_facts": any(
                                    e["has_peripheral_facts"] for e in entries
                                ),
                            }
                        )
                    if len(covered) != len(events) or set(covered) != {
                        e["event_id"] for e in events
                    }:
                        raise ValueError(
                            "calendar windows omitted or duplicated an event"
                        )
                    patient_rows.append(
                        {
                            **base,
                            "document_id": cohort
                            + "-full-history-"
                            + resolution
                            + "-"
                            + version,
                            "level": "patient",
                            "positive": any(e["positive"] for e in events),
                            "negative_type": "case"
                            if positive
                            else "synthea_other_conditions",
                            "window_document_ids": window_ids,
                            "n_windows": len(window_ids),
                            "n_events": len(events),
                            "window_start": events[0]["date"],
                            "window_end": events[-1]["date"],
                            "text": "\n".join(
                                e["date"] + ": " + e["text"] for e in events
                            ),
                        }
                    )
    metadata = {
        "generator": "synthea-csv-apr2020-v1",
        "source_url": SOURCE_URL,
        "source_sha256": SOURCE_SHA256,
        "source_patients": len(source_patients),
        "selected_unique_patients": len({p["patient_id"] for p in patient_rows}),
        "target_counts": counts,
        "excluded_conditions": [
            {
                "name": c.name,
                "concept_id": c.concept_id,
                "positive_patients": len(positives[c.concept_id]),
                "reason": "insufficient_recorded_cases_or_controls",
            }
            for c in excluded
        ],
        "source_patient_split_attempt": attempt,
        "label_policy": "recorded_SNOMED_diagnosis_or_indication_in_verified_descendant_set",
        "implicit_policy": "omit_all_diagnosis_and_indication_fields_then_remove_all_target_aliases",
        "paired_information_limit": "some_diagnosis_positive_events_have_no_other_peripheral_facts; has_peripheral_facts_is_recorded",
        "window_rows": len(windows),
        "patient_rows": len(patient_rows),
        "atomic_rows": len(atomic),
    }
    return windows, patient_rows, atomic, metadata
