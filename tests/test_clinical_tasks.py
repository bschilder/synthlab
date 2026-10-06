import copy
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import pytest

from synthlab.clinical_tasks import (
    generate_clinical_tasks,
    generate_controlled_stress,
    patient_split,
    write_clinical_bundle,
)
from synthlab.ehr_benchmark import CONDITIONS

FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "clinical_tiny.json").read_text()
)
PANEL = tuple(c for c in CONDITIONS if c.concept_id in FIXTURE["concept_sets"])
FAMILIES = {"respiratory": ["SNOMED:195967001", "SNOMED:13645005", "SNOMED:84114007"]}


def generate(**kwargs):
    return generate_clinical_tasks(
        FIXTURE["tables"],
        FIXTURE["concept_sets"],
        conditions=PANEL,
        differential_families=FAMILIES,
        cutoff_dates=FIXTURE["cutoff_dates"],
        source_end_date=FIXTURE["source_end_date"],
        **kwargs,
    )


def patient_row(bundle, patient, task="multilabel", version="implicit", horizon=None):
    docs = {d["document_id"]: d for d in bundle["documents"]}
    return next(
        (docs[row["document_id"]], row)
        for row in bundle["labels"]
        if row["patient_id"] == patient
        and row["task"] == task
        and row["level"] == "patient"
        and docs[row["document_id"]]["explicitness"] == version
        and docs[row["document_id"]]["resolution"] == "1d"
        and (horizon is None or row["horizon_days"] == horizon)
    )


def represented_text(doc, by_id):
    if doc.get("window_document_ids"):
        return "\n".join(by_id[key]["text"] for key in doc["window_document_ids"])
    return doc["text"]


def test_all_patients_natural_prevalence_global_split_and_no_input_labels():
    original = copy.deepcopy(FIXTURE)
    bundle = generate()
    assert FIXTURE == original
    patients = {row["Id"] for row in FIXTURE["tables"]["patients"]}
    assert {doc["patient_id"] for doc in bundle["documents"]} == patients
    assert bundle["metadata"]["source_patients"] == 8
    assert bundle["metadata"]["patients_without_events"] == 1
    assert bundle["metadata"]["source_archive_verified"] is False
    assert bundle["metadata"]["source_sha256"] is None
    assert bundle["metadata"]["prevalence"]["SNOMED:44054006"]["positive_patients"] == 2
    assert bundle["metadata"]["prevalence"]["SNOMED:84114007"]["positive_patients"] == 1
    splits = defaultdict(set)
    assert len({doc["document_id"] for doc in bundle["documents"]}) == len(
        bundle["documents"]
    )
    assert {row["document_id"] for row in bundle["labels"]} == {
        row["document_id"] for row in bundle["documents"]
    }
    for doc in bundle["documents"]:
        splits[doc["patient_id"]].add(doc["split"])
        assert not {
            "gold_labels",
            "information_status",
            "eligible_mask",
            "first_recorded_dates",
        } & set(doc)
        assert doc["patient_id"] not in doc["text"]
    assert all(values == {patient_split(patient)} for patient, values in splits.items())
    # Panel size and task selection never alter split allocation.
    small = generate_clinical_tasks(
        FIXTURE["tables"],
        FIXTURE["concept_sets"],
        conditions=PANEL[:1],
        tasks=("multilabel",),
    )
    assert all(
        splits[doc["patient_id"]] == {doc["split"]} for doc in small["documents"]
    )
    assert bundle["metadata"]["counts"]["unique_texts"] < len(bundle["documents"])


def test_panel_wide_masking_shared_text_comorbidities_and_evidence_semantics():
    bundle = generate()
    texts = defaultdict(set)
    by_id = {doc["document_id"]: doc for doc in bundle["documents"]}
    for doc in bundle["documents"]:
        if doc["explicitness"] == "implicit":
            assert not any(
                term in doc["text"].lower()
                for term in (
                    "asthma",
                    "copd",
                    "diabetes",
                    "heart failure",
                    "chronic obstructive",
                )
            )
        if doc["task"] != "incident":
            texts[
                (
                    doc["patient_id"],
                    doc["resolution"],
                    doc["explicitness"],
                    doc["level"],
                    doc["window_start"],
                )
            ].add(doc["text"])
    assert all(len(values) == 1 for values in texts.values())
    _, comorbid = patient_row(bundle, "toy-prevalent", task="differential")
    assert comorbid["gold_labels"] == ["SNOMED:195967001", "SNOMED:13645005"]
    _, hidden = patient_row(bundle, "toy-diagnosis-only", task="evidence")
    assert hidden["gold_labels"] == ["SNOMED:84114007"]
    assert (
        hidden["information_status"]["SNOMED:84114007"] == "recorded_hidden_no_context"
    )
    assert hidden["evidence_eligible_ids"] == []
    assert "gold_abstain" not in hidden
    _, visible = patient_row(
        bundle, "toy-diagnosis-only", task="evidence", version="explicit"
    )
    assert visible["evidence_eligible_ids"] == ["SNOMED:84114007"]
    assert visible["evidence_event_ids"]["SNOMED:84114007"] == [
        "toy-diagnosis-only-2018-06-01"
    ]
    for row in bundle["labels"]:
        assert row["eligible_mask"] == [
            key in row["eligible_ids"] for key in row["candidate_ids"]
        ]
        if row["task"] != "incident":
            assert all(
                set(ids) <= set(by_id[row["document_id"]]["event_ids"])
                for ids in row["evidence_event_ids"].values()
            )


def test_full_history_each_event_covered_once_at_all_resolutions():
    bundle = generate()
    docs = {row["document_id"]: row for row in bundle["documents"]}
    for patient in [row for row in docs.values() if row["level"] == "patient"]:
        windows = [docs[identifier] for identifier in patient["window_document_ids"]]
        covered = [
            identifier for window in windows for identifier in window["event_ids"]
        ]
        assert len(covered) == len(set(covered))
        assert set(covered) == set(patient["event_ids"])
        if patient["task"] != "incident":
            assert patient["event_ids"] == [
                event["event_id"]
                for event in bundle["events"]
                if event["patient_id"] == patient["patient_id"]
            ]


def test_incident_uses_only_pre_cutoff_and_respects_risk_sets_censoring():
    bundle = generate()
    events = {e["event_id"]: e for e in bundle["events"]}
    for doc in bundle["documents"]:
        if doc["task"] == "incident":
            assert "POSTCUTOFF" not in doc["text"]
            assert all(
                events[identifier]["date"] <= doc["cutoff_date"]
                for identifier in doc["event_ids"]
            )
            assert doc["window_end"] <= doc["cutoff_date"]
    for horizon in (90, 365):
        doc, incident = patient_row(
            bundle, "toy-incident", task="incident", horizon=horizon
        )
        assert incident["gold_labels"] == ["SNOMED:44054006"]
        assert (
            incident["information_status"]["SNOMED:44054006"]
            == "future_recorded_target"
        )
        assert (
            "toy-incident-2020-02-01"
            in incident["evidence_event_ids"]["SNOMED:44054006"]
        )
        assert not set(incident["evidence_event_ids"]["SNOMED:44054006"]) & set(
            doc["event_ids"]
        )
        _, prevalent = patient_row(
            bundle, "toy-prevalent", task="incident", horizon=horizon
        )
        assert prevalent["information_status"]["SNOMED:44054006"] == "prevalent"
        assert "SNOMED:44054006" not in prevalent["eligible_ids"]
        for patient in ("toy-censored", "toy-death"):
            _, row = patient_row(bundle, patient, task="incident", horizon=horizon)
            assert set(row["information_status"].values()) == {"censored"}
            assert row["eligible_ids"] == []
        for patient in ("toy-newborn", "toy-no-events"):
            _, row = patient_row(bundle, patient, task="incident", horizon=horizon)
            assert set(row["information_status"].values()) == {"not_at_risk"}
        _, negative = patient_row(
            bundle, "toy-control", task="incident", horizon=horizon
        )
        assert negative["gold_labels"] == []
        assert negative["eligible_ids"] == negative["candidate_ids"]
        assert negative["followup_end"] >= negative["prediction_end_date"]
    short, _ = patient_row(bundle, "toy-incident", task="incident", horizon=90)
    long, _ = patient_row(bundle, "toy-incident", task="incident", horizon=365)
    assert short["window_document_ids"] == long["window_document_ids"]
    docs = {doc["document_id"]: doc for doc in bundle["documents"]}
    assert all(docs[key]["task"] == "history" for key in short["window_document_ids"])
    assert not any(
        doc["task"] == "incident" and doc["level"] == "event"
        for doc in bundle["documents"]
    )
    risk = bundle["metadata"]["risk_sets"]["2020-01-01:90"]
    assert risk["patients"] == 8
    assert risk["conditions"]["SNOMED:44054006"] == {
        "eligible": 2,
        "positive": 1,
        "negative": 1,
        "censored": 2,
        "prevalent": 1,
        "not_at_risk": 3,
    }


def test_incident_boundaries_fixed_cutoffs_and_no_same_day_leakage():
    source = copy.deepcopy(FIXTURE["tables"])
    source["conditions"].append(
        {
            "PATIENT": "toy-control",
            "START": "2020-01-01",
            "CODE": "195967001",
            "DESCRIPTION": "Asthma",
        }
    )
    bundle = generate_clinical_tasks(
        source,
        FIXTURE["concept_sets"],
        conditions=PANEL,
        cutoff_dates=["2020-01-01"],
        horizons=[90],
        source_end_date=FIXTURE["source_end_date"],
    )
    _, row = patient_row(bundle, "toy-control", task="incident", horizon=90)
    assert row["information_status"]["SNOMED:195967001"] == "prevalent"
    assert "SNOMED:195967001" not in row["gold_labels"]
    with pytest.raises(ValueError, match="source_end_date"):
        generate_clinical_tasks(
            source,
            FIXTURE["concept_sets"],
            conditions=PANEL,
            cutoff_dates=["2020-01-01"],
        )
    with pytest.raises(ValueError, match="unvalidated"):
        generate_clinical_tasks(
            source,
            FIXTURE["concept_sets"],
            conditions=PANEL,
            differential_families={"bad": ["SNOMED:1", "SNOMED:2"]},
        )


def test_controlled_clue_deletion_is_paired_ambiguous_and_assertion_aware():
    bundle = generate_controlled_stress(
        conditions=PANEL, differential_families=FAMILIES, repetitions=1
    )
    docs = {row["document_id"]: row for row in bundle["documents"]}
    events = {row["event_id"]: row for row in bundle["events"]}
    assert len(events) == len(bundle["events"])
    assert any(e["assertion"] == "negated" for e in events.values())
    assert any(e["assertion"] == "family_history" for e in events.values())
    deleted = set()
    paired = defaultdict(set)
    for row in bundle["labels"]:
        doc = docs[row["document_id"]]
        paired[doc["patient_id"]].add(doc["split"])
        if row["stress_variant"] == "sufficient":
            assert row["gold_abstain"] is False
            assert row["gold_labels"] == row["latent_gold_labels"]
            assert all(
                events[event]["assertion"] == "affirmed"
                for ids in row["evidence_event_ids"].values()
                for event in ids
            )
        else:
            assert row["gold_abstain"] is True
            assert row["gold_labels"] == []
            assert row["evidence_eligible_ids"] == []
        if row["stress_variant"] == "clue_deleted":
            deleted.add(represented_text(doc, docs))
    assert len(deleted) == 1  # Different latent diseases become indistinguishable.
    assert all(len(splits) == 1 for splits in paired.values())
    assert any(len(row["gold_labels"]) == 2 for row in bundle["labels"])
    assert bundle["metadata"]["corpus"] == "controlled_clue_deletion_stress"


def test_incident_horizon_boundary_and_indication_before_formal_diagnosis():
    source = copy.deepcopy(FIXTURE["tables"])
    for day, code, description in [
        ("2020-03-31", "195967001", "Asthma"),
        ("2020-04-01", "13645005", "Chronic obstructive pulmonary disease"),
    ]:
        source["conditions"].append(
            {
                "PATIENT": "toy-control",
                "START": day,
                "CODE": code,
                "DESCRIPTION": description,
            }
        )
    source["conditions"].append(
        {
            "PATIENT": "toy-incident",
            "START": "2020-05-01",
            "CODE": "44054006",
            "DESCRIPTION": "Type 2 diabetes mellitus",
        }
    )
    bundle = generate_clinical_tasks(
        source,
        FIXTURE["concept_sets"],
        conditions=PANEL,
        cutoff_dates=["2020-01-01"],
        source_end_date=FIXTURE["source_end_date"],
    )
    _, short = patient_row(bundle, "toy-control", task="incident", horizon=90)
    _, long = patient_row(bundle, "toy-control", task="incident", horizon=365)
    assert short["prediction_end_date"] == "2020-03-31"
    assert short["gold_labels"] == ["SNOMED:195967001"]
    assert set(long["gold_labels"]) == {"SNOMED:195967001", "SNOMED:13645005"}
    _, incident = patient_row(bundle, "toy-incident", task="incident", horizon=90)
    assert incident["first_recorded_dates"]["SNOMED:44054006"] == "2020-02-01"
    assert incident["gold_labels"] == ["SNOMED:44054006"]


def test_all_natural_tasks_share_lossless_full_panel_windows_and_compact_patient_text():
    bundle = generate()
    docs = {row["document_id"]: row for row in bundle["documents"]}
    labels = {row["document_id"]: row for row in bundle["labels"]}
    events = {row["event_id"]: row for row in bundle["events"]}
    references = defaultdict(set)
    windows = [doc for doc in docs.values() if doc["level"] == "event"]
    assert {doc["task"] for doc in windows} == {"history"}
    assert all(
        labels[doc["document_id"]]["candidate_ids"]
        == bundle["metadata"]["candidate_ids"]
        for doc in windows
    )
    for doc in docs.values():
        if doc["level"] != "patient":
            continue
        version = doc["explicitness"]
        expected = "\n".join(
            events[key]["date"] + ": " + events[key][version + "_text"]
            for key in doc["event_ids"]
        )
        if doc["event_ids"]:
            assert len(doc["text"]) < 100
            assert doc["text_reference_policy"] == "all_referenced_dated_windows"
            assert represented_text(doc, docs) == expected
            assert (
                doc["input_text_sha256"]
                == hashlib.sha256(expected.encode()).hexdigest()
            )
        else:
            assert doc["window_document_ids"] == []
        if doc["task"] != "incident":
            references[(doc["patient_id"], doc["resolution"], version)].add(
                tuple(doc["window_document_ids"])
            )
            observed = {
                target
                for key in doc["event_ids"]
                for target in events[key]["recorded_labels"]
            }
            assert set(labels[doc["document_id"]]["gold_labels"]) == observed & set(
                labels[doc["document_id"]]["candidate_ids"]
            )
    assert all(len(values) == 1 for values in references.values())
    shared = patient_row(bundle, "toy-incident", task="multilabel")[0][
        "window_document_ids"
    ]
    past = patient_row(bundle, "toy-incident", task="incident", horizon=90)[0][
        "window_document_ids"
    ]
    assert set(past) < set(shared)  # Complete old windows reuse the same natural input.


def test_shared_window_at_cutoff_is_clipped_without_losing_future_full_history_facts():
    source = copy.deepcopy(FIXTURE["tables"])
    source["observations"].append(
        {
            "PATIENT": "toy-control",
            "DATE": "2019-12-15",
            "DESCRIPTION": "AFTER_DECEMBER_CUTOFF",
            "VALUE": "99",
            "UNITS": "mg/dL",
        }
    )
    source["conditions"].append(
        {
            "PATIENT": "toy-control",
            "START": "2019-12-15",
            "CODE": "13645005",
            "DESCRIPTION": "Chronic obstructive pulmonary disease",
        }
    )
    bundle = generate_clinical_tasks(
        source,
        FIXTURE["concept_sets"],
        conditions=PANEL,
        differential_families=FAMILIES,
        cutoff_dates=["2019-12-01"],
        source_end_date=FIXTURE["source_end_date"],
    )
    docs = {row["document_id"]: row for row in bundle["documents"]}
    incident = [
        doc
        for doc in docs.values()
        if doc["task"] == "incident" and doc["patient_id"] == "toy-control"
    ]
    natural = [
        doc
        for doc in docs.values()
        if doc["task"] == "multilabel"
        and doc["level"] == "patient"
        and doc["patient_id"] == "toy-control"
    ]
    for doc in incident:
        assert "AFTER_DECEMBER_CUTOFF" not in represented_text(doc, docs)
        assert all(
            docs[key]["window_end"] <= doc["cutoff_date"]
            for key in doc["window_document_ids"]
        )
    assert all(
        "AFTER_DECEMBER_CUTOFF" in represented_text(doc, docs) for doc in natural
    )
    month = next(doc for doc in incident if doc["resolution"] == "1mo")
    clipped = docs[month["window_document_ids"][0]]
    assert clipped["window_start"] == "2019-12-01"
    assert clipped["window_end"] == "2019-12-01"
    full = next(
        doc
        for doc in natural
        if doc["resolution"] == "1mo" and doc["explicitness"] == month["explicitness"]
    )
    assert clipped["document_id"] not in full["window_document_ids"]


def test_immutable_export_digests_and_reproducibility(tmp_path):
    bundle = generate()
    assert bundle == generate()
    destination = tmp_path / "bundle"
    write_clinical_bundle(bundle, destination)
    manifest = json.loads((destination / "SHA256SUMS.json").read_text())
    for filename, digest in manifest.items():
        assert (
            hashlib.sha256((destination / filename).read_bytes()).hexdigest() == digest
        )
    with pytest.raises(FileExistsError):
        write_clinical_bundle(bundle, destination)
    assert sum(1 for _ in (destination / "documents.jsonl").open()) == len(
        bundle["documents"]
    )
