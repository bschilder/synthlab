from collections import defaultdict

from synthlab.ehr_benchmark import CONDITIONS
from synthlab.synthea_benchmark import generate_synthea_benchmark, histories_from_csv


def tables():
    result = {
        k: []
        for k in [
            "patients",
            "encounters",
            "conditions",
            "observations",
            "medications",
            "procedures",
            "careplans",
        ]
    }
    for index in range(40):
        patient = f"p{index}"
        result["patients"].append({"Id": patient})
        for visit, day in [("old", "2001-01-01"), ("recent", "2020-02-29")]:
            event = patient + "-" + visit
            result["encounters"].append(
                {
                    "Id": event,
                    "PATIENT": patient,
                    "START": day,
                    "DESCRIPTION": "Outpatient assessment",
                }
            )
            result["observations"].append(
                {
                    "PATIENT": patient,
                    "ENCOUNTER": event,
                    "DATE": day,
                    "DESCRIPTION": "Serum glucose",
                    "VALUE": "180" if index < 20 and visit == "old" else "90",
                    "UNITS": "mg/dL",
                }
            )
        if index < 20:
            result["conditions"].append(
                {
                    "PATIENT": patient,
                    "ENCOUNTER": patient + "-old",
                    "START": "2001-01-01",
                    "CODE": "44054006",
                    "DESCRIPTION": "Type 2 diabetes mellitus",
                }
            )
    return result


def test_synthea_ground_truth_full_history_and_alias_removal():
    condition = CONDITIONS[0]
    windows, patients, events, metadata = generate_synthea_benchmark(
        tables(),
        {condition.concept_id: {"44054006"}},
        conditions=[condition],
        minimum_cases=20,
    )
    by_document = {r["document_id"]: r for r in windows}
    splits = defaultdict(set)
    for row in [*windows, *patients, *events]:
        splits[row["patient_id"]].add(row["split"])
        assert row["patient_id"] not in row["text"]
        if row["explicitness"] == "implicit":
            assert "diabetes" not in row["text"].lower()
            assert "condition under evaluation" not in row["text"]
    assert all(len(values) == 1 for values in splits.values())
    for patient in patients:
        selected = [by_document[d] for d in patient["window_document_ids"]]
        ids = [e for row in selected for e in row["event_ids"]]
        assert len(ids) == len(set(ids)) == 2
        assert patient["positive"] == any(row["positive"] for row in selected)
        if patient["positive"]:
            assert not max(selected, key=lambda row: row["window_end"])["positive"]
    assert metadata["source_patients"] == 40
    assert metadata["excluded_conditions"] == []


def test_orphan_conditions_are_retained_as_old_dated_events():
    source = tables()
    source["conditions"].append(
        {
            "PATIENT": "p39",
            "ENCOUNTER": "missing",
            "START": "1956-03-01",
            "CODE": "44054006",
            "DESCRIPTION": "Type 2 diabetes mellitus",
        }
    )
    history = histories_from_csv(source)["p39"]
    assert history[0]["date"] == "1956-03-01"
    assert "44054006" in history[0]["codes"]
