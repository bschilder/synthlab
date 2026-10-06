from collections import defaultdict

from synthlab.ehr_benchmark import generate_ehr_benchmark, window_start


def test_paired_windows_labels_and_patient_splits():
    rows = generate_ehr_benchmark(patients_per_condition=12, seed=1)
    groups = defaultdict(list)
    for row in rows:
        groups[row["patient_id"]].append(row)
        assert row["concept_id"] not in row["text"]
        if row["explicitness"] == "implicit":
            assert row["condition"].lower() not in row["text"].lower()
            # Section presence cannot serve as a universal case/control shortcut.
            assert "Symptoms:" in row["text"]
            assert "Findings:" in row["text"]
            assert "Treatment reviewed:" in row["text"]
    for pair in groups.values():
        assert len(pair) == 6
        assert len({r["split"] for r in pair}) == 1
        assert len({r["positive"] for r in pair}) == 1
        assert {r["resolution"] for r in pair} == {"1d", "1mo", "1y"}
    assert {r["split"] for r in rows} == {"train", "validation", "test"}
    assert {r["negative_type"] for r in rows if not r["positive"]} >= {
        "healthy",
        "lookalike",
        "negation",
        "family_history",
    }
    assert rows == generate_ehr_benchmark(patients_per_condition=12, seed=1)


def test_calendar_windows_include_leap_day():
    from datetime import date

    assert window_start(date(2024, 3, 31), "1mo") == date(2024, 2, 29)
    assert window_start(date(2024, 2, 29), "1y") == date(2023, 2, 28)


def test_longitudinal_labels_cover_full_history_and_all_events_once():
    from synthlab.ehr_benchmark import CONDITIONS, generate_longitudinal_ehr_benchmark

    windows, patients, events = generate_longitudinal_ehr_benchmark(
        patients_per_condition=36, conditions=CONDITIONS[:1]
    )
    by_document = {row["document_id"]: row for row in windows}
    splits = defaultdict(set)
    for row in [*windows, *patients, *events]:
        splits[row["patient_id"]].add(row["split"])
    assert all(len(values) == 1 for values in splits.values())
    for patient in patients:
        selected = [by_document[d] for d in patient["window_document_ids"]]
        assert patient["positive"] == any(row["positive"] for row in selected)
        event_ids = [event for row in selected for event in row["event_ids"]]
        assert len(event_ids) == len(set(event_ids)) == 11
    old_only = next(
        r
        for r in patients
        if r["patient_id"].endswith("-1-0003")
        and r["resolution"] == "1y"
        and r["explicitness"] == "implicit"
    )
    selected = [by_document[d] for d in old_only["window_document_ids"]]
    assert old_only["positive"]
    assert not max(selected, key=lambda r: r["window_end"])["positive"]
    assert any(row["positive"] for row in selected)
