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
