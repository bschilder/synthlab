"""Deterministic paired clinical vignettes for controlled representation experiments.

These are synthetic templates, not Synthea or clinical validation data. The same
patient's dated events supply all calendar windows and explicit/implicit variants.
Disease presence is metadata only; implicit text is audited against supplied aliases.
"""

from __future__ import annotations

import calendar
import random
import re
from dataclasses import dataclass
from datetime import date, timedelta


@dataclass(frozen=True)
class Condition:
    name: str
    concept_id: str
    aliases: tuple[str, ...]
    symptoms: tuple[str, ...]
    medications: tuple[str, ...]
    findings: tuple[str, ...]
    lookalikes: tuple[str, ...]


CONDITIONS = (
    Condition(
        "Type 2 diabetes mellitus",
        "SNOMED:44054006",
        ("type 2 diabetes", "diabetes mellitus", "T2DM"),
        ("increased thirst", "frequent urination", "blurred vision"),
        ("metformin", "empagliflozin"),
        ("HbA1c 8.2%", "fasting glucose 178 mg/dL"),
        (
            "transient thirst after exercise; HbA1c 5.3%",
            "frequent voiding with normal serum glucose",
        ),
    ),
    Condition(
        "Asthma",
        "SNOMED:195967001",
        ("bronchial asthma",),
        ("episodic wheezing", "nighttime cough", "exertional chest tightness"),
        ("albuterol inhaler", "inhaled budesonide"),
        ("reversible airflow obstruction on spirometry",),
        (
            "cough with postnasal drip; spirometry normal",
            "brief breathlessness during a panic episode",
        ),
    ),
    Condition(
        "Chronic obstructive pulmonary disease",
        "SNOMED:13645005",
        ("COPD",),
        (
            "progressive breathlessness",
            "persistent productive cough",
            "reduced exercise tolerance",
        ),
        ("tiotropium", "formoterol"),
        ("post-bronchodilator FEV1/FVC 0.58",),
        (
            "transient cough following an upper respiratory infection",
            "breathlessness with normal pulmonary function",
        ),
    ),
    Condition(
        "Heart failure",
        "SNOMED:84114007",
        ("congestive heart failure", "CHF"),
        ("orthopnea", "ankle edema", "exertional dyspnea"),
        ("furosemide", "sacubitril/valsartan"),
        ("LVEF 30%", "BNP 920 pg/mL"),
        (
            "dependent ankle edema; BNP normal and LVEF 62%",
            "breathlessness associated with deconditioning",
        ),
    ),
    Condition(
        "Chronic kidney disease",
        "SNOMED:709044004",
        ("CKD", "chronic renal disease"),
        ("fatigue", "nocturia", "mild peripheral edema"),
        ("losartan", "sodium bicarbonate"),
        ("eGFR 34 mL/min/1.73m2 on repeated tests", "urine albumin elevated"),
        (
            "transient creatinine rise after dehydration; eGFR recovered to 90",
            "nocturia with normal creatinine",
        ),
    ),
    Condition(
        "Rheumatoid arthritis",
        "SNOMED:69896004",
        ("rheumatoid disease",),
        (
            "symmetric small-joint swelling",
            "morning stiffness over an hour",
            "painful hand joints",
        ),
        ("methotrexate", "hydroxychloroquine"),
        ("anti-CCP antibody positive", "elevated ESR"),
        (
            "isolated joint pain after mechanical strain; inflammatory markers normal",
            "brief stiffness after inactivity",
        ),
    ),
    Condition(
        "Epilepsy",
        "SNOMED:84757009",
        ("seizure disorder",),
        (
            "recurrent unprovoked convulsions",
            "post-event confusion",
            "brief loss of awareness",
        ),
        ("levetiracetam", "lamotrigine"),
        ("interictal epileptiform discharges on EEG",),
        (
            "fainting after prolonged standing with rapid recovery",
            "shaking during low glucose that resolved after juice",
        ),
    ),
    Condition(
        "Migraine",
        "SNOMED:37796009",
        ("migraine disorder",),
        (
            "unilateral pulsating headache",
            "photophobia",
            "nausea accompanying headache",
        ),
        ("sumatriptan", "topiramate"),
        ("recurrent episodes lasting 12 hours with normal neurological exam",),
        (
            "bilateral pressure headache after a stressful day",
            "brief head discomfort with nasal congestion",
        ),
    ),
    Condition(
        "Hypertensive disorder",
        "SNOMED:38341003",
        ("hypertension", "essential hypertension"),
        ("intermittent headache", "occasional lightheadedness"),
        ("amlodipine", "chlorthalidone"),
        ("home blood pressure repeatedly 158/96 mmHg",),
        (
            "clinic BP 155/90 during anxiety; home readings 118/74",
            "headache with blood pressure 116/72",
        ),
    ),
    Condition(
        "Hypothyroidism",
        "SNOMED:40930008",
        ("hypothyroid",),
        ("cold intolerance", "constipation", "fatigue"),
        ("levothyroxine",),
        ("TSH 18 mIU/L with low free T4",),
        (
            "fatigue after poor sleep; TSH and free T4 normal",
            "cold hands with normal thyroid function tests",
        ),
    ),
    Condition(
        "Pneumonia",
        "SNOMED:233604007",
        ("lung infection",),
        ("fever", "productive cough", "pleuritic pain"),
        ("amoxicillin", "doxycycline"),
        ("focal lower-lobe infiltrate on chest imaging", "oxygen saturation 91%"),
        (
            "dry cough and rhinorrhea; chest imaging clear",
            "pleuritic discomfort after exercise without fever",
        ),
    ),
    Condition(
        "Depressive disorder",
        "SNOMED:35489007",
        ("depression", "major depressive disorder"),
        ("persistent low mood", "anhedonia", "early morning awakening"),
        ("sertraline", "escitalopram"),
        ("PHQ-9 score 19 for more than two weeks",),
        (
            "brief sadness after a difficult day with preserved interests",
            "fatigue from shift work; mood stable",
        ),
    ),
    Condition(
        "Osteoarthritis",
        "SNOMED:396275006",
        ("degenerative joint disease",),
        ("knee pain with activity", "brief morning stiffness", "crepitus"),
        ("topical diclofenac", "acetaminophen"),
        ("joint-space narrowing and osteophytes on radiograph",),
        (
            "knee pain after a minor injury; radiograph normal",
            "diffuse muscle soreness after exercise",
        ),
    ),
    Condition(
        "Atrial fibrillation",
        "SNOMED:49436004",
        ("AFib",),
        ("palpitations", "episodic lightheadedness", "exercise intolerance"),
        ("apixaban", "metoprolol"),
        ("irregularly irregular rhythm without P waves on ECG",),
        (
            "brief palpitations with sinus rhythm on monitoring",
            "regular tachycardia during anxiety",
        ),
    ),
    Condition(
        "Obstructive sleep apnea syndrome",
        "SNOMED:78275009",
        ("obstructive sleep apnea", "OSA"),
        ("loud snoring", "witnessed breathing pauses", "daytime somnolence"),
        ("positive airway pressure therapy",),
        ("apnea-hypopnea index 28 per hour",),
        (
            "daytime sleepiness after short sleep duration; polysomnography normal",
            "snoring without observed breathing pauses",
        ),
    ),
    Condition(
        "Hyperlipidemia",
        "SNOMED:55822004",
        ("hyperlipidaemia", "dyslipidemia"),
        ("no new symptoms",),
        ("atorvastatin", "rosuvastatin"),
        ("LDL cholesterol 192 mg/dL", "total cholesterol 280 mg/dL"),
        (
            "LDL cholesterol 88 mg/dL on routine testing",
            "family risk discussed; lipid panel normal",
        ),
    ),
)


def window_start(end: date, resolution: str) -> date:
    """Calendar arithmetic, clamped at month ends; selected interval is (start, end]."""
    if resolution == "1d":
        return end - timedelta(days=1)
    if resolution == "1mo":
        year, month = (
            (end.year - 1, 12) if end.month == 1 else (end.year, end.month - 1)
        )
    elif resolution == "1y":
        year, month = end.year - 1, end.month
    else:
        raise ValueError("resolution must be 1d, 1mo, or 1y")
    return date(year, month, min(end.day, calendar.monthrange(year, month)[1]))


def alias_mentions(text: str, aliases):
    return [
        a
        for a in aliases
        if a.strip()
        and re.search(
            r"(?<!\w)" + re.escape(a.strip()) + r"(?!\w)", text, re.IGNORECASE
        )
    ]


def _redact(text, aliases):
    for alias in sorted(aliases, key=len, reverse=True):
        if alias.strip():
            text = re.sub(
                r"(?<!\w)" + re.escape(alias.strip()) + r"(?!\w)",
                "the condition under evaluation",
                text,
                flags=re.IGNORECASE,
            )
    return text


def generate_ehr_benchmark(
    *, patients_per_condition=120, seed=42, extra_aliases=None, conditions=CONDITIONS
):
    """Balanced cases/controls within each condition and patient split.

    Split allocation is stratified by condition and case/control status. Four negative
    types distinguish absent disease from affirmative clinical evidence. Each patient
    gets six paired documents. Templates are a pilot; diverse Synthea notes should
    independently replicate any positive result before a foundation-model conclusion.
    """
    if patients_per_condition < 12 or patients_per_condition % 2:
        raise ValueError("patients_per_condition must be even and at least 12")
    rng = random.Random(seed)
    rows = []
    extra_aliases = extra_aliases or {}
    all_aliases = [
        a
        for c in conditions
        for a in (c.name, *c.aliases, *extra_aliases.get(c.concept_id, []))
    ]
    healthy = (
        "Routine preventive visit; appetite and activity normal.",
        "No active complaints. Examination and screening tests within reference ranges.",
        "Follow-up after travel; feels well, no ongoing treatment required.",
    )
    background = (
        "Reports irregular sleep during a busy work week.",
        "Discussed hydration, physical activity, and routine follow-up.",
        "Seasonal nasal congestion improved with saline spray.",
        "Prior minor muscle strain resolved without ongoing symptoms.",
        "Routine vaccination reviewed and preventive care updated.",
    )
    for condition in conditions:
        aliases = (
            condition.name,
            *condition.aliases,
            *extra_aliases.get(condition.concept_id, []),
        )
        for positive in (False, True):
            count = patients_per_condition // 2
            n_train = max(1, int(count * 0.6))
            n_val = max(1, int(count * 0.2))
            splits = (
                ["train"] * n_train
                + ["validation"] * n_val
                + ["test"] * (count - n_train - n_val)
            )
            rng.shuffle(splits)
            negative_types = ["healthy", "lookalike", "negation", "family_history"] * (
                (count + 3) // 4
            )
            rng.shuffle(negative_types)
            for i, split in enumerate(splits):
                patient_id = (
                    f"{condition.concept_id.split(':')[-1]}-{int(positive)}-{i:04d}"
                )
                anchor = date(2025, rng.randint(1, 12), rng.randint(1, 28))
                age = rng.randint(30, 85)
                negative_type = "case" if positive else negative_types[i]
                if positive:
                    symptoms = rng.sample(
                        condition.symptoms, rng.randint(1, len(condition.symptoms))
                    )
                    clinical = "; ".join(symptoms)
                    diagnosis = rng.choice(
                        [
                            "Assessment: {name}.",
                            "Known history of {name}.",
                            "Follow-up for {name}.",
                        ]
                    ).format(name=condition.name)
                else:
                    clinical = (
                        rng.choice(healthy)
                        if negative_type == "healthy"
                        else rng.choice(condition.lookalikes)
                    )
                    diagnosis = {
                        "healthy": "Screening discussion: no evidence of {name}.",
                        "lookalike": "Evaluation did not establish {name}.",
                        "negation": "Investigations exclude {name} in this patient.",
                        "family_history": "A relative has {name}; patient evaluation is negative.",
                    }[negative_type].format(name=condition.name)

                def visit_text(condition, positive, clinical):
                    # Match section headings in cases/controls; only clinical content differs.
                    complaint = (
                        "; ".join(
                            rng.sample(
                                condition.symptoms,
                                rng.randint(1, len(condition.symptoms)),
                            )
                        )
                        if positive
                        else clinical
                    )
                    finding = (
                        rng.choice(condition.findings)
                        if positive
                        else rng.choice(
                            (
                                "targeted investigations reassuring",
                                "no persistent abnormality on follow-up",
                            )
                        )
                    )
                    medication = (
                        rng.choice(condition.medications)
                        if positive
                        else rng.choice(
                            (
                                "saline nasal spray as needed",
                                "acetaminophen for occasional discomfort",
                                "daily multivitamin",
                            )
                        )
                    )
                    return f"Symptoms: {complaint}; Findings: {finding}; Treatment reviewed: {medication}."

                events = [
                    {
                        "date": (anchor - timedelta(days=offset)).isoformat(),
                        "text": (
                            visit_text(condition, positive, clinical)
                            if offset in (0, 7, 180)
                            else rng.choice(background)
                        ),
                        "diagnosis": (diagnosis if offset in (0, 7, 180) else ""),
                    }
                    for offset in (0, 3, 7, 21, 60, 180, 300)
                ]
                for resolution in ("1d", "1mo", "1y"):
                    start = window_start(anchor, resolution)
                    selected = [
                        e
                        for e in events
                        if start < date.fromisoformat(e["date"]) <= anchor
                    ]
                    for explicitness in ("explicit", "implicit"):
                        chunks = [
                            f"Adult aged {age}. Clinical record through {anchor.isoformat()}."
                        ]
                        for event in reversed(selected):
                            text = event["text"]
                            if explicitness == "explicit":
                                text += " " + event["diagnosis"]
                            else:
                                text = _redact(text, all_aliases)
                            chunks.append(event["date"] + ": " + text.strip())
                        text = "\n".join(chunks)
                        if explicitness == "implicit" and alias_mentions(
                            text, all_aliases
                        ):
                            raise ValueError("target-name leakage in implicit document")
                        rows.append(
                            {
                                "document_id": f"{patient_id}-{resolution}-{explicitness}",
                                "patient_id": patient_id,
                                "condition": condition.name,
                                "concept_id": condition.concept_id,
                                "positive": positive,
                                "negative_type": negative_type,
                                "split": split,
                                "resolution": resolution,
                                "explicitness": explicitness,
                                "window_start": start.isoformat(),
                                "window_end": anchor.isoformat(),
                                "n_events": len(selected),
                                "text": text,
                                "target_aliases": list(aliases),
                                "generator": "controlled-vignettes-v2",
                            }
                        )
    return rows
