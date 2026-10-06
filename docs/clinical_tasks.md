# Clinical tasks over the complete Synthea population

`synthlab.clinical_tasks` builds a natural-prevalence cohort and an independent
controlled stress corpus. It retains every source patient, including patients
without clinical events. It does not match or balance cases and controls.

The natural task labels are **recorded SNOMED diagnoses or indications**, using
root-plus-transitive-descendant sets prepared against a checked ontology release.
A zero means there is no matching recorded code in the specified observed scope;
it does not establish absence of disease. The diagnosis-hidden variant may omit
all information identifying a recorded positive. Peripheral fact availability is
reported as a source property, without claiming diagnostic adequacy.

## Generate on RunPod

Use BioDocs `scripts/snomed_ehr/prepare_synthea.py::prepare_inputs` with all
16 `synthlab.ehr_benchmark.CONDITIONS` to create `target_mapping.json` and obtain
the pinned official April 2020 CSV archive. The release-checked mapping contains
root/descendant memberships, ontology provenance and aliases, including observed
diagnosis/indication descriptions. No new live ontology query occurs in this CLI.

```bash
python -m synthlab.clinical_tasks \
  --source-archive /data/source/synthea_sample_data_csv_apr2020.zip \
  --mapping-json /data/source/target_mapping.json \
  --source-end-date auto \
  --output-dir /data/clinical/natural-v1 \
  --controlled-output-dir /data/clinical/stress-v1 \
  --controlled-repetitions 24
```

Default incident cutoffs are **January 1 of every year 2010 through 2019** and
prediction horizons are **90 and 365 days**. These dates are predetermined,
independent of diagnoses. `--cutoff-date` and `--horizon-days` may be repeated
for a separately declared experiment. `--task` may be repeated to select tasks;
the default creates all four. The API accepts a panel, families, tasks and
cutoffs, so tiny structural checks can avoid loading the clinical source.

Run complete source generation, embedding, training and evaluation on RunPod.
Only the eight-patient fixture and structural tests should run locally. Outputs
must be new directories: the writer refuses to replace any existing directory.
Publish artifacts privately under `standardmodelbio` with their checksum
manifest. The generator itself neither launches pods nor uploads datasets.

## Files and joins

| File | Purpose |
| --- | --- |
| `documents.jsonl` | Model inputs and nonclinical routing fields |
| `labels.jsonl` | Separate gold metadata, joined by `document_id` |
| `events.jsonl` | Source event provenance, dates, assertion metadata and paired text |
| `metadata.json` | Panel, mapping/source hashes, prevalence, risk sets and policies |
| `SHA256SUMS.json` | SHA-256 of the four immutable files |

Document fields are `document_id`, `patient_id`, `split`, `task`, `level`,
`resolution`, `explicitness`, `window_start`, `window_end`, `cutoff_date`, `text`,
`text_sha256`, and `event_ids`. Patient documents additionally have
`window_document_ids`; differential documents have `family_id`, and incident
documents have `horizon_days`. Dates use ISO format. Calendar windows are
inclusive; an empty history has `window_start=null` and no event references.

Nonempty patient histories use a short reference placeholder in `text`; the
encoder resolves every referenced window instead of embedding that placeholder.
All original clinical text remains in canonical windows and atomic events.
`input_text_sha256` hashes the exact full dated event text by streaming its
bytes, while `text_sha256` hashes the field actually stored in that row. The
clinical input fingerprint stays distinct even if a low-dimensional projection
maps different histories to identical vectors. Empty histories retain their
original no-record input text and have no windows.

Gold fields include ordered `candidate_ids`, `gold_labels`, `eligible_ids`,
`eligible_mask` in candidate order, `evidence_event_ids` mapping each candidate to
supporting source event IDs, `evidence_eligible_ids`, `information_status`,
`first_recorded_dates`, and `label_scope`. Routing fields are repeated in labels
for inspection; none of the gold fields enter model text. An incident label also
has `prediction_end_date`, `followup_end`, `source_end_date` and `last_event_date`.

Each event bundles records from the same source patient and day, retaining each
record's own `START`/`DATE`, independent of encounter links. All 11 clinical CSV
tables are retained, including old histories and orphan encounter references.
Administrative tables and identifying demographics are excluded from text.
Event IDs, patient IDs, encounter IDs and concept codes stay in metadata.

The fixed SHA-256 patient assignment uses `seed:patient_id` with approximately
60/20/20 train/validation/test allocation. It is independent of the panel, task,
resolution, diagnosis, and explicitness. Rare labels stay in the cohort. Their
zero-positive or zero-negative splits are reported rather than repaired through
resampling. This split differs from the legacy balanced case-control generator;
fit fresh clinical probes on this split.

The CLI verifies the pinned archive checksum. The in-memory API can also accept
fixtures; it marks archive provenance unverified until the CLI has checked it.
`source_tables_sha256` and `mapping_sha256` cover the actual supplied data.

## Text and temporal aggregation

Explicit text retains recorded diagnosis and indication descriptions. Implicit
text drops **all** diagnosis/indication fields and redacts aliases for the
**entire candidate panel** from remaining clinical facts. Differential and
multilabel tasks reuse identical generic text; there is no per-target redaction
variant, target prompt, disease-dependent placeholder or patient identifier.

Every representation is available at daily (`1d`), monthly (`1mo`) and annual
(`1y`) resolution, with explicit and implicit pairs. Full-history documents link
every window, and each source event occurs exactly once in a given history's
windows. Pool **all** referenced windows; do not discard old windows or truncate
to a last-year history. Embed every text chunk when an encoder has a token limit.
Deduplicate embedding computation by `text_sha256` across tasks and resolutions.

All natural patient documents refer to shared event-window documents with
`task="history"`. Multilabel, evidence and differential tasks reuse one canonical
window for each patient/resolution/boundary/explicitness combination. Its gold
covers the full validated panel; each patient row retains its task-specific gold
and candidate mask. Historical windows are reused across horizons and cutoffs;
the window intersecting a cutoff is clipped at that cutoff. Shared history
windows have only their own source-record labels, never future incident gold.
The evaluator must resolve `window_document_ids` globally and enforce the same
patient, split, resolution and explicitness. Every referenced event and window
end must be at or before the prediction cutoff. Incident metrics use patient
documents, with one risk label per patient/cutoff/horizon, not repeated window
labels. At ten cutoffs and two horizons the incident patient count is
`patients × 10 × 2 × 3 resolutions × 2 explicitness` (140,520 for 1,171 patients).
History-window rows are shared, and actual counts are recorded in metadata.
Storage scales as the canonical window count plus task-specific patient rows,
not one window copy per task. `counts.unique_input_texts` reports the exact
encoder input vocabulary, including atomic events used for attribution.

## Task definitions

**Multilabel identification.** Gold is the set of all panel candidates with a
matching recorded code in that window or full history. Comorbidities are retained.
No-record negatives remain eligible for record-detection metrics.

**Differential identification.** The same source population and inputs are
evaluated within declared candidate families: respiratory (asthma, COPD,
pneumonia, heart failure), arthritis (rheumatoid arthritis, osteoarthritis) and
fatigue (hypothyroidism, depressive disorder, obstructive sleep apnea). These
are evaluation groups, not claims of mutually exclusive clinical diagnoses.
Only families with at least two validated panel candidates are included.
Multiple family positives and patients with no family code are retained. A
forced single-choice accuracy is unsuitable when there are multiple positives
or none; evaluate multilabel/ranking behavior and report those strata.

**Incident record prediction.** Features contain only records dated
`date <= cutoff`. First-known membership includes **both diagnoses and every
indication/reason code**, avoiding medication indications predating a formal
diagnosis. A positive is the first recorded membership in
`(cutoff, cutoff + horizon]`. A candidate recorded at or before cutoff is
prevalent and ineligible. Birth after cutoff, no baseline records, or follow-up
ending before cutoff means not at risk. A future positive observed before
follow-up ends is eligible; a negative is eligible only with follow-up through
the entire horizon. Other candidate outcomes are censored. Same-day diagnosis
is prevalent, so it cannot be a future positive.

Follow-up conservatively ends at the earliest of the patient's last clinical
record, recorded death, and declared export bound. Death before the horizon is
censored for negative risk labels; this is not a competing-risk survival model.
`--source-end-date auto` uses the maximum captured clinical date as an export
bound proxy. Neither this global bound nor the last clinical record establishes
continuous observation: quiet healthy histories can be excluded. Metadata
reports this policy, source date, every calendar cutoff, age bins, and positive,
negative, prevalent, censored and not-at-risk counts by candidate and split.

**Evidence and abstention.** Natural evidence gold points to the source days
recording each candidate. Explicit source-record attribution is gradable;
implicit gold remains a code-record fact but is excluded from clinical evidence
scoring through empty `evidence_eligible_ids`. Natural data has no
`gold_abstain`: missing peripheral facts are insufficient to assign clinical
abstention ground truth.

Natural `information_status` values are `recorded_visible`,
`recorded_hidden_with_context`, `recorded_hidden_no_context`, and
`no_recorded_target`. Incident values are `future_recorded_target`, `prevalent`,
`no_recorded_target`, `censored`, or `not_at_risk`.

The separate controlled stress corpus contains a marked template finding for
one disease or two comorbid diseases, negated same-family findings, and findings
asserted only for a relative. `gold_diagnosis_assertions` records `affirmed`,
`negated`, and `family_history`; only affirmed events enter evidence gold. These
are generator rules, not a clinical validation of those findings. Paired clue
deletion removes every affirmed finding and disease name, leaving shared family
context or empty information. Different latent diseases then have identical
text. The deleted cases have `gold_abstain=true`, empty `gold_labels`,
`latent_gold_labels` preserved only in metadata, and all candidates listed as
acceptable unresolved differential options. Their information statuses are
`sufficient_by_construction` or `insufficient_by_construction`.

## Evaluation contract and tiny checks

Keep natural and stress results separate. Natural **AUPRC is primary**; include
micro and classwise/macro values, observed prevalence and train-prior baselines.
Report undefined rare conditions and class counts explicitly. AUROC is secondary.
Calibrate probabilities and choose multilabel/abstention thresholds using
validation only. Fit supervised probes on train only. Score abstention only
where `gold_abstain` exists, and source-event attribution only where candidate
IDs are evidence eligible. Never infer abstention from a code-record negative.

`tests/fixtures/clinical_tiny.json` is an eight-patient structural fixture, not a
performance benchmark. It includes descendant-code indications, overlapping
diagnoses, no-event patients, no peripheral facts, incomplete follow-up, death,
and birth after cutoff. The tests verify immutable exports, source-label
separation, global splits, all-event coverage, shared historical windows,
temporal leakage prevention and controlled evidence/abstention rules:

```bash
python -m pytest tests/test_clinical_tasks.py tests/test_synthea_benchmark.py tests/test_ehr_benchmark.py -q
```
