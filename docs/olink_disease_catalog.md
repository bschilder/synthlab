# Olink disease-effect catalog

This document walks through the schema, sourcing rules, and contribution
checklist for
[`synthlab/data/olink_disease_effects.csv`](../synthlab/data/olink_disease_effects.csv) —
the curated per-disease protein NPX shift catalog that powers
[`synthlab.olink.load_disease_effect_catalog`](../synthlab/olink.py).

The goal is to give users **realistic, source-cited** effect sizes for
simulating disease cohorts with
[`synthlab.simulate_olink_npx`](../synthlab/olink.py), without requiring
them to re-mine the literature themselves.

## Schema

| Column              | Type     | Meaning |
| ------------------- | -------- | ------- |
| `disease`           | `str`    | Canonical disease label (e.g. `T2D`, `CAD`, `Alzheimer`, `CKD`, `IBD`, `Cancer_broad`, `BRCA_hereditary`). Match these labels to the `group_assignments` you feed into `OlinkSimConfig`. |
| `protein_uniprot`   | `str`    | UniProt accession or gene symbol. Match these to the `proteins` field of your `OlinkPanelConfig`. |
| `delta_npx`         | `f64`    | **log2 NPX-unit mean shift** of cases vs a demographically-matched baseline. Negative values are allowed (e.g. adiponectin goes down in T2D). |
| `se_delta`          | `f64`    | Standard error of the delta, reflecting between-study / between-cohort variability. Used by `effects_for(noise_sd=...)` to draw Monte-Carlo perturbations. |
| `source`            | `str`    | Human-readable study tag, e.g. `"Sun et al. 2023 UKB-PPP (N=54,219) Nature"`. |
| `doi`               | `str`    | Real DOI (starts with `10.`). **Fabricated DOIs are not allowed** — if you cannot find one for a given protein/disease association, omit the row. |
| `evidence_strength` | `str`    | One of `"strong"` / `"moderate"` / `"weak"`. See the rubric below. |
| `meta` (optional)   | `str`    | Free-form note — e.g. "largest-N meta-estimate", "MR-supported causal locus", or which of several papers the estimate was chosen from. |

Every row **must** provide the first seven columns. `meta` is optional but
strongly encouraged.

## Effect-size units

Deltas are in **log2 NPX units** — the native scale used across all Olink
[Explore](https://olink.com/products/olink-explore) panels. A `delta_npx`
of `1.0` means cases sit at roughly `2x` the median baseline (log2(2) =
1). A `delta_npx` of `-0.5` means cases sit at roughly `0.7x` the baseline.

### Converting from other scales

- **Fold change** (e.g. microarray log2 fold change is already compatible):
  `delta_npx = log2(fold_change)`. So a reported 2-fold change maps to
  `delta_npx = 1.0`; a 4-fold change maps to `delta_npx = 2.0`.
- **Odds ratio** (from logistic GWAS / PRS work): approximately,
  `delta_npx = log2(OR) * (protein_sd)` — but this is often misleading
  because it doesn't translate a case/control association into a plasma
  NPX shift. Prefer studies that directly report case-baseline NPX
  differences.
- **Z-scored deltas** (reported as "effect in SD units"): multiply by
  the protein's baseline NPX SD (typically ~0.5-0.8 NPX units on
  Olink Explore).

### Defensibility sanity check

Keep magnitudes in a range the real assay has been observed to span:

| Scenario                                   | Plausible `delta_npx` |
| ------------------------------------------ | --------------------- |
| CRP in sepsis / active severe infection    | +3.0 to +4.0          |
| CRP in subclinical CAD / metabolic disease | +0.3 to +0.5          |
| GDF15 in advanced CKD or heart failure     | +0.7 to +1.2          |
| NEFL in mild cognitive impairment          | +0.4 to +0.8          |
| NEFL in clinically diagnosed AD            | +0.8 to +1.2          |
| Null hypothesis (no real effect)           | 0.0 +/- 0.3           |

If you're recording a delta > 3.0, double-check the source.

### Standard-error guidance

- If the source reports a 95% CI for the delta, set
  `se_delta = (upper - lower) / (2 * 1.96)`.
- If the source gives only a point estimate and the study is large
  (N >= 10,000), set `se_delta = 0.15` (2x the typical UKB-PPP per-protein
  SE).
- If the study is small (N < 3,000) or the effect is drawn from a single
  paper, set `se_delta = 0.3` (generic conservative default).

## Evidence-strength rubric

| Label       | Criteria |
| ----------- | -------- |
| `strong`    | Replicated in >= 2 large cohorts (N >= 5,000 each), OR supported by Mendelian randomization / pQTL causal inference in a large-N study, OR used clinically today (e.g. CRP for IBD activity). |
| `moderate`  | Single large cohort (N >= 5,000) with clear effect, OR multiple smaller cohorts (N < 5,000) in agreement. |
| `weak`      | Single small study, or mechanistic inference only, or null-hypothesis placeholder (e.g. pre-symptomatic hereditary cancer carriers where no consistent plasma signature exists). |

Weak rows are fine to include, but their downstream use should assume the
noise term dominates — e.g. set `noise_sd=1.0` in
`DiseaseEffectCatalog.effects_for` so the delta is meaningfully perturbed
during Monte-Carlo sampling.

## How to add a new disease / row

Checklist for a PR touching
[`olink_disease_effects.csv`](../synthlab/data/olink_disease_effects.csv):

- [ ] **Source**: I have a published paper (journal or preprint) with a
  real DOI. I've verified the DOI resolves at `https://doi.org/<mydoi>`.
  No synthesized DOIs.
- [ ] **Unit conversion**: I've converted the source's reported effect
  into log2 NPX units per the table above, and the resulting magnitude
  is within the defensibility range.
- [ ] **SE choice**: `se_delta` follows the standard-error guidance
  (explicit 95% CI -> computed; large-N point estimate -> 0.15; otherwise
  0.3).
- [ ] **Evidence tag**: `evidence_strength` is set per the rubric.
- [ ] **No duplicates**: `(disease, protein_uniprot)` is unique within the
  CSV. If multiple papers report an estimate for the same protein +
  disease, take the largest-N meta-estimate and record which paper in the
  `meta` column. Use multiple rows only when you truly want independent
  draws from N(delta, se_delta^2) for a single protein + disease cell.
- [ ] **Disease label**: either matches an existing label in `diseases()`,
  OR is a new canonical short label you've introduced consistently across
  all rows for that disease.
- [ ] **Tests**: I've added a test that loads the new rows via
  `load_disease_effect_catalog()` and asserts the expected disease +
  protein coverage.
- [ ] **Docs**: I've updated the per-disease row range in
  [`README.md`](../README.md) if a new disease was added.

## Example: adding a hypothetical "post-COVID" disease

```csv
post_covid,CRP,0.55,0.20,"Made-up et al. 2025 Made-up Journal",10.0/example.post-covid,moderate,placeholder only
post_covid,IL6,0.70,0.25,"Made-up et al. 2025 Made-up Journal",10.0/example.post-covid,moderate,placeholder only
post_covid,GDF15,0.45,0.25,"Made-up et al. 2025 Made-up Journal",10.0/example.post-covid,moderate,placeholder only
```

*(In a real PR, the DOI would be real — this is for illustration only.)*

## Open questions / deferred work

- **Covariate-adjusted effects**: Currently deltas are marginal (cases vs
  baseline, aggregated across age / sex / ancestry). A future rev could
  ship age- or BMI-conditional deltas, which would plug into an extended
  simulator that accepts per-sample covariate effects.
- **Longitudinal effects**: For incidence-cohort simulations, the delta
  grows over time pre-diagnosis. A future rev could add
  `time_to_dx_years` -> `delta_npx` curves per (disease, protein).
- **Interaction terms**: Some effects depend on comorbidities (e.g. GDF15
  in T2D with CKD is larger than T2D alone). Future rev: an interactions
  CSV that layers on top of the marginal catalog.
