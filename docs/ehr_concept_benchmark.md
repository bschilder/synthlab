# Paired EHR concept benchmark

```python
from synthlab.ehr_benchmark import generate_ehr_benchmark
rows = generate_ehr_benchmark(patients_per_condition=120, seed=42)
```

Each of 16 conditions has balanced positive and negative patients. Every patient
has 1d/1mo/1y documents and explicit/implicit diagnosis versions, sharing dated
clinical evidence and one patient-level split. Month/year windows use calendar
arithmetic, with clamping for month ends and leap days. Labels, concept IDs,
negative types, dates, patient IDs, and split assignments are metadata. The text
field is the sole encoder input. Pass `extra_aliases={concept_id: [names...]}`
from an audited ontology snapshot to extend the implicit-name leakage audit.

Healthy, lookalike, negation, and family-history controls support tests of whether
similarity responds to affirmed disease evidence rather than mentions alone.
The explicit documents preserve negative/family-history diagnosis statements;
the implicit documents omit diagnosis statements and redact remaining aliases.
Implicit does not mean deliberately obscuring recognizable drugs or symptoms.

This generator produces controlled templates, not diverse Synthea notes or
clinically validated cases. Disease fingerprints and repeated language can make
linear probes easy; any positive result needs independent replication. Split by
patient, never document. The default train/validation/test split is 60/20/20,
stratified by condition and case status with rounding. See
https://github.com/standardmodelbio/BioDocs/issues/130 for vector benchmarks and
the separate private HF EHR dataset with texts, labels, and vectors.
