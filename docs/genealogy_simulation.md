# True genealogies and phased genotype simulation

`synthlab.genealogy` provides an optional msprime adapter for independent
simulation benchmarks. It retains true marginal trees, explicit diploid
sample-node pairs and complete phased literal-allele genotypes. It complements
HAPNEST's realistic synthetic haplotypes; HAPNEST output is not automatically a
known ancestral recombination graph.

The optional simulation engine needs Python3.11 or newer:

```bash
pip install 'synthlab[simulation]'
```

```python
from synthlab.genealogy import GenealogySimConfig, simulate_genealogy

truth = simulate_genealogy(GenealogySimConfig(
    sample_ids=('person-a', 'person-b', 'person-c'),
    sequence_length=100_000,
    seed=91,
    population_size=10_000,
    dtwf_generations=50,
))

print(truth.tree_sequence.num_trees)
print(truth.genotypes.shape)  # [person, site, homologue]
print(truth.sample_node_pairs)
print(truth.scientific_signature)
```

Recent discrete-time Wright–Fisher simulation precedes a standard coalescent.
`dtwf_generations=0` selects only the latter. Independent JC69 mutation draws
produce literal A/C/G/T alleles, including recurrent and multiallelic sites.
Coordinates are zero-based integers and times are generations. Allele index0 is
the simulated ancestral state, explicitly chosen as the synthetic REF; this is
not a rule for a real human reference assembly. Empty zero-mutation panels are
valid. The adapter requests true marginal genealogies, not a complete record of
every recombination event.

For a custom `msprime.Demography`, supply `sample_demes` as one exact population
name per person, in the same order as `sample_ids`. The adapter copies the
demography and extracts two sample nodes through each generated individual ID.
Deme names describe the generating mechanism; they are not mandatory ancestry
categories for a downstream association model.

Named random streams separate genealogy, mutation, phenotype, missingness,
genotype errors and phase errors. This adapter uses only the first two and
records all derived seeds. Changing the mutation rate leaves the simulated
ancestry unchanged. Genotypes are complete truth; observation errors and missing
calls must be retained separately rather than overwriting it.

The scientific signature hashes canonical table content and the recipe,
including software versions. It excludes runtime provenance timestamps and file
UUIDs. It is not a transport checksum; a saved artifact needs its own SHA256.
Cross-version bit identity is not promised. Output arrays are owned and read-only.

Default limits are2048 people,100,000 sites and256MiB of conservatively estimated
principal output arrays. These limits cannot constrain msprime's internal event
allocations before it returns. Large cases need separate process time/memory
controls on remote compute. No people or sites are silently dropped to fit a
limit. The adapter does not generate causal phenotypes, validate inferred ARGs,
or establish association calibration.

For a small numerical test environment without the unrelated EHR/LLM stack:

```bash
uv venv --python 3.12 .venv-simulation
uv pip sync --python .venv-simulation/bin/python --require-hashes requirements/simulation-test.lock
.venv-simulation/bin/python -m pytest tests/test_genealogy.py -q
```

These tests load the optional numerical module directly. The repository's
existing full-package CI gate is preserved. The exact numerical test lock uses
NumPy2.5.4 and requires Python3.12 or newer; CI tests Python3.12 and3.13.

msprime1.4.4 is GPL-3.0-or-later and remains an external simulation dependency;
none of its implementation is copied into synthlab. When used to benchmark a
permissive association library, run this generator in a separate process and
exchange verified data artifacts. Do not import or link the simulation engine
into that library's runtime.

Sources: [msprime ancestry models](https://tskit.dev/msprime/docs/stable/ancestry.html),
[release metadata and license](https://pypi.org/project/msprime/1.4.4/).
