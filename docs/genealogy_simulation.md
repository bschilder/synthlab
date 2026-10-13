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
limit. The genealogy adapter does not generate causal phenotypes, validate
inferred ARGs, or establish association calibration. The separate quantitative
generator below supplies planted phenotype mechanisms.

## Quantitative phenotype truth

`synthlab.genotype_phenotypes` consumes complete diploid integer allele indices
from this adapter or another validated source, including HAPNEST. It requires
exact ordered person, variant, allele, trait and covariate identifiers. Index0
has the meaning declared by the caller's allele vocabulary; all allele-specific
effects count exact copies, including at multiallelic sites. Missing or invalid
truth genotypes are rejected; contaminated observed calls belong in a separate
artifact.

```python
import numpy as np
from synthlab.genotype_phenotypes import AlleleEffect, simulate_quantitative_phenotypes

phenotypes = simulate_quantitative_phenotypes(
    sample_ids=('person-a', 'person-b', 'person-c'),
    variant_ids=('site-a',), alleles=(('A', 'C'),),
    genotypes=np.array([[[0, 0]], [[0, 1]], [[1, 1]]], dtype=np.int8),
    trait_ids=('expression', 'height'),
    effects=(AlleleEffect('site-a', 1, (2., -1.)),),
    covariate_ids=('intercept',), covariates=np.ones((3, 1)),
    covariate_effects=np.array([[5., -2.]]),
    residual_covariance=np.array([[1., .2], [.2, 2.]]), seed=91,
    observed_probabilities=np.array([[1., .5], [.8, 1.], [.5, .5]]),
    missingness_description='MCAR: predeclared person/trait observation probabilities',
)
print(phenotypes.genetic_means)  # [[0, 0], [2, -1], [4, -2]]
print(phenotypes.observed_values)  # NaN only at cells marked unobserved
```

No intercept, centering, allele frequency or scaling is inferred. `AlleleEffect`
is a signed per-copy coefficient vector in trait order. Other mechanisms have
separate inputs and returned means:

- `DominanceEffect(variant_id, allele_index, effects)` applies when exactly one
  homologue carries the target allele, including at a multiallelic site.
- `InteractionEffect(first_variant_id, first_allele_index, second_variant_id,
  second_allele_index, effects, first_center=0., second_center=0.)` multiplies
  the two dosages after subtracting caller-fixed centers. This permits a balanced
  pure-epistasis scenario without marginal linear effects.
- `ContextEffect(variant_id, allele_index, context_id, effects,
  context_center=0.)` multiplies dosage by a continuous context deviation.
  Contexts have exact column identifiers; no ancestry categories are required.
- `CompoundHetEffect(first_variant_id, first_allele_index, second_variant_id,
  second_allele_index, effects)` requires two distinct nonreference alleles,
  each carried once, on opposite homologues. The same multiallelic site is
  permitted. Cis configurations and target-allele homozygotes do not qualify.

Genetic means, fixed covariate means, supplied `environment_means`, independent
noise, structured environmental noise, complete outcomes and recorded cells
remain separate. Perfectly linked causal features keep their distinct planted
coefficients; this does not imply they can be uniquely localized by an analysis.
Optional `residual_row_factors` A and their exact `residual_factor_ids` generate
shared environmental noise. In column-major person-by-trait order its full
covariance is `sampling_covariance ⊗ (I + A @ A.T)`. Households or other shared
environments can therefore be simulated separately from the genetic mechanism.

Covariance validation and factorization use a correlation scale, then restore
caller units. A zero variance requires exactly zero covariances with that trait;
negative variances and invalid correlations are rejected. Positive eigenvalues
are retained without a relative rank cutoff. At a numerically singular boundary,
negative eigenvalues within the declared roundoff tolerance are omitted. The
requested covariance, actual sampling covariance, rank, tolerance and maximum
difference are returned; no jitter is added.

Independent, shared-factor and observation-mask RNG streams are separate.
Sorting exact person, trait and factor IDs internally makes permutations of
those axes preserve individual draws, while output order stays as supplied.
Changing the cohort or trait set need not preserve realizations. Changing only
observation probabilities preserves complete outcomes. To simulate MAR or MNAR,
generate complete outcomes, derive declared probabilities from measured inputs
or complete truth respectively, then repeat with the same seed and those
probabilities. Retain the formula in `missingness_description`. This generates
informative missingness; it does not correct its bias.

Complete-case removal and imputation are not performed. People with no recorded
traits remain in the output, and `observed_values` uses NaN for unrecorded cells.
Arrays are owned and read-only. A deterministic scientific signature covers
inputs, mechanism records, numerical choices and complete/observed outputs;
transport artifacts still need separate checksums.

The default preflight bounds256 traits,1billion estimated element visits and
256MiB of principal array allocations, including wide coefficient matrices.
Caller-resident arrays, metadata and numerical-library allocator overhead are
excluded. This reference implementation has no demonstrated biobank throughput.
Binary/count heads, genotype contamination and association calibration remain
separate work.

For a small numerical test environment without the unrelated EHR/LLM stack:

```bash
uv venv --python 3.12 .venv-simulation
uv pip sync --python .venv-simulation/bin/python --require-hashes requirements/simulation-test.lock
.venv-simulation/bin/python -m pytest tests/test_genealogy.py tests/test_genotype_phenotypes.py -q
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
