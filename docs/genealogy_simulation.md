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

## Independent genealogy-correlated environmental truth

`synthlab.genealogical_covariance.genealogical_environment_factors` builds an
independent dense covariance oracle directly from explicit descendants of each
branch of every marginal tree. It imports no association-analysis code. For
person `i` and branch `e`, the feature is the fraction of that person's two
declared haplotypes below the branch, multiplied by the square root of the
branch's genomic span fraction times its duration in the requested time band.
Summing outer products of these features gives the environmental covariance.

```python
from synthlab.genealogical_covariance import genealogical_environment_factors

environment = genealogical_environment_factors(
    truth.tree_sequence,
    sample_ids=truth.sample_ids,
    sample_node_pairs=truth.sample_node_pairs,
    lower_time=0,
    upper_time=float('inf'),
)
# Supply these arguments to simulate_quantitative_phenotypes, alongside its
# genotype, fixed-effect, trait-covariance and seed arguments:
phenotype_environment_options = {
    'residual_row_factors': environment.factors,
    'residual_factor_ids': environment.factor_ids,
}
```

Default centering uses all declared people, independently of phenotype
availability. Mean-trace normalization makes the average covariance diagonal
one. Reference-only tips contribute to the genealogy without becoming phenotype
rows. `lower_time` and `upper_time` select explicit branch-time bands in the
tree sequence's known time units; no ancestry labels are required.

Reduced QR of the transposed feature matrix compresses the returned factor
`B` to at most one column per person while preserving `B @ B.T`. Singular
centered covariances need no diagonal jitter or eigenvalue clipping. Factor
coordinates are compressed basis coordinates rather than named ancestral
nodes. Canonical person-ID order and fixed QR signs preserve factor realizations
by exact ID when input rows are permuted, within the same numerical environment.
Changing the people or graph need not preserve realizations. Output order
matches the caller, and arrays are owned and read-only.

The returned covariance is the actual numerical `B @ B.T`. With
`residual_row_factors=sqrt(tau)*B`, the quantitative generator's person
covariance is `I + tau*(B @ B.T)`, tensor-multiplied by its actual trait sampling
covariance. This models genealogy-correlated environmental confounding; it does
not imply a causal genetic effect or heritability. Retain the covariance
provenance and scientific signature separately from the transport manifest.

Defaults bound this truth reference to2048 people,1million branch columns,
1billion estimated scalar operations and256MiB of principal array allocations.
Traversal, QR and covariance allocation are preflighted. Resident tree tables,
Python metadata and library allocator overhead are outside these array bounds.
This implementation establishes small-panel truth, with no demonstrated
biobank construction speed. Empty time bands return zero factors only when
`normalization='none'`; attempting to normalize them raises an error.

## Verified external association bundles

`synthlab.association_bundle.write_association_bundle` verifies the genealogy's
logical signature, original person-to-individual join, exact site coordinates,
literal allele order and diploid calls against the tree sequence. It then calls
the quantitative generator directly on those verified genotypes; phenotype
options cannot replace person, variant, allele or genotype join arguments.
Caller-supplied variant IDs follow site order, and trait units follow trait order.

The writer creates a fresh directory, preserves existing paths, writes the
original `truth.trees` plus uncompressed numeric `.npy` arrays with pickle
disabled, and publishes `manifest.json` last. Failure removes only its fresh
directory. The manifest records exact IDs, axes, phase/coordinate/time
conventions, recipes, logical signatures, source revision, and SHA256/bytes/
shape/dtype for every payload. Inputs, complete biological/environmental truth
and observed outcomes remain separate. Preserve the returned manifest SHA256
separately as the consumer's trusted transport pin; checksums do not establish
an untrusted producer's authenticity.

The standalone `scripts/generate_association_bundle.py` entry point works in
the minimal simulation environment without importing the unrelated EHR stack.
Save this example as `recipe.json`:

```json
{
  "genealogy": {
    "sample_ids": ["a", "b", "c", "d"], "sequence_length": 1000,
    "seed": 91, "population_size": 100, "mutation_rate": 0.0001,
    "dtwf_generations": 5
  },
  "reference": "synthetic ancestral-reference coordinates", "contig": "simulation-1",
  "trait_units": ["expression units", "height units"],
  "phenotypes": {
    "trait_ids": ["expression", "height"], "effects": [],
    "covariate_ids": ["intercept"], "covariates": [[1], [1], [1], [1]],
    "covariate_effects": [[5, -2]], "residual_covariance": [[1, 0.2], [0.2, 2]],
    "seed": 113
  }
}
```

```bash
.venv-simulation/bin/python scripts/generate_association_bundle.py \
  --recipe recipe.json --destination new-association-bundle \
  --source-revision "$(git rev-parse HEAD)"
```

The JSON receipt on stdout reports the manifest path, SHA256 and total file
bytes. This initial CLI uses the single-deme genealogy configuration and assigns
variant IDs `site-<zero-based-position>`. Mechanism lists contain the dataclass
fields documented above, using those exact IDs. The programmatic writer also
accepts validated custom-demography genealogy output. No seed or locus is chosen
by inspecting association performance.

The format is an experimental version0 quantitative true-SNP interchange, not
a real-cohort importer or a calibrated association result. Defaults bound output
to256MiB, the manifest to16MiB and genotype verification to1million person-site
comparisons. These output limits do not bound parser/simulator process RSS.
Run simulation in an external environment when benchmarking CuEvo, and pass only
the data files and pinned receipt across that process boundary.

For a small numerical test environment without the unrelated EHR/LLM stack:

```bash
uv venv --python 3.12 .venv-simulation
uv pip sync --python .venv-simulation/bin/python --require-hashes requirements/simulation-test.lock
.venv-simulation/bin/python -m pytest tests/test_genealogy.py tests/test_genealogical_covariance.py tests/test_genotype_phenotypes.py tests/test_association_bundle.py -q
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
