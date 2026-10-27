# Alignment with R DRDID and Python CSDID

The reviewed R baseline is DRDID 1.4.0 at
[`eaac3b49`](https://github.com/pedrohcgs/DRDID/commit/eaac3b49cd225d0d7706c633858555a5f204e21e).
The prior Python baseline is 1.1.6 / `75cac04e`. The six existing Python kernels
are traditional DR, normalized IPW and regression, each for panel and repeated
cross-section data. This change preserves their tuple return and methods.

The correction addresses observed failures: omitted weights were converted to an
array before the `None` check in three functions; RC DR added rather than
subtracted pre-period nuisance contributions; propensity trimming was missing;
and the propensity Hessian used an earlier IRLS covariance rather than the final
capped probabilities. Inputs and weighted fit designs are now checked before
fitting. All six methods share those checks. Python's existing automatic
intercept convention is retained and explicitly documented.

## Evidence

`tests/r_ref` contains common seeded inputs, 80 fixed configurations, complete R
ATT/influence/SE/interval references and a manifest of hashes. The grid covers
all six methods, omitted/unit/varying/zero weights, intercept-only and covariate
models, controls above trimming thresholds, two thresholds, and outcome unit
changes by factors of `1e-10` and `1e10`. Comparisons use `atol=1e-7, rtol=1e-6`
in original outcome units. Full influence vectors and nonfinite behavior are
checked, not only point estimates. The input/unit conversion is fixed before
comparison. Ordinary tests need no R installation.

The independent contract tests include closed forms, empirical observation-weight
derivatives with nuisance refitting, weight scaling, zero-mass rows and malformed
inputs. They can expose mistakes even when two implementations share a bug.
The exporter rejects dirty/untracked R checkouts, rebuilds native code and binds
reference metadata to the selected commit, the complete tracked source tree and
exact input/exporter hashes.
The current-R CI job also regenerates references from the official R default
branch. Changes to any tracked R repository file require review,
even when this fixed numerical grid passes. A passing grid does not establish
parity for methods outside the six listed above.

The workflow also installs the proposed DRDID and the current CSDID checkout
together, compares the actual downstream dispatch on 72 common-input cases and
runs CSDID's regression suite. CI runs on Python repository pushes and pull
requests. An R-only push does not start it; new R changes are checked on the next
Python run and still require a reviewed port rather than automatic code changes.

Local replay from the repository root, using a clean R checkout with dependencies
and `pkgload`, `jsonlite`, `digest` installed:

```sh
python -m pip install pytest .
python -m pytest tests -q
Rscript tests/r_ref/export.R /path/to/DRDID /tmp/new-r-reference.json
DRDID_R_REFERENCE=/tmp/new-r-reference.json python -m pytest tests/test_r_parity.py -q
python scripts/check_r_source.py /path/to/DRDID
```

Regenerating accepted references is a review action: preserve the existing input
CSVs and case definitions, inspect the R source/test diff, port applicable
assertions, diagnose numerical changes, and update provenance together. Do not
widen tolerances or remove cases to make an update pass. `generate_inputs.py` is
the original deterministic input recipe, not part of routine reference updates.

## Coordinated delivery

Python CSDID currently keeps local DR/IPW kernels with additional performance
caching, while importing regression kernels from `drdid.reg_did`. Therefore a
change in only one repository cannot certify both packages. The six CSDID routes
must continue to be compared against R did/DRDID, and standalone DRDID against R
DRDID. After upstream review and a new PyPI release, CSDID can declare a tested
minimum dependency version and consume the shared implementation without losing
its cached-fit semantics. That follow-up requires its complete installed-package
and group-time/aggregation tests. No unreleased dependency version is required
by this PR, and no PyPI release is performed by these validation commands.

CI checks should be required in upstream branch protection. The contributor
cannot change upstream repository rules; adding workflow files alone does not
make those checks mandatory. The existing PyPI publication workflow and version
are unchanged in this correction.
