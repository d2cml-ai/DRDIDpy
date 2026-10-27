# DRDID in Python

Traditional doubly robust, normalized inverse probability weighted, and outcome
regression difference-in-differences estimators for panel data and repeated cross
sections, following [R DRDID](https://github.com/pedrohcgs/DRDID).

## Installation

```sh
pip install drdid
```

For the development version:

```sh
pip install git+https://github.com/d2cml-ai/DRDIDpy.git
```

## Estimators and alignment

The existing functions return `(att, influence)`, with one influence contribution
per observation:

| Module | Panel | Repeated cross sections |
|---|---|---|
| `drdid.drdid` | `drdid_panel` | `drdid_rc` |
| `drdid.ipwd_did` | `std_ipw_did_panel` | `std_ipw_did_rc` |
| `drdid.reg_did` | `reg_did_panel` | `reg_did_rc` |

`covariates=None` fits an intercept-only model. Python retains its historical
convention of adding an intercept unless the first supplied column is all ones.
R's low-level functions use exactly the supplied design: include that same
intercept when comparing. Weights default to ones and are normalized to mean one.
DR and IPW accept `trim_level=0.995`; only controls at or above the fitted-score
threshold are trimmed, after capping scores at `1 - 1e-6`, matching R.

```python
from drdid.drdid import drdid_panel

att, influence = drdid_panel(y1, y0, treated, covariates=X, i_weights=weights)
```

Comparisons require the same sample, outcome, design, weights, estimator and
trimming. These DR functions correspond to R's **traditional** estimators. R's
high-level `drdid()` defaults to improved/IPT estimation; select `estMethod="trad"`
when comparing. The IPW functions here are normalized (Hajek), not R's
nonnormalized IPW variant. Improved/IPT, alternative RC estimators, TWFE, R's
formula preprocessing and refit bootstrap API are not implemented here.

This package returns the ATT and influence function rather than an inference
object. R's analytical standard error can be reconstructed as
`np.linalg.norm(influence - influence.mean()) / len(influence)`; its pointwise
95% limits are `att +/- 1.96 * se`. CSDID adds its own group-time preprocessing,
aggregation, overlap checks and inference layers.

See [the validation and release notes](docs/r-csdid-alignment.md) for the tested
R source revision, regression evidence and the CSDID integration sequence.
