"""Traditional doubly robust DiD kernels for panel and repeated cross sections.

Both functions return ``(ATT, observation_influence_function)``. Covariates
retain Python's automatic intercept; supply that augmented design explicitly
when comparing with R's low-level functions. ``trim_level`` trims controls
whose fitted propensity is at or above the threshold (default 0.995).
"""
from ._estimators import drdid_panel, drdid_rc

__all__ = ["drdid_panel", "drdid_rc"]
