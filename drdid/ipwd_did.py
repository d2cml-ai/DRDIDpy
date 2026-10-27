"""Normalized IPW DiD kernels for panel and repeated cross sections.

Both functions return ``(ATT, observation_influence_function)``. Covariates
retain Python's automatic intercept; supply that augmented design explicitly
when comparing with R's low-level functions. ``trim_level`` trims controls
whose fitted propensity is at or above the threshold (default 0.995).
"""
from ._estimators import std_ipw_did_panel, std_ipw_did_rc

__all__ = ["std_ipw_did_panel", "std_ipw_did_rc"]
