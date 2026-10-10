"""Reference countermodels for M7 frozen SERS test cases; not new physics.

This adapter is optional and independent of generic ScientificProgramDraft.
"""
from __future__ import annotations
from math import exp, expm1, isfinite


def normalized_time_average_variance(T: float, tau: float) -> float:
    """For C(t)=sigma^2 exp(-|t|/tau), Var(mean_T)/sigma^2.

    Stable dimensionless expression 2(x -1 + exp(-x))/x^2, x=T/tau.
    """
    if T <= 0 or tau <= 0 or not isfinite(T) or not isfinite(tau):
        raise ValueError("finite positive T and tau required")
    x = T / tau
    if x < 1e-3:
        # Taylor expansion about zero avoids catastrophic cancellation.
        return 1 - x / 3 + x*x/12 - x*x*x/60 + x**4/360
    return 2 * (x + expm1(-x)) / (x*x)


def two_state_covariance(pi0: float, pi1: float, delta_ai: float, delta_aj: float,
                         lag: float, tau: float) -> float:
    """Continuous-time two-state stationary Markov intensity covariance."""
    if not 0 <= pi0 <= 1 or not 0 <= pi1 <= 1 or abs(pi0 + pi1 - 1) > 1e-8:
        raise ValueError("two stationary probabilities must sum to one")
    if lag < 0 or tau <= 0:
        raise ValueError("nonnegative lag; positive tau")
    return pi0*pi1*delta_ai*delta_aj*exp(-lag/tau)


def proportional_emission_ratio(vectors: list[list[float]], tol: float = 1e-9) -> bool:
    """True if all nonzero 2-band spectral state vectors are proportional.

    When true, a noiseless pooled band ratio is invariant to occupancy/exchange
    weighting; temporal covariance can still vary. No assertion for zero vectors.
    """
    if not vectors or any(len(v) != 2 or min(v) <= 0 for v in vectors):
        raise ValueError("positive two-band vectors required")
    ref = vectors[0][0] / vectors[0][1]
    return all(abs(v[0]/v[1] - ref) <= tol*max(1,abs(ref)) for v in vectors)


def frozen_sers_challenges() -> dict:
    xs = [0.1, 1.0, 10.0]
    return {
        "schema_version": "m7-frozen-sers-null-examples-v1",
        "scientific_truth_authority": False,
        "cases": [
            {
                "case_id": "P2_RATIO_INVARIANCE",
                "statement": "For proportional state-specific band vectors, any positive occupancy weighting leaves the noiseless pooled ratio unchanged; kinetics alone is not sufficient to predict a mean ratio change.",
                "proportional_vectors": [[2., 1.], [6., 3.]],
                "proportional_invariance": proportional_emission_ratio([[2.,1.],[6.,3.]]),
                "counterexample_nonproportional": [[2.,1.],[3.,3.]],
                "nonproportional_invariance": proportional_emission_ratio([[2.,1.],[3.,3.]]),
                "research_next_step": "Use independent occupancy/residence times + cross-band covariance and variance-vs-window; first calibrate state-specific emission vectors",
            },
            {
                "case_id": "P2_MARKOV_TEMPORAL_NULL",
                "statement": "A stationary two-state Markov process already predicts exponential intensity covariance and finite-window scaling; observing these alone is not novel evidence for a new mechanism.",
                "T_over_tau": xs,
                "Var_time_mean_over_sigma2": [normalized_time_average_variance(x,1) for x in xs],
                "reference_cross_band_cov_lag0": two_state_covariance(.4,.6,2.,-1.,0.,1.),
                "research_next_step": "Compare stationary Markov, hidden-state Markov mixtures, semi-Markov and photophysical models on independent dwell survival plus higher-order held-out time statistics",
            },
            {
                "case_id": "P2_HIDDEN_STATE_NONIDENTIFIABILITY",
                "statement": "A non-exponential aggregate dwell/survival curve need not imply fundamental non-Markov dynamics: an unobserved mixture of exponential conditional dwell times can yield nonexponential aggregate survival.",
                "mixture_survival_t1": .5*exp(-1/.5)+.5*exp(-1/3),
                "single_exponential_at_weighted_rate_t1": exp(-(.5/.5+.5/3)),
                "research_next_step": "Measure independently resolved states; require predictive superiority over explicit hidden Markov state-mixture null, not just an exponential-vs-nonexponential test",
            },
            {
                "case_id": "P1_PHOTOPHYSICS_CONFOUNDING",
                "statement": "Power or duty-cycle dependence alone does not identify excitation-induced molecular state dynamics: heating, bleaching, optical-transfer drift and orientation changes are competing explanations.",
                "research_next_step": "Factorially perturb illumination and independent thermal/chemical-state controls; hold out preparations and excitation wavelengths",
            },
        ],
    }
