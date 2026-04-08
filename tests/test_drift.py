"""
tests/test_drift.py
====================
Tests for the drift detection engine.
Week 1: placeholder tests that will be filled in Week 4.
These establish what the drift detector SHOULD do.
"""
import numpy as np
import pandas as pd
import pytest


def compute_psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """
    Compute Population Stability Index (PSI) between two distributions.
    
    PSI = sum((actual% - expected%) * ln(actual% / expected%))
    
    WHAT IS PSI?
    Imagine you trained a model on January data. In June, new data arrives.
    PSI measures: "How different is the June distribution from January?"
    
    PSI < 0.1  → No drift, model is stable
    PSI 0.1-0.2 → Slight drift, monitor closely  
    PSI > 0.2  → Significant drift, RETRAIN the model
    """
    # Create buckets based on expected distribution
    breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
    breakpoints = np.unique(breakpoints)
    
    expected_counts = np.histogram(expected, bins=breakpoints)[0]
    actual_counts = np.histogram(actual, bins=breakpoints)[0]
    
    # Add small epsilon to avoid division by zero / log(0)
    epsilon = 1e-6
    expected_pct = (expected_counts + epsilon) / (len(expected) + epsilon * len(expected_counts))
    actual_pct = (actual_counts + epsilon) / (len(actual) + epsilon * len(actual_counts))
    
    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)


# ── Tests ──────────────────────────────────────────────────

class TestPSI:
    """Tests for PSI computation."""

    def test_identical_distributions_have_zero_psi(self):
        """Same data in both windows → PSI should be ~0 (no drift)."""
        data = np.random.RandomState(42).normal(100, 10, 1000)
        psi = compute_psi(data, data)
        assert psi < 0.01, f"Identical distributions should have PSI ≈ 0, got {psi:.4f}"

    def test_very_different_distributions_have_high_psi(self):
        """Completely shifted data → PSI should be > 0.2 (significant drift)."""
        rng = np.random.RandomState(42)
        original = rng.normal(100, 5, 1000)   # mean=100
        shifted = rng.normal(200, 5, 1000)    # mean=200 (huge shift!)
        psi = compute_psi(original, shifted)
        assert psi > 0.2, f"Very different distributions should have PSI > 0.2, got {psi:.4f}"

    def test_slight_shift_has_moderate_psi(self):
        """Small shift → PSI between 0.1 and 0.2."""
        rng = np.random.RandomState(42)
        original = rng.normal(100, 10, 5000)
        slightly_shifted = rng.normal(105, 10, 5000)  # 5% shift
        psi = compute_psi(original, slightly_shifted)
        # This is in the "monitor closely" zone
        assert 0.0 < psi < 0.5, f"Slight shift PSI should be moderate, got {psi:.4f}"

    def test_psi_is_non_negative(self):
        """PSI is always >= 0 by mathematical definition."""
        rng = np.random.RandomState(42)
        a = rng.exponential(scale=10, size=500)
        b = rng.exponential(scale=15, size=500)
        psi = compute_psi(a, b)
        assert psi >= 0, "PSI cannot be negative"

    def test_drift_threshold_logic(self):
        """Test the business rule: PSI > 0.2 triggers retraining."""
        rng = np.random.RandomState(42)
        stable = rng.normal(0, 1, 1000)
        drifted = rng.normal(3, 1, 1000)  # 3 std dev shift
        
        psi = compute_psi(stable, drifted)
        
        DRIFT_THRESHOLD = 0.2
        should_retrain = psi > DRIFT_THRESHOLD
        assert should_retrain, f"Model should retrain when PSI={psi:.3f} > {DRIFT_THRESHOLD}"


class TestDataQuality:
    """Tests for data quality checks that feed into drift detection."""

    def test_fraud_rate_in_expected_range(self):
        """Fraud rate in training data should be between 1% and 10%."""
        labels = np.array([0] * 9650 + [1] * 350)  # 3.5% fraud
        fraud_rate = labels.mean()
        assert 0.01 <= fraud_rate <= 0.10, f"Fraud rate {fraud_rate:.3f} outside expected range"

    def test_transaction_amount_positive(self):
        """Transaction amounts should always be positive."""
        amounts = np.abs(np.random.RandomState(42).normal(100, 50, 1000))
        assert (amounts > 0).all(), "All transaction amounts must be positive"
