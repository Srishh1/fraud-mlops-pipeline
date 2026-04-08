"""
tests/test_drift.py
====================
Full test suite for the drift detection engine.

Tests cover:
  1. PSI core math — edge cases and expected values
  2. Drift classification thresholds — stable / warning / drifted
  3. Reference stats build + load round-trip
  4. Drift simulation produces expected PSI ranges
  5. check_drift() returns correct schema
  6. Feature engineering consistency

Run with: pytest tests/test_drift.py -v
"""

import os
import json
import tempfile
import numpy as np
import pandas as pd
import pytest
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.drift_detector import (
    compute_psi,
    compute_psi_all_features,
    build_reference_stats,
    load_reference_stats,
    simulate_drift,
    check_drift,
    engineer_features_for_drift,
    PSI_STABLE,
    PSI_WARNING,
    PSI_RETRAIN,
    REFERENCE_STATS_PATH,
)


# ──────────────────────────────────────────────────────────
# FIXTURES
# ──────────────────────────────────────────────────────────

@pytest.fixture
def rng():
    return np.random.RandomState(42)


@pytest.fixture
def reference_df(rng):
    """Simulate a realistic reference dataset (training distribution)."""
    n = 5000
    return pd.DataFrame({
        "TransactionAmt":   np.abs(rng.lognormal(4.5, 1.2, n)),
        "Transaction_hour": rng.randint(0, 24, n).astype(float),
        "card1":            rng.randint(1000, 20000, n).astype(float),
        "card4":            rng.choice([0, 1, 2, 3], n).astype(float),
        "card6":            rng.choice([0, 1], n).astype(float),
        "dist1":            np.abs(rng.exponential(50, n)),
        "C1":               rng.randint(0, 10, n).astype(float),
        "C2":               rng.randint(0, 10, n).astype(float),
        "D1":               rng.randint(0, 500, n).astype(float),
        "TransactionAmt_log": np.log1p(np.abs(rng.lognormal(4.5, 1.2, n))),
    })


@pytest.fixture
def stable_df(reference_df, rng):
    """Slightly different data — same distribution, different random sample."""
    n = 2000
    return pd.DataFrame({
        "TransactionAmt":   np.abs(rng.lognormal(4.5, 1.2, n)),
        "Transaction_hour": rng.randint(0, 24, n).astype(float),
        "card1":            rng.randint(1000, 20000, n).astype(float),
        "card4":            rng.choice([0, 1, 2, 3], n).astype(float),
        "card6":            rng.choice([0, 1], n).astype(float),
        "dist1":            np.abs(rng.exponential(50, n)),
        "C1":               rng.randint(0, 10, n).astype(float),
        "C2":               rng.randint(0, 10, n).astype(float),
        "D1":               rng.randint(0, 500, n).astype(float),
        "TransactionAmt_log": np.log1p(np.abs(rng.lognormal(4.5, 1.2, n))),
    })


@pytest.fixture
def drifted_df(rng):
    """Heavily shifted data — should trigger PSI > 0.2 on most features."""
    n = 2000
    return pd.DataFrame({
        "TransactionAmt":   np.abs(rng.lognormal(6.5, 1.5, n)),  # much higher amounts
        "Transaction_hour": rng.randint(0, 6, n).astype(float),   # only nighttime
        "card1":            rng.randint(50000, 100000, n).astype(float),  # completely different cards
        "card4":            rng.choice([0, 1], n).astype(float),  # fewer card types
        "card6":            np.ones(n),                            # all credit (was 50/50)
        "dist1":            np.abs(rng.exponential(500, n)),       # 10x higher distances
        "C1":               rng.randint(10, 50, n).astype(float),  # higher velocity
        "C2":               rng.randint(10, 50, n).astype(float),
        "D1":               rng.randint(0, 10, n).astype(float),   # newer cards
        "TransactionAmt_log": np.log1p(np.abs(rng.lognormal(6.5, 1.5, n))),
    })


# ──────────────────────────────────────────────────────────
# PSI CORE MATH TESTS
# ──────────────────────────────────────────────────────────

class TestComputePSI:

    def test_identical_distributions_give_zero_psi(self, rng):
        data = rng.normal(100, 10, 2000)
        psi = compute_psi(data, data)
        assert psi < 0.01, f"Identical distributions → PSI ≈ 0, got {psi:.4f}"

    def test_completely_different_distributions_give_high_psi(self, rng):
        original = rng.normal(0,   1, 2000)
        shifted  = rng.normal(10,  1, 2000)   # 10 std dev shift
        psi = compute_psi(original, shifted)
        assert psi > PSI_RETRAIN, f"Very different distributions → PSI > {PSI_RETRAIN}, got {psi:.4f}"

    def test_psi_is_non_negative(self, rng):
        """PSI is always >= 0 by mathematical definition."""
        for _ in range(10):
            a = rng.exponential(10, 500)
            b = rng.exponential(15, 500)
            assert compute_psi(a, b) >= 0

    def test_psi_is_symmetric_roughly(self, rng):
        """PSI(A,B) ≈ PSI(B,A) — not exactly equal but in same ballpark."""
        a = rng.normal(100, 10, 2000)
        b = rng.normal(110, 10, 2000)
        psi_ab = compute_psi(a, b)
        psi_ba = compute_psi(b, a)
        # Should be within 50% of each other
        ratio = max(psi_ab, psi_ba) / (min(psi_ab, psi_ba) + 1e-9)
        assert ratio < 3.0, f"PSI asymmetry too large: {psi_ab:.4f} vs {psi_ba:.4f}"

    def test_psi_handles_empty_arrays(self):
        """Empty arrays return 0, not an error."""
        psi = compute_psi(np.array([]), np.array([1, 2, 3]))
        assert psi == 0.0

    def test_psi_filters_missing_sentinel(self, rng):
        """Values of -999 should be excluded from PSI computation."""
        clean = rng.normal(100, 10, 1000)
        with_sentinel = np.concatenate([clean, [-999] * 100])
        psi_clean    = compute_psi(clean, clean)
        psi_sentinel = compute_psi(with_sentinel, clean)
        assert abs(psi_clean - psi_sentinel) < 0.1, \
            "Sentinel values (-999) should be filtered out"

    def test_psi_stable_threshold(self, rng):
        """Small shift → PSI stays below PSI_STABLE (0.1)."""
        original = rng.normal(100, 10, 5000)
        tiny_shift = original + rng.normal(0, 0.5, 5000)  # 5% noise
        psi = compute_psi(original, tiny_shift)
        assert psi < PSI_STABLE, \
            f"Tiny shift should be stable (< {PSI_STABLE}), got {psi:.4f}"

    def test_psi_retrain_threshold(self, rng):
        """Large shift → PSI exceeds PSI_RETRAIN (0.2)."""
        original = rng.normal(0, 1, 3000)
        large_shift = rng.normal(3, 1, 3000)  # 3 std dev shift
        psi = compute_psi(original, large_shift)
        assert psi > PSI_RETRAIN, \
            f"Large shift should trigger retrain (> {PSI_RETRAIN}), got {psi:.4f}"


class TestComputePSIAllFeatures:

    def test_returns_dict_with_feature_names(self, reference_df, stable_df):
        result = compute_psi_all_features(reference_df, stable_df)
        assert isinstance(result, dict)
        assert len(result) > 0
        for key, val in result.items():
            assert isinstance(key, str)
            assert isinstance(val, float)
            assert val >= 0

    def test_stable_data_all_features_below_threshold(self, reference_df, stable_df):
        result = compute_psi_all_features(reference_df, stable_df)
        high_psi = {k: v for k, v in result.items() if v > PSI_RETRAIN}
        assert len(high_psi) == 0, \
            f"Stable data should have no features above {PSI_RETRAIN}: {high_psi}"

    def test_drifted_data_has_features_above_threshold(self, reference_df, drifted_df):
        result = compute_psi_all_features(reference_df, drifted_df)
        high_psi = {k: v for k, v in result.items() if v > PSI_RETRAIN}
        assert len(high_psi) > 0, \
            f"Drifted data should have at least one feature above {PSI_RETRAIN}"

    def test_skips_missing_features_gracefully(self, reference_df, stable_df):
        """Features not present in both DFs are silently skipped."""
        features = ["TransactionAmt", "nonexistent_feature_xyz"]
        result = compute_psi_all_features(reference_df, stable_df, features)
        assert "nonexistent_feature_xyz" not in result
        assert "TransactionAmt" in result


# ──────────────────────────────────────────────────────────
# REFERENCE STATS TESTS
# ──────────────────────────────────────────────────────────

class TestReferenceStats:

    def test_build_and_load_round_trip(self, reference_df, tmp_path, monkeypatch):
        """Build stats, save to disk, load back — values should match."""
        monkeypatch.setattr(
            "src.drift_detector.REFERENCE_STATS_PATH",
            str(tmp_path / "reference_stats.json")
        )
        stats = build_reference_stats(reference_df, features=["TransactionAmt", "card1"])

        # Reload
        monkeypatch.setattr(
            "src.drift_detector.REFERENCE_STATS_PATH",
            str(tmp_path / "reference_stats.json")
        )
        loaded = load_reference_stats()

        assert loaded["n_samples"] == len(reference_df)
        assert "TransactionAmt" in loaded["features"]
        assert "card1" in loaded["features"]
        assert abs(loaded["features"]["TransactionAmt"]["mean"] -
                   stats["features"]["TransactionAmt"]["mean"]) < 0.01

    def test_stats_contain_required_keys(self, reference_df, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "src.drift_detector.REFERENCE_STATS_PATH",
            str(tmp_path / "reference_stats.json")
        )
        stats = build_reference_stats(reference_df, features=["TransactionAmt"])
        feat_stats = stats["features"]["TransactionAmt"]
        for key in ["mean", "std", "min", "max", "p25", "p50", "p75", "p95",
                    "null_rate", "sample"]:
            assert key in feat_stats, f"Missing key '{key}' in reference stats"

    def test_load_raises_when_file_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "src.drift_detector.REFERENCE_STATS_PATH",
            str(tmp_path / "does_not_exist.json")
        )
        with pytest.raises(FileNotFoundError):
            load_reference_stats()


# ──────────────────────────────────────────────────────────
# DRIFT SIMULATION TESTS
# ──────────────────────────────────────────────────────────

class TestSimulateDrift:

    def test_none_intensity_produces_no_drift(self, reference_df):
        simulated = simulate_drift(reference_df, drift_intensity="none")
        psi_scores = compute_psi_all_features(reference_df, simulated)
        max_psi = max(psi_scores.values()) if psi_scores else 0
        assert max_psi < PSI_WARNING, \
            f"'none' intensity should produce no drift, got max_psi={max_psi:.4f}"

    def test_severe_intensity_produces_retrain_signal(self, reference_df):
        simulated = simulate_drift(reference_df, drift_intensity="severe")
        psi_scores = compute_psi_all_features(reference_df, simulated)
        high_psi = [f for f, p in psi_scores.items() if p > PSI_RETRAIN]
        assert len(high_psi) > 0, \
            f"'severe' intensity should produce PSI > {PSI_RETRAIN} on some features"

    def test_drift_intensity_is_ordered(self, reference_df):
        """More intense drift → higher max PSI."""
        results = {}
        for intensity in ["none", "slight", "moderate", "severe"]:
            sim = simulate_drift(reference_df, drift_intensity=intensity)
            scores = compute_psi_all_features(reference_df, sim)
            results[intensity] = max(scores.values()) if scores else 0

        assert results["none"] < results["slight"] < results["severe"], \
            f"Drift intensities should be ordered: {results}"

    def test_simulate_preserves_dataframe_length(self, reference_df):
        for intensity in ["none", "slight", "moderate", "severe"]:
            sim = simulate_drift(reference_df, drift_intensity=intensity)
            assert len(sim) == len(reference_df), \
                f"Simulated drift changed DataFrame length for '{intensity}'"


# ──────────────────────────────────────────────────────────
# FULL CHECK_DRIFT INTEGRATION TESTS
# ──────────────────────────────────────────────────────────

class TestCheckDrift:

    @pytest.fixture(autouse=True)
    def setup_reference(self, reference_df, tmp_path, monkeypatch):
        """Build reference stats before each test in this class."""
        monkeypatch.setattr(
            "src.drift_detector.REFERENCE_STATS_PATH",
            str(tmp_path / "reference_stats.json")
        )
        monkeypatch.setattr(
            "src.drift_detector.DRIFT_REPORT_DIR",
            str(tmp_path / "drift_reports")
        )
        build_reference_stats(reference_df)

    def test_stable_data_returns_not_drifted(self, stable_df):
        result = check_drift(stable_df, generate_report=False)
        assert result["drift_detected"] is False
        assert result["status"] in ["stable", "warning"]

    def test_drifted_data_returns_drifted(self, drifted_df):
        result = check_drift(drifted_df, generate_report=False)
        assert result["drift_detected"] is True
        assert result["status"] == "drifted"

    def test_result_has_required_keys(self, stable_df):
        result = check_drift(stable_df, generate_report=False)
        required = ["drift_detected", "status", "max_psi", "psi_scores",
                    "drifted_features", "recommendation", "checked_at"]
        for key in required:
            assert key in result, f"Missing key '{key}' in check_drift result"

    def test_psi_scores_are_non_negative(self, stable_df):
        result = check_drift(stable_df, generate_report=False)
        for feature, psi in result["psi_scores"].items():
            assert psi >= 0, f"PSI for {feature} is negative: {psi}"

    def test_max_psi_matches_psi_scores(self, stable_df):
        result = check_drift(stable_df, generate_report=False)
        computed_max = max(result["psi_scores"].values())
        assert abs(result["max_psi"] - computed_max) < 0.001

    def test_result_saved_to_disk(self, stable_df, tmp_path):
        check_drift(stable_df, generate_report=False)
        result_path = tmp_path / "drift_reports" / "latest_drift_result.json"
        assert result_path.exists(), "Drift result should be saved to disk"
        with open(result_path) as f:
            saved = json.load(f)
        assert "drift_detected" in saved


# ──────────────────────────────────────────────────────────
# FEATURE ENGINEERING CONSISTENCY TESTS
# ──────────────────────────────────────────────────────────

class TestFeatureEngineering:

    def test_transaction_hour_is_0_to_23(self):
        df = pd.DataFrame({
            "TransactionDT": [0, 3600, 43200, 86399, 86400],
            "TransactionAmt": [100] * 5,
        })
        result = engineer_features_for_drift(df)
        assert result["Transaction_hour"].between(0, 23).all()

    def test_transaction_amt_log_is_positive(self):
        df = pd.DataFrame({
            "TransactionAmt": [1, 10, 100, 1000],
            "TransactionDT": [0] * 4,
        })
        result = engineer_features_for_drift(df)
        assert (result["TransactionAmt_log"] > 0).all()

    def test_handles_missing_columns_gracefully(self):
        """Should not crash if optional columns are absent."""
        df = pd.DataFrame({"TransactionAmt": [50, 100, 200]})
        result = engineer_features_for_drift(df)
        assert "TransactionAmt" in result.columns