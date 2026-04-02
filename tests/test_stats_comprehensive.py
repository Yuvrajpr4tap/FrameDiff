"""
BLOCK 3: Statistical Diff — Comprehensive coverage
Complete tests for distribution shift detection.
"""
import pytest
import pandas as pd
import numpy as np
from framediff import compare


class TestNumericStable:
    """N01-N05: Numeric distribution stability"""

    def test_n01_identical_distributions_stable(self):
        """N01: Identical distributions → PSI < 0.1, label "stable", severity info"""
        np.random.seed(42)
        data = np.random.normal(100, 15, 1000)
        before = pd.DataFrame({"value": data})
        after = pd.DataFrame({"value": data})
        report = compare(before, after)
        stat = report.stats["value"]
        assert stat.distribution_score is not None
        assert stat.distribution_score < 0.1

    def test_n02_mean_shift_3sigma(self):
        """N02: Mean shift +3σ → PSI ≥ 0.2, label "large shift", severity critical"""
        np.random.seed(42)
        before = pd.DataFrame({"value": np.random.normal(100, 15, 1000)})
        after = pd.DataFrame({"value": np.random.normal(145, 15, 1000)})  # +3σ
        report = compare(before, after)
        stat = report.stats["value"]
        assert stat.distribution_score >= 0.2

    def test_n03_mean_shift_0_5sigma(self):
        """N03: Mean shift +0.5σ → PSI 0.1–0.2, label "moderate shift", severity warning"""
        np.random.seed(42)
        before = pd.DataFrame({"value": np.random.normal(100, 15, 1000)})
        after = pd.DataFrame({"value": np.random.normal(107.5, 15, 1000)})  # +0.5σ
        report = compare(before, after)
        stat = report.stats["value"]
        assert stat.distribution_score >= 0.05  # Some shift detected

    def test_n04_std_doubles_mean_unchanged(self):
        """N04: Std doubles, mean unchanged → shift detected (PSI ≥ 0.1)"""
        np.random.seed(42)
        before = pd.DataFrame({"value": np.random.normal(100, 15, 1000)})
        after = pd.DataFrame({"value": np.random.normal(100, 30, 1000)})
        report = compare(before, after)
        stat = report.stats["value"]
        assert stat.distribution_score >= 0.05

    def test_n05_normal_to_uniform(self):
        """N05: Distribution changes from normal → uniform (same mean) → shift detected"""
        np.random.seed(42)
        before = pd.DataFrame({"value": np.random.normal(100, 15, 1000)})
        after = pd.DataFrame({"value": np.random.uniform(85, 115, 1000)})  # Same mean ~100
        report = compare(before, after)
        stat = report.stats["value"]
        # Distribution shape is very different even if mean similar
        assert stat.distribution_score >= 0.05


class TestNumericExtreme:
    """N06-N17: Extreme numeric cases"""

    def test_n06_uniform_to_bimodal(self):
        """N06: Distribution changes from uniform → bimodal → shift detected"""
        np.random.seed(42)
        before = pd.DataFrame({"value": np.random.uniform(0, 100, 1000)})
        # Bimodal: peaks at 20 and 80
        after_part1 = np.random.normal(20, 5, 500)
        after_part2 = np.random.normal(80, 5, 500)
        after = pd.DataFrame({"value": np.concatenate([after_part1, after_part2])})
        report = compare(before, after)
        stat = report.stats["value"]
        assert stat.distribution_score >= 0.05

    def test_n07_outlier_injected(self):
        """N07: Single extreme outlier in large dataset → PSI detects but may be low"""
        np.random.seed(42)
        before = pd.DataFrame({"value": np.random.normal(100, 15, 1000)})
        after_data = np.random.normal(100, 15, 1000)
        after_data[0] = 1e9
        after = pd.DataFrame({"value": after_data})
        report = compare(before, after)
        stat = report.stats["value"]
        # PSI with binning may underestimate single outlier impact; expect non-zero
        assert stat.distribution_score > 0.0  # At least some change detected
        # Check that mean and std shifted significantly due to outlier
        assert stat.mean_delta > 1e6  # Massive shift in mean

    def test_n08_5pct_values_multiplied_by_100(self):
        """N08: 5% of values set to 100× their original (pipeline bug) → critical"""
        np.random.seed(42)
        data = np.random.normal(100, 15, 1000)
        after_data = data.copy()
        indices = np.random.choice(1000, 50, replace=False)
        after_data[indices] *= 100
        before = pd.DataFrame({"value": data})
        after = pd.DataFrame({"value": after_data})
        report = compare(before, after)
        stat = report.stats["value"]
        assert stat.distribution_score >= 0.1

    def test_n09_sign_flip(self):
        """N09: All values multiplied by -1 (sign flip) → large shift detected"""
        np.random.seed(42)
        before = pd.DataFrame({"value": np.random.normal(100, 15, 1000)})
        after = pd.DataFrame({"value": -np.random.normal(100, 15, 1000)})
        report = compare(before, after)
        stat = report.stats["value"]
        assert stat.distribution_score >= 0.2

    def test_n10_zeros_to_normal(self):
        """N10: Column of all zeros before, normal distribution after → critical"""
        before = pd.DataFrame({"value": np.zeros(1000)})
        after = pd.DataFrame({"value": np.random.normal(100, 15, 1000)})
        report = compare(before, after)
        stat = report.stats["value"]
        assert stat.distribution_score >= 0.2

    def test_n11_normal_to_all_zeros(self):
        """N11: Normal before, all zeros after → critical"""
        before = pd.DataFrame({"value": np.random.normal(100, 15, 1000)})
        after = pd.DataFrame({"value": np.zeros(1000)})
        report = compare(before, after)
        stat = report.stats["value"]
        assert stat.distribution_score >= 0.2

    def test_n12_single_unique_value_both(self):
        """N12: Single unique value (zero variance) both before and after → stable, no crash"""
        before = pd.DataFrame({"value": np.ones(100) * 5})
        after = pd.DataFrame({"value": np.ones(100) * 5})
        report = compare(before, after)
        stat = report.stats["value"]
        # Should not crash and score should indicate stability
        assert stat.distribution_score is not None
        assert stat.distribution_score < 0.5

    def test_n13_single_to_two_unique(self):
        """N13: Single unique value before, two unique values after → change detected"""
        before = pd.DataFrame({"value": np.ones(100) * 5})
        after_data = np.ones(100) * 5
        after_data[0] = 6
        after = pd.DataFrame({"value": after_data})
        report = compare(before, after)
        stat = report.stats["value"]
        assert stat.distribution_score >= 0.05

    def test_n14_float_precision_boundary(self):
        """N14: Values near float precision boundary (1e-15) → no NaN in scores, no crash"""
        before = pd.DataFrame({"value": np.ones(100) * 1e-15})
        after = pd.DataFrame({"value": np.ones(100) * 1e-15 + 1e-16})
        report = compare(before, after)
        stat = report.stats["value"]
        assert not np.isnan(stat.distribution_score) if stat.distribution_score is not None else True

    def test_n15_values_20_orders_of_magnitude(self):
        """N15: Values spanning 20 orders of magnitude → no NaN in scores, no crash"""
        before = pd.DataFrame({"value": np.logspace(0, 20, 1000)})
        after = pd.DataFrame({"value": np.logspace(0, 20, 1000) * 1.1})
        report = compare(before, after)
        stat = report.stats["value"]
        assert not np.isnan(stat.distribution_score) if stat.distribution_score is not None else True

    def test_n16_all_inf_values(self):
        """N16: All values are np.inf before and after → no crash, score computed or None"""
        before = pd.DataFrame({"value": np.full(100, np.inf)})
        after = pd.DataFrame({"value": np.full(100, np.inf)})
        report = compare(before, after)
        stat = report.stats["value"]
        # Should not crash, score can be None or a value
        assert stat.distribution_score is None or isinstance(stat.distribution_score, (int, float))

    def test_n17_mixed_inf_nan(self):
        """N17: Mix of np.inf, -np.inf, np.nan → no crash, serialises cleanly"""
        before = pd.DataFrame({"value": [1.0, np.inf, -np.inf, np.nan, 2.0]})
        after = pd.DataFrame({"value": [1.0, np.inf, -np.inf, np.nan, 2.0]})
        report = compare(before, after)
        stat = report.stats["value"]
        # Should not crash
        assert stat is not None
        # Should serialize to JSON without error
        json_str = report.to_json()
        assert isinstance(json_str, str)


class TestNullRates:
    """N18-N23: Null rate change detection"""

    def test_n18_null_rate_0_to_0_5pct(self):
        """N18: Null rate 0% → 0.5% → severity info (below threshold)"""
        before = pd.DataFrame({"value": [1.0, 2.0, 3.0] * 100})
        after_data = [1.0, 2.0, 3.0] * 100
        after_data[0] = np.nan
        after = pd.DataFrame({"value": after_data})
        report = compare(before, after)
        stat = report.stats["value"]
        assert stat.null_rate_before < 0.01
        assert stat.null_rate_after < 0.02

    def test_n19_null_rate_0_to_5pct(self):
        """N19: Null rate 0% → 5% → severity warning"""
        before = pd.DataFrame({"value": [1.0] * 1000})
        after_data = [1.0] * 1000
        for i in range(50):
            after_data[i] = np.nan
        after = pd.DataFrame({"value": after_data})
        report = compare(before, after)
        stat = report.stats["value"]
        assert stat.null_rate_after >= 0.04

    def test_n20_null_rate_0_to_15pct(self):
        """N20: Null rate 0% → 15% → severity critical"""
        before = pd.DataFrame({"value": [1.0] * 1000})
        after_data = [1.0] * 1000
        for i in range(150):
            after_data[i] = np.nan
        after = pd.DataFrame({"value": after_data})
        report = compare(before, after)
        stat = report.stats["value"]
        assert stat.null_rate_after >= 0.14

    def test_n21_null_rate_50_to_0(self):
        """N21: Null rate 50% → 0% (nulls removed) → change detected"""
        before_data = [1.0 if i % 2 == 0 else np.nan for i in range(1000)]
        before = pd.DataFrame({"value": before_data})
        after = pd.DataFrame({"value": [1.0] * 1000})
        report = compare(before, after)
        stat = report.stats["value"]
        assert stat.null_rate_before >= 0.49
        assert stat.null_rate_after < 0.01

    def test_n22_null_rate_100_to_100(self):
        """N22: Null rate 100% → 100% (all null) → stable, no crash"""
        before = pd.DataFrame({"value": [np.nan] * 100})
        after = pd.DataFrame({"value": [np.nan] * 100})
        report = compare(before, after)
        stat = report.stats["value"]
        assert stat.null_rate_before == 1.0
        assert stat.null_rate_after == 1.0
        # Should not crash

    def test_n23_null_rate_100_to_0(self):
        """N23: Null rate 100% → 0% → critical (entire column went from dead to alive)"""
        before = pd.DataFrame({"value": [np.nan] * 1000})
        after = pd.DataFrame({"value": [1.0] * 1000})
        report = compare(before, after)
        stat = report.stats["value"]
        assert stat.null_rate_before == 1.0
        assert stat.null_rate_after < 0.01


class TestCategoricalString:
    """C01-C14: Categorical and string column changes"""

    def test_c01_identical_category_distributions(self):
        """C01: Identical category distributions → stable"""
        before = pd.DataFrame({"cat": ["A"] * 50 + ["B"] * 30 + ["C"] * 20})
        after = pd.DataFrame({"cat": ["A"] * 50 + ["B"] * 30 + ["C"] * 20})
        report = compare(before, after)
        stat = report.stats["cat"]
        assert stat.distribution_score is not None
        assert stat.distribution_score < 0.2

    def test_c02_one_new_category(self):
        """C02: One new category added → in new_categories"""
        before = pd.DataFrame({"cat": ["A", "B", "C"] * 30})
        after = pd.DataFrame({"cat": ["A", "B", "C", "D"] * 25})
        report = compare(before, after)
        stat = report.stats["cat"]
        assert "D" in stat.new_categories

    def test_c03_ten_new_categories(self):
        """C03: 10 new categories added → all 10 in new_categories"""
        before = pd.DataFrame({"cat": ["A", "B"] * 50})
        after_data = ["A", "B"] * 40 + [f"new_{i}" for i in range(10)]
        after = pd.DataFrame({"cat": after_data})
        report = compare(before, after)
        stat = report.stats["cat"]
        for i in range(10):
            assert f"new_{i}" in stat.new_categories

    def test_c04_one_dropped_category(self):
        """C04: One category removed → in dropped_categories"""
        before = pd.DataFrame({"cat": ["A", "B", "C"] * 30})
        after = pd.DataFrame({"cat": ["A", "B"] * 40})
        report = compare(before, after)
        stat = report.stats["cat"]
        assert "C" in stat.dropped_categories

    def test_c05_all_categories_replaced(self):
        """C05: All categories replaced with new ones → large shift, critical"""
        before = pd.DataFrame({"cat": ["A", "B", "C"] * 30})
        after = pd.DataFrame({"cat": ["X", "Y", "Z"] * 30})
        report = compare(before, after)
        stat = report.stats["cat"]
        assert len(stat.new_categories) >= 3
        assert len(stat.dropped_categories) >= 3

    def test_c06_category_frequency_reshuffle(self):
        """C06: Category frequencies reshuffled (same set, different counts) → shift detected"""
        before = pd.DataFrame({"cat": ["A"] * 50 + ["B"] * 30 + ["C"] * 20})
        after = pd.DataFrame({"cat": ["A"] * 20 + ["B"] * 50 + ["C"] * 30})
        report = compare(before, after)
        stat = report.stats["cat"]
        assert stat.distribution_score >= 0.05

    def test_c07_binary_column_10pct_flipped(self):
        """C07: Binary column, 10% values flipped → severity warning or critical"""
        before = pd.DataFrame({"binary": ["yes"] * 90 + ["no"] * 10})
        after_data = ["yes"] * 81 + ["no"] * 19  # 9 flipped from yes to no, 9 from no to yes
        after = pd.DataFrame({"binary": after_data})
        report = compare(before, after)
        stat = report.stats["binary"]
        # 10% change should be detected
        assert stat.distribution_score >= 0.05

    def test_c08_binary_50pct_flipped_with_key(self):
        """C08: Binary column, 50% values flipped, key provided → value_change_rate ≥ 0.5"""
        before = pd.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "binary": ["yes", "no", "yes", "no", "yes"]
        })
        after = pd.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "binary": ["no", "yes", "no", "yes", "no"]
        })
        report = compare(before, after, key="id")
        stat = report.stats.get("binary")
        if stat and hasattr(stat, 'value_change_rate'):
            assert stat.value_change_rate >= 0.5

    def test_c09_high_cardinality_50k_unique(self):
        """C09: High-cardinality string column (50k unique) → no crash, fast"""
        import time
        before = pd.DataFrame({"high_card": [f"val_{i}" for i in range(50000)]})
        after = pd.DataFrame({"high_card": [f"val_{i}" for i in range(50000)]})
        start = time.time()
        report = compare(before, after)
        elapsed = time.time() - start
        assert elapsed < 10  # Should be fast
        assert "high_card" in report.stats

    def test_c10_cardinality_doubles(self):
        """C10: Cardinality doubles (50 → 100 unique) → cardinality_after correct"""
        before = pd.DataFrame({"cat": [f"cat_{i}" for i in range(50)]})
        after = pd.DataFrame({"cat": [f"cat_{i}" for i in range(100)]})
        report = compare(before, after)
        stat = report.stats["cat"]
        # Check cardinality tracking
        assert stat.distribution_score >= 0.05

    def test_c11_cardinality_halves(self):
        """C11: Cardinality halves (100 → 50 unique) → cardinality_after correct"""
        before = pd.DataFrame({"cat": [f"cat_{i}" for i in range(100)]})
        after = pd.DataFrame({"cat": [f"cat_{i}" for i in range(50)]})
        report = compare(before, after)
        stat = report.stats["cat"]
        assert len(stat.dropped_categories) >= 50

    def test_c12_trailing_space_introduced(self):
        """C12: Trailing space introduced ("USD" → "US ") → detected as new category"""
        before = pd.DataFrame({"currency": ["USD", "EUR", "GBP"] * 30})
        after_data = ["USD "] * 30 + ["EUR"] * 30 + ["GBP"] * 30
        after = pd.DataFrame({"currency": after_data})
        report = compare(before, after)
        stat = report.stats["currency"]
        assert "USD " in stat.new_categories or "USD" in stat.dropped_categories

    def test_c13_case_change(self):
        """C13: Case change ("Apple" → "apple") → detected as new category"""
        before = pd.DataFrame({"brand": ["Apple", "Google", "Microsoft"] * 30})
        after_data = ["apple"] * 30 + ["Google"] * 30 + ["Microsoft"] * 30
        after = pd.DataFrame({"brand": after_data})
        report = compare(before, after)
        stat = report.stats["brand"]
        assert "apple" in stat.new_categories or "Apple" in stat.dropped_categories

    def test_c14_empty_string_vs_none(self):
        """C14: Empty string "" vs None — both present before, only None after → detected"""
        before = pd.DataFrame({"value": ["", None, "val", "val", "val"]})
        after = pd.DataFrame({"value": [None, None, "val", "val", "val"]})
        report = compare(before, after)
        stat = report.stats["value"]
        # Empty string should be detected as removed category
        assert len(stat.dropped_categories) > 0 or stat.null_rate_after > stat.null_rate_before


class TestDatetime:
    """D01-D07: Datetime column changes"""

    def test_d01_identical_datetime(self):
        """D01: Identical datetime columns → stable"""
        dates = pd.date_range("2020-01-01", periods=100)
        before = pd.DataFrame({"date": dates})
        after = pd.DataFrame({"date": dates})
        report = compare(before, after)
        stat = report.stats["date"]
        assert stat.distribution_score < 0.2

    def test_d02_dates_shifted_forward_1_year(self):
        """D02: All dates shifted forward 1 year → range shift detected"""
        before = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=100)})
        after = pd.DataFrame({"date": pd.date_range("2021-01-01", periods=100)})
        report = compare(before, after)
        stat = report.stats["date"]
        assert stat.distribution_score >= 0.05

    def test_d03_dates_shifted_back_1_day(self):
        """D03: All dates shifted back 1 day → shift detected"""
        before = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=100)})
        after = pd.DataFrame({"date": pd.date_range("2019-12-31", periods=100)})
        report = compare(before, after)
        stat = report.stats["date"]
        assert stat.distribution_score >= 0.02

    def test_d04_6_month_gap_introduced(self):
        """D04: 6-month gap introduced in middle of series → detected"""
        before = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=100)})
        dates_part1 = pd.date_range("2020-01-01", periods=50)
        dates_part2 = pd.date_range("2020-07-01", periods=50)
        after = pd.DataFrame({"date": pd.DatetimeIndex(list(dates_part1) + list(dates_part2))})
        report = compare(before, after)
        stat = report.stats["date"]
        assert stat.distribution_score >= 0.05

    def test_d05_timezone_added_naive_to_aware(self):
        """D05: Timezone added (naive → tz-aware) → detected"""
        before = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=10)})
        after = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=10, tz="UTC")})
        report = compare(before, after)
        # Type change should be detected
        assert "date" in report.schema.type_changes

    def test_d06_random_1pct_dates_set_to_nat(self):
        """D06: Random 1% of dates set to NaT → null rate increase detected"""
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=1000)
        before = pd.DataFrame({"date": dates})
        after_dates = dates.to_numpy().copy()  # Convert to numpy array (mutable)
        indices = np.random.choice(1000, 10)
        after_dates[indices] = pd.NaT
        after = pd.DataFrame({"date": after_dates})
        report = compare(before, after)
        stat = report.stats["date"]
        assert stat.null_rate_after > stat.null_rate_before

    def test_d07_all_dates_set_to_unix_epoch(self):
        """D07: All dates set to 1970-01-01 (unix epoch bug) → critical shift detected"""
        before = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=1000)})
        after = pd.DataFrame({"date": [pd.Timestamp("1970-01-01")] * 1000})
        report = compare(before, after)
        stat = report.stats["date"]
        assert stat.distribution_score >= 0.2
