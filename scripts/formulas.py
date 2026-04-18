"""
Effect size calculation, impuation, and pooling formulas. 
"""

import numpy as np
from scipy import stats
from scipy.special import gammaln
from scipy.stats import norm
from scipy.optimize import minimize_scalar

class ImputationMath:
    """Estimate missing data"""

    @staticmethod
    def iqr_conversion_to_sd(q3, q1, n):
        """Convert IQR to Standard Deviation"""

        numerator = q3 - q1
        denominator = 2 * norm.ppf((0.75 * n - 0.125) / (n + 0.25))
        return numerator / denominator

    @staticmethod
    def check_skewness(study, q1, median, q3, group_name):
        """Flags highly skewed data"""

        q1, median, q3 = float(q1), float(median), float(q3)
        iqr_spread = q3 - q1

        if iqr_spread > 0:
            skew_difference = abs((q3 - median) - (median - q1))
            skew_ratio = skew_difference / iqr_spread

            if skew_ratio > 0.20:
                print(f"Study {study} ({group_name}) has highly skewed data. Ratio: {skew_ratio:.2f}.")
                return True
        return False

    @staticmethod
    def range_conversion(max_val, min_val, n):
        """Estimates Standard Deviation from reported minimum and maximum range"""

        standard_constant = None
        if n <= 15:
            standard_constant = np.sqrt(12)
        elif 15 < n <= 70:
            standard_constant = 4
        elif n > 70:
            standard_constant = 6

        estimated_sd = (max_val - min_val) / standard_constant
        return estimated_sd

    @staticmethod
    def cv_substitution(target_mean, known_cv):
        """Imputes Standard Deviation using Coefficient of Variation (CV)"""

        return target_mean * known_cv

    @staticmethod
    def estimate_mean_from_iqr(q1, m, q3):
        """Estimate sample mean using median & IQR"""

        return (q1 + m + q3) / 3

    @staticmethod
    def estimate_mean_from_range(min_val, m, max_val):
        """Estimate sample mean using median & range"""

        return (min_val + (2 * m) + max_val) / 4


class EffectSizeMath:
    """Calculate standardize effect size (Hedges' g) and variances"""

    @staticmethod
    def calculate_pooled_sd(sd1, sd2, n1, n2):
        """Calculates pooled standard deviation for two groups"""

        pooled_variance = ((n1 - 1) * (sd1 ** 2) + (n2 - 1) * (sd2 ** 2)) / (n1 + n2 - 2)
        if pooled_variance <= 0:
            raise ValueError(
                f"Pooled variance is {pooled_variance}. "
                f"Standard deviations: {sd1}, {sd2}. "
                f"There must be a typo in the CSV."
            )
        return np.sqrt(pooled_variance)

    @staticmethod
    def calculate_cohens_d(mean1, mean2, pooled_sd):
        """Calculate standardized mean difference, (Cohen's d)"""

        return (mean2 - mean1) / (pooled_sd + 1e-9)

    @staticmethod
    def calculate_hedges_correction_factor(n1, n2, d, is_paired = False):
        """Calculate Hedges' g and exact Gamma correction factor J"""

        if is_paired:
            df = n1 - 1
        else:
            df = (n1 + n2) - 2

        if df < 2:
            return d, 1.0

        # In logarithms
        numerator = gammaln(df / 2)
        denominator1 = 0.5 * np.log(df / 2)
        denominator2 = gammaln((df - 1) / 2)
        log_j = numerator - denominator1 - denominator2

        # Inverse Logarithm
        j_factor = np.exp(log_j)
        g = d * j_factor
        return g, j_factor

    @staticmethod
    def extract_r_from_p(mean_difference, sd1, sd2, n, t_stat):
        """calculate within-subject correlation r from paired p-value"""

        se_difference = abs(mean_difference / t_stat)
        sd_difference = se_difference * np.sqrt(n)
        r = ((sd1 ** 2) + (sd2 ** 2) - (sd_difference ** 2)) / (2 * sd1 * sd2)
        result = max(min(r, 1.0), -1.0)
        return result

    @staticmethod
    def unified_standard_error(n1, n2, d, j_factor, r = None, is_paired = False):
        """Calculate standard error for g"""

        if is_paired and r is not None:
            variance_d = ((2 * (1 - r)) / n1) + ((d ** 2) / (2 * (n1 - 1)))
        else:
            variance_d = ((n1 + n2) / (n1 * n2)) + ((d ** 2) / (2 * (n1 + n2)))

        variance_g = variance_d * (j_factor ** 2)
        return np.sqrt(variance_g)


class PoolingMath:
    """Handles random-effects, heterogenity testing, pooling"""

    @staticmethod
    def initial_weight(seg):
        """Calculate inverse-variance weight for fixed-effect model (not used)"""
        return 1 / (seg ** 2)

    @staticmethod
    def calculate_ci(g, se):
        """Calculate 95% confidence interval for standard normal distribution."""
        lower = g - (1.96 * se)
        upper = g + (1.96 * se)
        return lower, upper

    @staticmethod
    def calculate_fixed_effect_mean(weights, g_values):
        """Calculates pooled mean under fixed-effect assumption"""
        weighted_sum = np.sum(np.array(weights) * np.array(g_values))
        total_weight = np.sum(weights)
        return weighted_sum / total_weight

    @staticmethod
    def calculate_cochrans_q(weights, g_values, fixed_mean):
        """Calculate Cochran Q for heterogenity"""
        w = np.array(weights)
        g = np.array(g_values)
        return np.sum(w * ((g - fixed_mean) ** 2))

    @staticmethod
    def calculate_i_squared(q_stat, k):
        """Calculate I-squared statistic for heterogenity variance percentage"""
        if q_stat <= (k - 1):
            return 0.0
        return ((q_stat - (k - 1)) / q_stat) * 100

    @staticmethod
    def calculate_i_squared_ci(q_stat, k):
        """Calculate 95% confidence interval for I-squared statistic"""
        if k <= 2:
            return 0.0, 100.0

        df = k - 1
        h_stat = 1.0 if q_stat <= df else np.sqrt(q_stat / df)

        if q_stat > k:
            se_ln_h = (np.log(q_stat) - np.log(df)) / (
                2 * (np.sqrt(2 * q_stat) - np.sqrt(2 * k - 3))
            )
        else:
            se_ln_h = np.sqrt(
                (1 / (2 * (k - 2))) * (1 - (1 / (3 * ((k - 2) ** 2))))
            )

        ln_h = np.log(h_stat)
        ln_h_lower = ln_h - (1.96 * se_ln_h)
        ln_h_upper = ln_h + (1.96 * se_ln_h)

        h_lower = np.exp(ln_h_lower)
        h_upper = np.exp(ln_h_upper)

        i2_lower = max(0.0, ((h_lower ** 2) - 1) / (h_lower ** 2)) * 100
        i2_upper = max(0.0, ((h_upper ** 2) - 1) / (h_upper ** 2)) * 100

        return i2_lower, min(100.0, i2_upper)

    @staticmethod
    def calculate_tau_squared(se_values, g_values):
        """Estimate tau-squared w/ REML"""
        v = np.array(se_values) ** 2
        y = np.array(g_values)
        k = len(y)

        if k <= 1:
            return 0.0

        def reml_negative_log_likelihood(tau2):
            if tau2 < 0:
                return np.inf
            w = 1.0 / (v + tau2)
            mu = np.sum(w * y) / np.sum(w)
            term1 = np.sum(np.log(v + tau2))
            term2 = np.log(np.sum(w))
            term3 = np.sum(w * (y - mu) ** 2)
            result = 0.5 * (term1 + term2 + term3)
            return result

        result = minimize_scalar(reml_negative_log_likelihood, bounds = (0, 10), method = "bounded")
        return result.x if result.success else 0.0

    @staticmethod
    def random_effect_weight(seg, tau_squared):
        """Calculate adjusted weight for random-effects model"""
        return 1 / ((seg ** 2) + tau_squared)

    @staticmethod
    def calculate_random_effects_mean(random_weights, g_values):
        """Calculate pooled mean under random effects assumption"""
        w_star = np.array(random_weights)
        g = np.array(g_values)
        return np.sum(w_star * g) / np.sum(w_star)

    @staticmethod
    def apply_knapp_hartung(effects, weights, pooled_effect, k, alpha = 0.05):
        """Apply Knapp-Hartung variance for small sample sizes"""
        if k <= 1:
            standard_se = np.sqrt(1.0 / np.sum(weights))
            return (
                pooled_effect - (1.96 * standard_se),
                pooled_effect + (1.96 * standard_se),
                standard_se
            )

        effects = np.array(effects)
        weights = np.array(weights)

        standard_var = 1.0 / np.sum(weights)
        q_squared = np.sum(weights * (effects - pooled_effect) ** 2) / (k - 1)
        q_squared = max(1.0, q_squared)

        kh_var = q_squared * standard_var
        kh_se = np.sqrt(kh_var)

        t_crit = stats.t.ppf(1 - alpha / 2, df = k - 1)
        ci_lower = pooled_effect - (t_crit * kh_se)
        ci_upper = pooled_effect + (t_crit * kh_se)

        return ci_lower, ci_upper, kh_se