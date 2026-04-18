"""
Module for calculating and pooling effect sizes for the sunscreen meta-analysis.
Handles individual study calculations and random-effects pooling via REML.
"""
import pandas as pd
from scipy import stats
import scripts.formulas as fm

def extract_study_correlation(study, assumed_r=0.5):
    """Calculate r from p-value or assume r = 0.5"""
    if study.metadata.study_design.lower() != "paired":
        return None

    if pd.notna(study.metadata.correlation_r) and study.metadata.correlation_r != "":
        return float(study.metadata.correlation_r)

    if pd.notna(study.metadata.p_value) and study.metadata.p_value != "":
        p_val_str = str(study.metadata.p_value).strip()

        if "<" in p_val_str or ">" in p_val_str:
            pass
        elif not getattr(study.metadata, "is_parametric_p", False):
            pass
        else:
            try:
                p_val = float(p_val_str)
                n = study.group_1.n
                mean_diff = study.group_2.mean - study.group_1.mean
                t_stat = stats.t.ppf(1 - (p_val / 2), n - 1)

                if study.group_1.sd is not None and study.group_2.sd is not None:
                    return fm.EffectSizeMath.extract_r_from_p(
                        mean_diff, study.group_1.sd, study.group_2.sd, n, t_stat
                    )
            except ValueError:
                pass

    return assumed_r

def calculate_study_effect(study, r_value):
    """Executes the math pathway to generate final Hedges' g and SE."""
    sd_pooled = fm.EffectSizeMath.calculate_pooled_sd(
        study.group_1.sd, study.group_2.sd, study.group_1.n, study.group_2.n
    )

    d = fm.EffectSizeMath.calculate_cohens_d(study.group_1.mean, study.group_2.mean, sd_pooled)
    if study.metadata.lower_is_better:
        d *= -1

    is_paired = r_value is not None
    g, j_factor = fm.EffectSizeMath.calculate_hedges_correction_factor(
        study.group_1.n, study.group_2.n, d, is_paired = is_paired
    )

    se = fm.EffectSizeMath.unified_standard_error(
        study.group_1.n, study.group_2.n, d, j_factor, r = r_value, is_paired = is_paired
    )

    return {
        "id": study.metadata.study_id,
        "g": g,
        "se": se,
        "w": fm.PoolingMath.initial_weight(se),
        "unit": study.metadata.metric_unit,
        "type": study.metadata.test_type
    }

def pool_subgroup(group):
    """Calculate random-effects pooling using REML"""
    k = len(group)
    if k == 0:
        return None

    g_values = [res["g"] for res in group]
    se_values = [res["se"] for res in group]
    weight_values = [res["w"] for res in group]

    if k == 1:
        g, se = g_values[0], se_values[0]
        lower, upper = fm.PoolingMath.calculate_ci(g, se)
        return {
            "k": 1, "mean": g, "lower": lower, "upper": upper,
            "i2": 0.0, "i2_lower": 0.0, "i2_upper": 0.0,
            "tau2": 0.0
        }

    tau_squared = fm.PoolingMath.calculate_tau_squared(se_values, g_values)
    fixed_mean = fm.PoolingMath.calculate_fixed_effect_mean(weight_values, g_values)
    q_stat = fm.PoolingMath.calculate_cochrans_q(weight_values, g_values, fixed_mean)
    i_squared = fm.PoolingMath.calculate_i_squared(q_stat, k)
    i2_lower, i2_upper = fm.PoolingMath.calculate_i_squared_ci(q_stat, k)

    random_weights = [fm.PoolingMath.random_effect_weight(se, tau_squared) for se in se_values]
    random_mean = fm.PoolingMath.calculate_random_effects_mean(random_weights, g_values)

    for i, study_dict in enumerate(group):
        study_dict["w"] = random_weights[i]

    lower, upper, _ = fm.PoolingMath.apply_knapp_hartung(
        effects = g_values, weights = random_weights, pooled_effect = random_mean, k = k
    )

    return {
        "k": k, "mean": random_mean, "lower": lower, "upper": upper,
        "i2": i_squared, "i2_lower": i2_lower, "i2_upper": i2_upper,
        "tau2": tau_squared
    }

def run_pipeline(records, assumed_r):
    """Iterate through records to compile effect sizes"""
    results = []
    for study in records:
        r_value = extract_study_correlation(study, assumed_r)
        effect_data = calculate_study_effect(study, r_value)
        results.append(effect_data)
    return results