"""
Utility classes for processing, organizing, data imputation, output formatting
"""

import json
from collections import defaultdict
from typing import Any

import pandas as pd
from pydantic import BaseModel, field_validator, model_validator

import scripts.analyzer as an
import scripts.formulas as fm


class CleanedBaseModel(BaseModel):
    """Pydantic model cleans missing values before validation"""

    @field_validator("*", mode = "before")
    @classmethod
    def clean_missing_values(cls, value: Any) -> Any:
        """Convert panda NaNs & missing value strings to None"""

        if pd.isna(value):
            return None

        if isinstance(value, str):
            cleaned_str = value.strip().lower()
            if cleaned_str in {"unknown_sd", "none", "nan", "", "n/a", "nr"}:
                return None
            return value.strip()

        return value


class StudyGroup(CleanedBaseModel):
    """Represents single experimental or control group in a study"""

    name: str | None = None

    n: float | None = None
    mean: float | None = None

    sd: Any = None
    se: float | None = None

    ci_low: float | None = None
    ci_high: float | None = None

    median: float | None = None
    q1: float | None = None
    q3: float | None = None

    min_val: float | None = None
    max_val: float | None = None

    @property
    def has_cv_data(self) -> bool:
        """Return True if study group has data to calculate CV"""

        return self.sd is not None and self.mean is not None and self.n is not None

    @property
    def cv_tuple(self) -> tuple[float, float]:
        """Return tuple w/ CV and sample size (cv, n)"""

        sd = self.sd
        mean = self.mean
        n = self.n
        return (sd / mean), n


class StudyMetadata(CleanedBaseModel):
    """Holds metadata for each study"""

    # identification
    study_id: str = ""
    author: str | None = None
    year: float | None = None
    funding_source: str | None = None

    # population
    mean_age: float | None = None
    percent_female: float | None = None
    skin_type: str | None = None
    body_site: str | None = None

    # methodology
    study_design: str | None = None
    test_type: str | None = None
    labeled_spf: float | None = None
    application_interval_mins: float | None = None

    # statistics
    reported_metric: str | None = None
    metric_unit: str | None = None
    p_value: float | str | None = None
    correlation_r: float | None = None
    lower_is_better: bool = False

    # flags
    risk_of_bias: str | None = None
    attrition_n: float | None = None
    is_skewed: bool = False
    used_cv_imputation: bool = False
    is_parametric_p: bool = False


class StudyRecord(BaseModel):
    """Organize metadata and both study groups"""

    metadata: StudyMetadata
    group_1: StudyGroup
    group_2: StudyGroup

    @model_validator(mode = "before")
    @classmethod
    def nest_flat_csv_data(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Formats CSV rows into nested disctionary"""

        if "metadata" in values and "group_1" in values:
            return values

        group_1_data = {}
        group_2_data = {}
        metadata_data = {}

        for key, value in values.items():
            if key.startswith("group_1_"):
                attribute_name = key.replace("group_1_", "")

                if attribute_name in ["min", "max"]:
                    attribute_name = f"{attribute_name}_val"
                group_1_data[attribute_name] = value

            elif key.startswith("group_2_"):
                attribute_name = key.replace("group_2_", "")

                if attribute_name in ["min", "max"]:
                    attribute_name = f"{attribute_name}_val"
                group_2_data[attribute_name] = value

            else:
                metadata_data[key] = value

        return {
            "metadata": metadata_data,
            "group_1": group_1_data,
            "group_2": group_2_data
        }


class MissingDataImputer:
    """Handle imputation of missing statistics"""

    def __init__(self, dynamic_cvs: dict[str, float], fallback_cv: float = 0.25):
        self.dynamic_cvs = dynamic_cvs
        self.fallback_cv = fallback_cv

    def process_study(self, record: StudyRecord):
        """Impute missing variance for both groups in study record"""

        metric = record.metadata.metric_unit
        estimated_cv = self.dynamic_cvs.get(metric, self.fallback_cv)

        self._impute_group(record.group_1, record.metadata, "Group 1", estimated_cv)
        self._impute_group(record.group_2, record.metadata, "Group 2", estimated_cv)

    def _impute_group(
        self, group: StudyGroup, metadata: StudyMetadata, group_name: str, cv: float
    ):
        """Core logic for imputions"""

        if group.sd is not None:
            return

        # Impute from IQR
        if group.q1 is not None and group.q3 is not None and group.n is not None:
            if group.median is not None:
                is_skewed = fm.ImputationMath.check_skewness(
                    metadata.study_id, group.q1, group.median, group.q3, group_name
                )

                if is_skewed:
                    metadata.is_skewed = True

            group.sd = fm.ImputationMath.iqr_conversion_to_sd(group.q3, group.q1, group.n)
            if group.mean is None and group.median is not None:
                group.mean = fm.ImputationMath.estimate_mean_from_iqr(
                    group.q1, group.median, group.q3
                )
            return

        # Impute from Range
        if group.min_val is not None and group.max_val is not None and group.n is not None:
            group.sd = fm.ImputationMath.range_conversion(
                group.max_val, group.min_val, group.n
            )

            if group.mean is None and group.median is not None:
                group.mean = fm.ImputationMath.estimate_mean_from_range(
                    group.min_val, group.median, group.max_val
                )
            return

        # Impute via CV Substitution
        if group.mean is not None:
            group.sd = fm.ImputationMath.cv_substitution(group.mean, cv)
            metadata.used_cv_imputation = True
            return


class MetaAnalysisDataset:
    """Manage full dataset, loading records & applying operations"""

    def __init__(self):
        self.records: list[StudyRecord] = []
        self.dynamic_cvs: dict[str, float] = {}

    def load_rob_judgements(self, filepath: str):
        """
        Load Risk of Bias judgment from JSON file. data/rob_judgements.json
        Rational found data/methodologies&comments.ipynb
        """

        with open(filepath, "r", encoding = "utf-8") as file:
            return json.load(file)

    def load_from_csv(self, filepath: str):
        """Load raw data trail from CSV into Pydantic models"""

        df = pd.read_csv(filepath)
        self.records = [StudyRecord(**row) for row in df.to_dict(orient = "records")]
        return self.records

    def calculate_dynamic_cvs(self, fallback_cv: float = 0.25):
        """Build dynamic correlation of variation (CV) dictionary"""

        cv_tracker = {}

        for record in self.records:
            metric = record.metadata.metric_unit
            if metric not in cv_tracker:
                cv_tracker[metric] = []

            if record.group_1.has_cv_data:
                cv_tracker[metric].append(record.group_1.cv_tuple)
            if record.group_2.has_cv_data:
                cv_tracker[metric].append(record.group_2.cv_tuple)

        for metric, cv_data in cv_tracker.items():
            if len(cv_data) == 0:
                self.dynamic_cvs[metric] = fallback_cv
                continue

            total_weighted_cv = sum(cv * n for cv, n in cv_data)
            total_n = sum(n for cv, n in cv_data)
            self.dynamic_cvs[metric] = total_weighted_cv / total_n

    def apply_imputations(self, fallback_cv: float = 0.25):
        """Run MissingDataImputer class for all records in dataset"""

        imputer = MissingDataImputer(self.dynamic_cvs, fallback_cv)
        for record in self.records:
            imputer.process_study(record)


class AnalysisReporter:
    """Handles grouping data and formatting output"""

    @staticmethod
    def group_and_filter(study_data_pairs, condition = None):
        """Group study by metric w/ filter condition"""

        groups = defaultdict(list)

        for effect, record in study_data_pairs:
            if condition is None or condition(record):
                category = f"{effect["type"].upper()} | {effect["unit"].upper()}"
                groups[category].append(effect)
        return groups

    @staticmethod
    def print_results(analysis_title, grouped_studies):
        """Standardize printed output"""

        print(f"\n  [{analysis_title}]")
        for group_name, effect_sizes in grouped_studies.items():
            pooled = an.pool_subgroup(effect_sizes)

            if not pooled:
                continue

            print(
                f"    {group_name} (k = {pooled["k"]}) -> "
                f"RE Mean: {pooled["mean"]:.2f} "
                f"[95% CI: {pooled["lower"]:.2f}, {pooled["upper"]:.2f}], "
                f"I^2: {pooled["i2"]:.1f}% "
                f"[95% CI: {pooled["i2_lower"]:.1f}%, {pooled["i2_upper"]:.1f}%]"
            )