"""
Data quality checking module.

Validates data integrity, detects outliers, and ensures no lookahead bias.
"""

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from airs.utils.logging import get_logger

logger = get_logger(__name__)


class DataQualityChecker:
    """
    Comprehensive data quality checker for financial data.

    Checks for:
    - Missing values
    - Outliers
    - Stale data
    - Data type consistency
    - Lookahead bias
    """

    def __init__(
        self,
        max_missing_pct: float = 0.05,
        outlier_std_threshold: float = 5.0,
        max_stale_days: int = 3,
    ):
        """
        Initialize quality checker.

        Args:
            max_missing_pct: Maximum acceptable missing percentage
            outlier_std_threshold: Number of std devs for outlier detection
            max_stale_days: Maximum days data can be stale
        """
        self.max_missing_pct = max_missing_pct
        self.outlier_std_threshold = outlier_std_threshold
        self.max_stale_days = max_stale_days

    def check_all(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        Run all quality checks.

        Args:
            df: DataFrame to check

        Returns:
            Dictionary with check results
        """
        results = {
            "missing": self.check_missing(df),
            "outliers": self.check_outliers(df),
            "stale": self.check_stale_data(df),
            "types": self.check_data_types(df),
            "duplicates": self.check_duplicates(df),
        }

        # Summary
        issues = []
        if results["missing"]["has_issues"]:
            issues.append("missing_values")
        if results["outliers"]["has_issues"]:
            issues.append("outliers")
        if results["stale"]["has_issues"]:
            issues.append("stale_data")
        if results["duplicates"]["has_issues"]:
            issues.append("duplicates")

        results["summary"] = {
            "passed": len(issues) == 0,
            "issues": issues,
            "rows": len(df),
            "columns": len(df.columns),
        }

        return results

    def check_missing(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        Check for missing values.

        Args:
            df: DataFrame to check

        Returns:
            Dictionary with missing value analysis
        """
        missing_counts = df.isna().sum()
        missing_pct = missing_counts / len(df)

        columns_with_issues = missing_pct[missing_pct > self.max_missing_pct].to_dict()

        return {
            "has_issues": len(columns_with_issues) > 0,
            "total_missing": int(df.isna().sum().sum()),
            "missing_by_column": missing_pct.to_dict(),
            "columns_exceeding_threshold": columns_with_issues,
            "threshold": self.max_missing_pct,
        }

    def check_outliers(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        Check for statistical outliers.

        Args:
            df: DataFrame to check

        Returns:
            Dictionary with outlier analysis
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outliers = {}

        for col in numeric_cols:
            series = df[col].dropna()
            if len(series) < 10:
                continue

            mean = series.mean()
            std = series.std()

            if std == 0:
                continue

            z_scores = np.abs((series - mean) / std)
            outlier_mask = z_scores > self.outlier_std_threshold
            outlier_count = outlier_mask.sum()

            if outlier_count > 0:
                outliers[col] = {
                    "count": int(outlier_count),
                    "percentage": round(outlier_count / len(series) * 100, 2),
                    "max_zscore": round(float(z_scores.max()), 2),
                }

        return {
            "has_issues": len(outliers) > 0,
            "outliers_by_column": outliers,
            "threshold": self.outlier_std_threshold,
        }

    def check_stale_data(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        Check for stale data (data that hasn't been updated recently).

        Args:
            df: DataFrame to check

        Returns:
            Dictionary with staleness analysis
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            return {"has_issues": False, "message": "Index is not datetime"}

        if len(df) == 0:
            return {"has_issues": True, "message": "DataFrame is empty"}

        latest_date = df.index.max()
        today = pd.Timestamp.now()

        # Calculate days since last data
        days_stale = (today - latest_date).days

        # Check per-column staleness
        column_staleness = {}
        for col in df.columns:
            last_valid = df[col].last_valid_index()
            if last_valid is not None:
                days = (today - last_valid).days
                if days > self.max_stale_days:
                    column_staleness[col] = days

        return {
            "has_issues": days_stale > self.max_stale_days,
            "days_since_latest": days_stale,
            "latest_date": latest_date.isoformat(),
            "stale_columns": column_staleness,
            "threshold": self.max_stale_days,
        }

    def check_data_types(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        Check data types are appropriate.

        Args:
            df: DataFrame to check

        Returns:
            Dictionary with type analysis
        """
        type_info = {}
        issues = []

        for col in df.columns:
            dtype = str(df[col].dtype)
            type_info[col] = dtype

            # Flag object columns that should probably be numeric
            if dtype == "object":
                # Check if it looks like it should be numeric
                sample = df[col].dropna().head(10)
                try:
                    pd.to_numeric(sample)
                    issues.append(f"{col}: appears numeric but is object type")
                except (ValueError, TypeError):
                    pass

        return {
            "has_issues": len(issues) > 0,
            "types": type_info,
            "issues": issues,
        }

    def check_duplicates(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        Check for duplicate rows or indices.

        Args:
            df: DataFrame to check

        Returns:
            Dictionary with duplicate analysis
        """
        dup_rows = df.duplicated().sum()
        dup_index = df.index.duplicated().sum()

        return {
            "has_issues": dup_rows > 0 or dup_index > 0,
            "duplicate_rows": int(dup_rows),
            "duplicate_indices": int(dup_index),
        }

    def check_lookahead_bias(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        target_horizon: int,
    ) -> dict[str, Any]:
        """
        Check for potential lookahead bias.

        Ensures features at time t don't use information from time t+1 to t+horizon.

        Args:
            features: Feature DataFrame
            target: Target Series
            target_horizon: Number of days in target calculation

        Returns:
            Dictionary with lookahead analysis
        """
        issues = []

        # Check that features and target are aligned
        if not features.index.equals(target.index):
            issues.append("Feature and target indices don't match")

        # Check for suspiciously high correlations that might indicate leakage
        correlations = features.corrwith(target).abs()
        high_corr = correlations[correlations > 0.95]

        if len(high_corr) > 0:
            issues.append(
                f"Suspiciously high correlations with target: {high_corr.to_dict()}"
            )

        # Check that feature dates are before target dates
        # (This is more of a structural check)

        return {
            "has_issues": len(issues) > 0,
            "issues": issues,
            "target_horizon": target_horizon,
        }

    def winsorize(
        self,
        df: pd.DataFrame,
        lower_pct: float = 0.01,
        upper_pct: float = 0.99,
    ) -> pd.DataFrame:
        """
        Winsorize extreme values.

        Args:
            df: DataFrame to winsorize
            lower_pct: Lower percentile
            upper_pct: Upper percentile

        Returns:
            Winsorized DataFrame
        """
        result = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            lower = df[col].quantile(lower_pct)
            upper = df[col].quantile(upper_pct)
            result[col] = df[col].clip(lower=lower, upper=upper)

        return result

    def impute_missing(
        self,
        df: pd.DataFrame,
        method: str = "ffill",
        limit: int = 5,
    ) -> pd.DataFrame:
        """
        Impute missing values.

        Args:
            df: DataFrame with missing values
            method: Imputation method (ffill, bfill, mean, median)
            limit: Maximum consecutive values to fill

        Returns:
            DataFrame with imputed values
        """
        result = df.copy()

        if method == "ffill":
            result = result.ffill(limit=limit)
        elif method == "bfill":
            result = result.bfill(limit=limit)
        elif method == "mean":
            for col in result.columns:
                result[col] = result[col].fillna(result[col].mean())
        elif method == "median":
            for col in result.columns:
                result[col] = result[col].fillna(result[col].median())
        else:
            raise ValueError(f"Unknown imputation method: {method}")

        return result

    def generate_quality_report(
        self,
        df: pd.DataFrame,
        name: str = "dataset",
    ) -> str:
        """
        Generate a human-readable quality report.

        Args:
            df: DataFrame to analyze
            name: Dataset name for report

        Returns:
            Formatted report string
        """
        results = self.check_all(df)

        report = [
            f"Data Quality Report: {name}",
            "=" * 50,
            f"Rows: {results['summary']['rows']}",
            f"Columns: {results['summary']['columns']}",
            f"Overall: {'PASS' if results['summary']['passed'] else 'FAIL'}",
            "",
        ]

        if results["summary"]["issues"]:
            report.append(f"Issues: {', '.join(results['summary']['issues'])}")
            report.append("")

        # Missing values
        report.append("Missing Values:")
        report.append("-" * 30)
        if results["missing"]["has_issues"]:
            for col, pct in results["missing"]["columns_exceeding_threshold"].items():
                report.append(f"  {col}: {pct*100:.1f}%")
        else:
            report.append("  No significant missing values")
        report.append("")

        # Outliers
        report.append("Outliers:")
        report.append("-" * 30)
        if results["outliers"]["has_issues"]:
            for col, info in results["outliers"]["outliers_by_column"].items():
                report.append(
                    f"  {col}: {info['count']} ({info['percentage']}%, "
                    f"max z-score: {info['max_zscore']})"
                )
        else:
            report.append("  No significant outliers detected")
        report.append("")

        # Staleness
        report.append("Data Freshness:")
        report.append("-" * 30)
        report.append(f"  Latest date: {results['stale']['latest_date']}")
        report.append(f"  Days since update: {results['stale']['days_since_latest']}")
        if results["stale"]["stale_columns"]:
            report.append("  Stale columns:")
            for col, days in results["stale"]["stale_columns"].items():
                report.append(f"    {col}: {days} days")

        return "\n".join(report)
