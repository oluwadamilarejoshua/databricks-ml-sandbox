"""
dq_checks.py
------------
Reusable data quality check functions for Malaria Forecast Bronze pipelines.

Every Bronze notebook must call at minimum:
    - check_null_thresholds()  -- reject if critical columns exceed null rate
    - check_min_rows()         -- reject if DataFrame is suspiciously small
    - check_no_duplicates()    -- warn if duplicate rows found

All check failures raise ValueError by default. Callers can pass
raise_on_failure=False to get a warning instead (useful for non-critical columns).
"""

import logging
from typing import Dict, List, Optional

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

logger = logging.getLogger(__name__)


def check_null_thresholds(
    df: DataFrame,
    critical_columns: Dict[str, float],
    raise_on_failure: bool = True,
) -> Dict[str, float]:
    """
    Check that critical columns do not exceed their maximum allowed null rate.

    Args:
        df:               Input Spark DataFrame.
        critical_columns: Dict mapping column name to maximum allowed null fraction.
                          Example: {'ward': 0.10, 'lga': 0.10, 'report_date': 0.05}
                          A threshold of 0.10 means: fail if >10% of values are null.
        raise_on_failure: If True, raise ValueError on breach. If False, log a warning.

    Returns:
        Dict mapping column name to actual null fraction observed.

    Raises:
        ValueError: If any column exceeds its null threshold and raise_on_failure=True.
        KeyError:   If a column in critical_columns does not exist in the DataFrame.
    """
    total_rows = df.count()
    if total_rows == 0:
        raise ValueError("DataFrame is empty -- cannot run null threshold checks.")

    # Build a single-pass aggregation for all columns
    agg_exprs = [
        F.count(F.when(F.col(col).isNull(), 1)).alias(col)
        for col in critical_columns
    ]
    null_counts = df.agg(*agg_exprs).first().asDict()

    null_fractions = {}
    breaches = []

    for col, max_null_frac in critical_columns.items():
        null_frac = null_counts[col] / total_rows
        null_fractions[col] = null_frac

        if null_frac > max_null_frac:
            msg = (
                f"DQ FAIL -- column '{col}': null fraction {null_frac:.1%} "
                f"exceeds threshold {max_null_frac:.1%} "
                f"({null_counts[col]:,} nulls out of {total_rows:,} rows)"
            )
            breaches.append(msg)

    if breaches:
        breach_text = "\n".join(breaches)
        logger.error(f"DQ check failures:\n{breach_text}")
        if raise_on_failure:
            raise ValueError(f"Bronze DQ checks failed:\n{breach_text}")
    else:
        logger.info(
            f"DQ PASS -- null threshold checks passed for: {list(critical_columns.keys())}. "
            f"Total rows: {total_rows:,}"
        )

    return null_fractions


def check_min_rows(
    df: DataFrame,
    min_rows: int,
    context: str = "",
    raise_on_failure: bool = True,
) -> int:
    """
    Reject the pipeline run if the DataFrame has fewer rows than expected.
    Protects against empty exports, failed API calls, or corrupt files.

    Args:
        df:               Input Spark DataFrame.
        min_rows:         Minimum acceptable row count.
        context:          Optional label for log messages (e.g. "Malaria cases Abuja").
        raise_on_failure: Raise ValueError or just warn.

    Returns:
        Actual row count.
    """
    actual = df.count()
    label = f"[{context}] " if context else ""

    if actual < min_rows:
        msg = (
            f"{label}DQ FAIL -- row count {actual:,} is below minimum {min_rows:,}. "
            "The source file may be empty or the API call may have returned partial data."
        )
        logger.error(msg)
        if raise_on_failure:
            raise ValueError(msg)
    else:
        logger.info(f"{label}DQ PASS -- row count {actual:,} >= minimum {min_rows:,}")

    return actual


def check_no_duplicates(
    df: DataFrame,
    key_columns: List[str],
    raise_on_failure: bool = False,
) -> int:
    """
    Check for duplicate rows based on a set of key columns.
    Bronze tables are append-only so duplicates from re-runs are expected,
    but duplicates within a single ingestion file should be flagged.

    Args:
        df:               Input Spark DataFrame (before writing to Bronze).
        key_columns:      List of columns that together form a unique row key.
        raise_on_failure: Default False -- duplicate detection raises a warning, not an error.

    Returns:
        Number of duplicate rows found.
    """
    total = df.count()
    distinct = df.dropDuplicates(key_columns).count()
    n_dupes = total - distinct

    if n_dupes > 0:
        msg = (
            f"DQ WARN -- {n_dupes:,} duplicate rows found on key columns: {key_columns}. "
            f"Total: {total:,}, Distinct: {distinct:,}."
        )
        logger.warning(msg)
        if raise_on_failure:
            raise ValueError(msg)
    else:
        logger.info(f"DQ PASS -- No duplicates found on key columns: {key_columns}")

    return n_dupes


def check_date_range(
    df: DataFrame,
    date_column: str,
    min_date: str,
    max_date: Optional[str] = None,
    raise_on_failure: bool = False,
) -> Dict[str, str]:
    """
    Verify that dates in a column fall within an expected range.
    Flags records with dates before min_date or after max_date.

    Args:
        df:               Input Spark DataFrame.
        date_column:      Name of the date column to check.
        min_date:         Earliest acceptable date (ISO format: YYYY-MM-DD).
        max_date:         Latest acceptable date. Defaults to today's date.
        raise_on_failure: Raise ValueError or just warn.

    Returns:
        Dict with 'min_observed', 'max_observed', 'out_of_range_count' keys.
    """
    from datetime import date

    if max_date is None:
        max_date = date.today().isoformat()

    agg = df.agg(
        F.min(F.col(date_column)).alias("min_date"),
        F.max(F.col(date_column)).alias("max_date"),
        F.count(F.when(F.col(date_column) < F.lit(min_date), 1)).alias("below_min"),
        F.count(F.when(F.col(date_column) > F.lit(max_date), 1)).alias("above_max"),
    ).first()

    result = {
        "min_observed": str(agg["min_date"]),
        "max_observed": str(agg["max_date"]),
        "out_of_range_count": agg["below_min"] + agg["above_max"],
    }

    if result["out_of_range_count"] > 0:
        msg = (
            f"DQ WARN -- {result['out_of_range_count']:,} rows have dates outside "
            f"[{min_date}, {max_date}]. Observed range: "
            f"[{result['min_observed']}, {result['max_observed']}]"
        )
        logger.warning(msg)
        if raise_on_failure:
            raise ValueError(msg)
    else:
        logger.info(
            f"DQ PASS -- All dates in [{result['min_observed']}, {result['max_observed']}] "
            f"are within expected range [{min_date}, {max_date}]"
        )

    return result


def log_dq_summary(
    df: DataFrame,
    table_name: str,
    pipeline_version: str,
    null_fractions: Optional[Dict[str, float]] = None,
) -> None:
    """
    Print a structured DQ summary block to the notebook output.
    Visible in Databricks notebook run history.

    Args:
        df:               Input Spark DataFrame (post-all-checks).
        table_name:       Target Bronze table name.
        pipeline_version: Pipeline semantic version.
        null_fractions:   Optional dict of null fraction results from check_null_thresholds().
    """
    row_count = df.count()
    col_count = len(df.columns)

    print("=" * 60)
    print(f"  DQ SUMMARY -- {table_name}")
    print("=" * 60)
    print(f"  Pipeline version : {pipeline_version}")
    print(f"  Row count        : {row_count:,}")
    print(f"  Column count     : {col_count}")
    if null_fractions:
        print("  Null fractions   :")
        for col, frac in null_fractions.items():
            print(f"    {col:30s}  {frac:.2%}")
    print("=" * 60)
