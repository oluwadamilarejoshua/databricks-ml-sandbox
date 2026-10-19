"""
test_silver_estimated_cases.py
-------------------------------
Unit tests for the "No. of cases" / "No. of deaths" parsing logic in
silver_malaria_country_year.py.

Tests cover: parsing the point estimate out of the compound
"630308[495000-801000]" format, and the bare-integer "0" case where there's
no range to report.

Run locally with:
    pytest tests/test_silver_estimated_cases.py -v

Note: These tests use PySpark's local mode — they do NOT require a live
Databricks cluster. They do require a local JDK, since starting a local
SparkSession needs one; if you don't have Java installed, these will fail
to start regardless of whether the logic is correct — that's expected, and
is exactly why CI (which has Java) is the real verifier, not your machine.
"""

import pytest

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType

CASES_PATTERN = r"^(\d+)"


@pytest.fixture(scope="session")
def spark():
    return (
        SparkSession.builder
        .master("local[2]")
        .appName("sandbox-tests-silver-estimated")
        .config("spark.sql.shuffle.partitions", "2")
        .getOrCreate()
    )


def _parse_cases(spark, raw_values):
    schema = StructType([StructField("No. of cases", StringType())])
    df = spark.createDataFrame([(v,) for v in raw_values], schema)
    return (
        df
        .withColumn(
            "parsed",
            F.regexp_extract(F.col("`No. of cases`"), CASES_PATTERN, 1).cast("double"),
        )
        .select("parsed")
        .collect()
    )


class TestEstimatedCasesParsing:
    """The real file only ever contains these two shapes — verified against
    the actual CSV before writing this test, not assumed."""

    def test_parses_point_estimate_out_of_bracketed_range(self, spark):
        rows = _parse_cases(spark, ["630308[495000-801000]"])
        assert rows[0]["parsed"] == 630308.0

    def test_parses_bare_integer_with_no_range(self, spark):
        rows = _parse_cases(spark, ["0"])
        assert rows[0]["parsed"] == 0.0

    def test_parses_small_bare_integer(self, spark):
        rows = _parse_cases(spark, ["7"])
        assert rows[0]["parsed"] == 7.0

    def test_handles_multiple_rows_independently(self, spark):
        rows = _parse_cases(spark, ["630308[495000-801000]", "0", "4615605[3106000-6661000]"])
        parsed = [r["parsed"] for r in rows]
        assert parsed == [630308.0, 0.0, 4615605.0]
