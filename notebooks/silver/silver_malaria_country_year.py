# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # Silver — Malaria Country-Year
# MAGIC
# MAGIC **Pipeline:** `silver_malaria_country_year`
# MAGIC **Sources:** `bronze_estimated_numbers`, `bronze_incidence_per_1000_pop_at_risk`, `bronze_reported_numbers`
# MAGIC **Output:** `silver_data.silver_malaria_country_year` — one row per (country, year)
# MAGIC **Grain note:** this dataset is annual and country-level, not weekly/ward-level like the
# MAGIC real eHA pipeline. We use Country where the real pipeline uses LGA, and Year where it
# MAGIC uses epi-week. Same concepts, coarser grain.

# COMMAND ----------

from pyspark.sql import functions as F

dbutils.widgets.text("catalog", "dummyhealthdata")
CATALOG = dbutils.widgets.get("catalog")

BRONZE_SCHEMA = "bronze_data"
SILVER_SCHEMA = "silver_data"
PIPELINE_VERSION = "1.0.0"

print(f"Catalog: {CATALOG}")
print(f"Reading from: {CATALOG}.{BRONZE_SCHEMA}")

# COMMAND ----------

# MAGIC %md ### Cell 1 — Read the three Bronze tables

# COMMAND ----------

estimated = spark.table(f"{CATALOG}.{BRONZE_SCHEMA}.bronze_estimated_numbers")
incidence = spark.table(f"{CATALOG}.{BRONZE_SCHEMA}.bronze_incidence_per_1000_pop_at_risk")
reported = spark.table(f"{CATALOG}.{BRONZE_SCHEMA}.bronze_reported_numbers")

print(f"estimated: {estimated.count():,} rows")
print(f"incidence: {incidence.count():,} rows")
print(f"reported:  {reported.count():,} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cell 2 — Clean `estimated`
# MAGIC The raw `No. of cases` / `No. of deaths` columns are compound strings like
# MAGIC `630308[495000-801000]` — the point estimate, followed by its `[min-max]` range.
# MAGIC Every value is either a bare integer (no range to report, e.g. `0`) or exactly
# MAGIC that shape — checked against the real file, no other format exists. We parse the
# MAGIC leading number out directly with a regex rather than substituting the file's
# MAGIC separate `_median` column, so the lineage is "parsed from the named column," not
# MAGIC "borrowed from a different one that happens to agree."

# COMMAND ----------

CASES_PATTERN = r"^(\d+)"

estimated_clean = (
    estimated
    .select(
        F.trim(F.col("Country")).alias("country"),
        F.col("Year").cast("int").alias("year"),  # numeric already — don't trim() it, trim() stringifies it first
        F.trim(F.col("WHO Region")).alias("who_region"),
        F.regexp_extract(F.col("`No. of cases`"), CASES_PATTERN, 1).cast("double").alias("estimated_cases"),
        F.regexp_extract(F.col("`No. of deaths`"), CASES_PATTERN, 1).cast("double").alias("estimated_deaths"),
        # _min/_max are genuinely separate information (the range bounds), not
        # duplicated in the base column, so these still come from their own columns.
        # Where a country had 0 cases, min/max are blank in the source — that's a
        # real 0, not a missing measurement, so we fill it rather than drop the row.
        F.coalesce(F.expr("try_cast(`No. of cases_min` as double)"), F.lit(0.0)).alias("estimated_cases_min"),
        F.coalesce(F.expr("try_cast(`No. of cases_max` as double)"), F.lit(0.0)).alias("estimated_cases_max"),
    )
)

print(f"estimated_clean: {estimated_clean.count():,} rows")
estimated_clean.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cell 2b — Validate the parse against the file's own pre-split median
# MAGIC The file already provides `No. of cases_median`, which should equal the number we
# MAGIC just parsed out of `No. of cases`. If it ever doesn't, our regex assumption about
# MAGIC the column's format is wrong for at least one row, and that should fail loudly here
# MAGIC — not silently feed a wrong number into Silver.

# COMMAND ----------

_mismatch_count = (
    estimated
    .withColumn("_parsed", F.regexp_extract(F.col("`No. of cases`"), CASES_PATTERN, 1).cast("double"))
    .withColumn("_median", F.expr("try_cast(`No. of cases_median` as double)"))
    .filter(F.col("_parsed") != F.col("_median"))
    .count()
)

if _mismatch_count > 0:
    raise ValueError(
        f"{_mismatch_count} rows where parsed 'No. of cases' disagrees with "
        f"'No. of cases_median' — the regex assumption about this column's format is wrong."
    )

print("Validated: parsed 'No. of cases' matches 'No. of cases_median' for every row.")

# COMMAND ----------

# MAGIC %md ### Cell 3 — Clean `incidence`
# MAGIC Despite the column name, `No. of cases` in this file is actually the incidence rate
# MAGIC per 1,000 population at risk — a continuous number. This becomes our regression target.

# COMMAND ----------

incidence_clean = (
    incidence
    .select(
        F.trim(F.col("Country")).alias("country"),
        F.col("Year").cast("int").alias("year"),
        F.trim(F.col("WHO Region")).alias("who_region"),
        F.expr("try_cast(`No. of cases` as double)").alias("incidence_per_1000"),
    )
)

print(f"incidence_clean: {incidence_clean.count():,} rows")
incidence_clean.display()

# COMMAND ----------

# MAGIC %md ### Cell 4 — Clean `reported`

# COMMAND ----------

reported_clean = (
    reported
    .select(
        F.trim(F.col("Country")).alias("country"),
        F.col("Year").cast("int").alias("year"),
        F.trim(F.col("WHO Region")).alias("who_region"),
        F.expr("try_cast(`No. of cases` as double)").alias("reported_cases"),
        F.expr("try_cast(`No. of deaths` as double)").alias("reported_deaths"),
    )
)

print(f"reported_clean: {reported_clean.count():,} rows")
reported_clean.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cell 5 — Join on (country, year)
# MAGIC `incidence` is our target table, so it drives the join — every row we can score
# MAGIC against must have an incidence value. `estimated` and `reported` are joined in as
# MAGIC extra features; a country-year missing from either becomes a null feature, not a
# MAGIC dropped row. We print counts at each step so a silent row-loss would be visible.

# COMMAND ----------

join_keys = ["country", "year"]

silver_df = (
    incidence_clean
    .join(estimated_clean.drop("who_region"), on=join_keys, how="left")
    .join(reported_clean.drop("who_region"), on=join_keys, how="left")
)

print(f"incidence_clean rows:  {incidence_clean.count():,}")
print(f"after join (should match incidence_clean): {silver_df.count():,}")

missing_estimated = silver_df.filter(F.col("estimated_cases").isNull()).count()
missing_reported = silver_df.filter(F.col("reported_cases").isNull()).count()
print(f"rows missing estimated features: {missing_estimated:,}")
print(f"rows missing reported features:  {missing_reported:,}")

# COMMAND ----------

# MAGIC %md ### Cell 6 — Add lineage columns and write

# COMMAND ----------

silver_final = (
    silver_df
    .withColumn("_produced_at", F.current_timestamp())
    .withColumn("_pipeline_version", F.lit(PIPELINE_VERSION))
)

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SILVER_SCHEMA}")

target_table = f"{CATALOG}.{SILVER_SCHEMA}.silver_malaria_country_year"

# This is a full rebuild from Bronze every run, not an incremental append — so it
# overwrites. (The real pipeline hit exactly this bug the other way: a full-snapshot
# source that was wrongly set to append, silently duplicating history.)
(
    silver_final.write
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(target_table)
)

print(f"Wrote {silver_final.count():,} rows to {target_table}")

# COMMAND ----------

# MAGIC %md ### Cell 7 — Sanity check

# COMMAND ----------

check = spark.table(target_table)
print(f"Row count: {check.count():,}")
print(f"Year range: {check.agg(F.min('year'), F.max('year')).first()}")
print(f"Distinct countries: {check.select('country').distinct().count()}")
check.orderBy(F.col("incidence_per_1000").desc()).limit(10).display()


# COMMAND ----------

