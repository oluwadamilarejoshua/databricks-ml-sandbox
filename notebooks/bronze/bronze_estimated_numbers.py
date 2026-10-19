# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # Bronze — Estimated Numbers
# MAGIC
# MAGIC **Pipeline:** `bronze_estimated_numbers`
# MAGIC **Source:** `/Volumes/<catalog>/default/malaria/estimated_numbers.csv` (WHO malaria estimates, Kaggle)
# MAGIC **Output:** `bronze_data.bronze_estimated_numbers`
# MAGIC **Version:** 1.0.0
# MAGIC
# MAGIC ## Write mode: overwrite, not append
# MAGIC This file is a **full snapshot** — every row of history, every time. Our usual rule is
# MAGIC "Bronze is append-only," but that rule assumes each run brings *new* rows (a new day's
# MAGIC file, a new batch of cases). This source brings the *same whole file* every run, so
# MAGIC `append` would duplicate every row on a second run. This is the same class of bug the
# MAGIC real pipeline shipped and fixed (commit `0b11061`): a full-snapshot source wrongly set
# MAGIC to append. We overwrite here on purpose, not by oversight.

# COMMAND ----------

import uuid
from pyspark.sql import functions as F

dbutils.widgets.text("catalog", "dummyhealthdata")
CATALOG = dbutils.widgets.get("catalog")

BRONZE_SCHEMA = "bronze_data"
PIPELINE_VERSION = "1.0.0"
RUN_ID = str(uuid.uuid4())[:8]   # fresh id every run, not a hardcoded string

SOURCE_PATH = f"/Volumes/{CATALOG}/default/malaria/estimated_numbers.csv"
TARGET_TABLE = f"{CATALOG}.{BRONZE_SCHEMA}.bronze_estimated_numbers"

print(f"Source: {SOURCE_PATH}")
print(f"Target: {TARGET_TABLE}")
print(f"Run id: {RUN_ID}")

# COMMAND ----------

# MAGIC %md ### Cell 1 — Read the raw CSV

# COMMAND ----------

df_raw = (
    spark.read
    .format("csv")
    .option("header", "true")
    .option("inferSchema", "true")
    .load(SOURCE_PATH)
)

print(f"Raw rows: {df_raw.count():,}")
df_raw.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cell 2 — Add lineage columns
# MAGIC `_metadata.file_path` is a hidden column Unity Catalog attaches to every file-based
# MAGIC read — the DataFrame-API way to reach it is `F.col("_metadata.file_path")`, not the
# MAGIC older `input_file_name()` function, which UC blocks because it bypasses governance
# MAGIC and lineage tracking. This is the same field your original SQL already used correctly.

# COMMAND ----------

df_bronze = (
    df_raw
    .withColumn("_source_file", F.col("_metadata.file_path"))
    .withColumn("_ingested_at", F.current_timestamp())
    .withColumn("_pipeline_version", F.lit(PIPELINE_VERSION))
    .withColumn("_run_id", F.lit(RUN_ID))
)

df_bronze.display()

# COMMAND ----------

# MAGIC %md ### Cell 3 — Write to Bronze (overwrite — see note above)

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{BRONZE_SCHEMA}")

(
    df_bronze.write
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .option("delta.columnMapping.mode", "name")
    .saveAsTable(TARGET_TABLE)
)

print(f"Wrote {df_bronze.count():,} rows to {TARGET_TABLE}")

# COMMAND ----------

# MAGIC %md ### Cell 4 — Sanity check

# COMMAND ----------

check = spark.table(TARGET_TABLE)
print(f"Row count: {check.count():,}")
check.select("Country", "Year", "`No. of cases_median`", "_run_id").limit(5).display()
