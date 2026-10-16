# Databricks notebook source
# MAGIC %md
# MAGIC # Bronze — Reported Numbers
# MAGIC
# MAGIC **Pipeline:** `bronze_reported_numbers`
# MAGIC **Source:** `/Volumes/<catalog>/default/malaria/reported_numbers.csv`
# MAGIC **Output:** `bronze_data.bronze_reported_numbers`
# MAGIC **Version:** 1.0.0
# MAGIC **Write mode:** overwrite — full snapshot, same reasoning as `bronze_estimated_numbers`.

# COMMAND ----------

import uuid
from pyspark.sql import functions as F

dbutils.widgets.text("catalog", "dummyhealthdata")
CATALOG = dbutils.widgets.get("catalog")

BRONZE_SCHEMA = "bronze_data"
PIPELINE_VERSION = "1.0.0"
RUN_ID = str(uuid.uuid4())[:8]

SOURCE_PATH = f"/Volumes/{CATALOG}/default/malaria/reported_numbers.csv"
TARGET_TABLE = f"{CATALOG}.{BRONZE_SCHEMA}.bronze_reported_numbers"

print(f"Source: {SOURCE_PATH}")
print(f"Target: {TARGET_TABLE}")

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

df_bronze = (
    df_raw
    .withColumn("_source_file", F.col("_metadata.file_path"))
    .withColumn("_ingested_at", F.current_timestamp())
    .withColumn("_pipeline_version", F.lit(PIPELINE_VERSION))
    .withColumn("_run_id", F.lit(RUN_ID))
)
df_bronze.display()

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
# MAGIC %md ### Sanity check

# COMMAND ----------

check = spark.table(TARGET_TABLE)
print(f"Row count: {check.count():,}")
check.limit(5).display()
