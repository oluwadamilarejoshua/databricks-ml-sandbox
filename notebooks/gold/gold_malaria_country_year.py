# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # Gold — Malaria Country-Year (Model-Ready)
# MAGIC
# MAGIC **Pipeline:** `gold_malaria_country_year`
# MAGIC **Source:** `silver_data.silver_malaria_country_year`
# MAGIC **Output:** `gold_data.gold_malaria_country_year` — one row per (country, year)
# MAGIC **Version:** 1.0.0
# MAGIC
# MAGIC ## Why this is lag-based, not same-year
# MAGIC `estimated_cases`, `estimated_deaths`, `reported_cases`, `reported_deaths` and
# MAGIC `incidence_per_1000` are all restatements of the same underlying thing -- how much
# MAGIC malaria a country had that year. Feeding this year's case counts in as a "feature"
# MAGIC to predict this year's incidence would be leakage, not modelling: the model would be
# MAGIC learning an accounting identity, not an epidemiological pattern.
# MAGIC
# MAGIC So every feature here is a **prior-year** (lagged) value of these signals. Only
# MAGIC `who_region` is safe as a same-year feature, because it doesn't move with the target.
# MAGIC
# MAGIC ## Two kinds of columns in this table -- do not mix them up later
# MAGIC - **Features (safe):** `who_region`, and every `_lag1y` / `_lag2y` / `_roll2y_avg` column.
# MAGIC - **Same-year raw values (NOT features):** `incidence_per_1000` (the regression target),
# MAGIC   `outbreak_flag` (the classification target), and the same-year count columns kept
# MAGIC   here as *candidate alternative targets* for count-based models (Poisson / Negative
# MAGIC   Binomial) -- not as predictors of anything.

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window

dbutils.widgets.text("catalog", "dummyhealthdata")
CATALOG = dbutils.widgets.get("catalog")

SILVER_SCHEMA = "silver_data"
GOLD_SCHEMA = "gold_data"
PIPELINE_VERSION = "1.0.0"

LAG_YEARS = [1, 2]
MIN_YEARS_HISTORY = 15   # a country needs at least this many Silver rows to be modelled at all
OUTBREAK_PERCENTILE = 0.90   # per-country, not global -- see Cell 4

SOURCE_TABLE = f"{CATALOG}.{SILVER_SCHEMA}.silver_malaria_country_year"
TARGET_TABLE = f"{CATALOG}.{GOLD_SCHEMA}.gold_malaria_country_year"

print(f"Source: {SOURCE_TABLE}")
print(f"Target: {TARGET_TABLE}")
print(f"Lag years: {LAG_YEARS}")
print(f"Minimum years of history to keep a country: {MIN_YEARS_HISTORY}")

# COMMAND ----------

# MAGIC %md ### Cell 1 — Read Silver

# COMMAND ----------

silver = spark.table(SOURCE_TABLE)
print(f"Silver rows: {silver.count():,}")
print(f"Distinct countries: {silver.select('country').distinct().count():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cell 2 — Minimum-history guard
# MAGIC Drop any country with fewer than `MIN_YEARS_HISTORY` rows entirely, before computing
# MAGIC any lag feature. A country just below the line would otherwise produce a handful of
# MAGIC rows that are almost all null lags -- noise, not signal. This mirrors the real
# MAGIC pipeline's `MIN_WEEKS_TO_TRAIN` guard, applied per-country instead of pipeline-wide.

# COMMAND ----------

years_per_country = silver.groupBy("country").agg(F.count("*").alias("n_years"))

dropped_countries = (
    years_per_country
    .filter(F.col("n_years") < MIN_YEARS_HISTORY)
    .select("country", "n_years")
    .collect()
)

if dropped_countries:
    print(f"Dropping {len(dropped_countries)} countries below {MIN_YEARS_HISTORY} years of history:")
    for row in dropped_countries:
        print(f"  {row['country']}: only {row['n_years']} years")
else:
    print(f"No countries dropped -- every country has at least {MIN_YEARS_HISTORY} years of history.")

kept_countries = years_per_country.filter(F.col("n_years") >= MIN_YEARS_HISTORY).select("country")
silver_filtered = silver.join(kept_countries, on="country", how="inner")

print(f"Rows after filter: {silver_filtered.count():,} (was {silver.count():,})")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cell 3 — Lag and rolling features
# MAGIC Every lag is computed per-country, ordered by year -- `Window.partitionBy("country")`
# MAGIC is what keeps one country's history from bleeding into another's. The first
# MAGIC `len(LAG_YEARS)` years of each country's series will have null lag values -- there's
# MAGIC genuinely nothing before them to look back on. We leave those nulls in Gold rather
# MAGIC than drop the rows; the training notebook decides how to handle them, same split of
# MAGIC responsibility as the real pipeline.

# COMMAND ----------

display(spark.table("dummyhealthdata.silver_data.silver_malaria_country_year").columns)

# COMMAND ----------

country_year_window = Window.partitionBy("country").orderBy("year")

LAG_SOURCE_COLS = ["incidence_per_1000", "estimated_cases", "reported_cases"]

gold_df = silver_filtered
for col in LAG_SOURCE_COLS:
    for lag_n in LAG_YEARS:
        gold_df = gold_df.withColumn(
            f"{col}_lag{lag_n}y",
            F.lag(F.col(col), lag_n).over(country_year_window),
        )

# A 2-year rolling average of incidence, looking back only -- rowsBetween(-2, -1) means
# "the two rows before this one," excluding the current (same-year) row on purpose.
rolling_window = country_year_window.rowsBetween(-2, -1)
gold_df = gold_df.withColumn(
    "incidence_roll2y_avg",
    F.avg(F.col("incidence_per_1000")).over(rolling_window),
)

print("Lag/rolling columns added:")
for c in gold_df.columns:
    if "_lag" in c or "_roll" in c:
        print(f"  {c}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cell 4 — Outbreak flag, per-country percentile
# MAGIC A single global threshold across all 107 countries would just permanently flag
# MAGIC high-burden countries and never flag low-burden ones -- it would detect "which
# MAGIC country is this" instead of "is this an unusual year for this country." The
# MAGIC threshold has to be computed within each country's own history.

# COMMAND ----------

country_window_unordered = Window.partitionBy("country")

gold_df = gold_df.withColumn(
    "incidence_p90_country",
    F.expr(f"percentile_approx(incidence_per_1000, {OUTBREAK_PERCENTILE})").over(country_window_unordered),
)
gold_df = gold_df.withColumn(
    "outbreak_flag",
    F.when(F.col("incidence_per_1000") > F.col("incidence_p90_country"), 1).otherwise(0),
)

outbreak_rate = gold_df.agg(F.avg("outbreak_flag")).first()[0]
expected_rate = 1 - OUTBREAK_PERCENTILE
print(f"outbreak_flag positive rate: {outbreak_rate:.1%} (expected close to {expected_rate:.0%} by construction)")

# COMMAND ----------

# MAGIC %md ### Cell 5 — Data-quality checks

# COMMAND ----------

# sys/os/utils are imported here, not at file top, on purpose: this cell's only job is
# resolving REPO_ROOT so utils/ becomes importable, and it belongs next to the code that
# uses it -- same structure the real repo's notebooks use.
import sys  # noqa: E402
import os  # noqa: E402

try:
    REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
except NameError:
    _nb_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
    REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname("/Workspace" + _nb_path)))
sys.path.insert(0, REPO_ROOT)

from utils.dq_checks import check_null_thresholds, check_no_duplicates  # noqa: E402

null_fractions = check_null_thresholds(
    gold_df,
    {"country": 0.0, "year": 0.0, "incidence_per_1000": 0.0, "who_region": 0.0},
    raise_on_failure=True,
)

# A duplicate (country, year) in Gold means the join upstream fanned out -- a real bug,
# not tolerable noise, so this raises (unlike Bronze's default).
check_no_duplicates(gold_df, key_columns=["country", "year"], raise_on_failure=True)

# COMMAND ----------

# MAGIC %md ### Cell 6 — Add lineage columns and write

# COMMAND ----------

gold_final = (
    gold_df
    .withColumn("_produced_at", F.current_timestamp())
    .withColumn("_pipeline_version", F.lit(PIPELINE_VERSION))
)

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{GOLD_SCHEMA}")

(
    gold_final.write
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(TARGET_TABLE)
)

print(f"Wrote {gold_final.count():,} rows to {TARGET_TABLE}")

# COMMAND ----------

# MAGIC %md ### Cell 7 — Sanity check

# COMMAND ----------

check = spark.table(TARGET_TABLE)
print(f"Row count: {check.count():,}")
print(f"Distinct countries: {check.select('country').distinct().count():,}")

rows_with_full_lags = check.filter(F.col("incidence_per_1000_lag2y").isNotNull()).count()
print(f"Rows with both lag years available: {rows_with_full_lags:,} (modellable rows)")

print("outbreak_flag distribution:")
check.groupBy("outbreak_flag").count().display()

check.select(
    "country", "year", "incidence_per_1000",
    "incidence_per_1000_lag1y", "incidence_per_1000_lag2y",
    "incidence_roll2y_avg", "outbreak_flag",
).orderBy("country", "year").limit(15).display()