# Databricks notebook source
# MAGIC %md ### Model Validate
# MAGIC 
# MAGIC This notebook is called from Orchestrate to test the current Staging model and then push it into production.
# MAGIC 
# MAGIC First, let's load the data passed from Orchestrate. 

# COMMAND ----------

dbutils.widgets.text("file_path", "Default")
dbutils.widgets.text("registry_model_name", "Default")

file_path = dbutils.widgets.get("file_path")
registry_model_name = dbutils.widgets.get("registry_model_name")

# COMMAND ----------

# MAGIC %md Now we can load the model in the current Staging branch. 

# COMMAND ----------

import mlflow
from mlflow.tracking import MlflowClient

stage = "Staging"

# pyfunc model to get signature
pyfunc_model = mlflow.pyfunc.load_model(
    model_uri=f"models:/{registry_model_name}/{stage}"
)

# spark model to get predictions
spark_model = mlflow.spark.load_model(
    model_uri=f"models:/{registry_model_name}/{stage}"
)

# COMMAND ----------

# MAGIC %md Next, let's load the data and use the model to make some example predictions on the featurized data. 

# COMMAND ----------

df = spark.read.format("delta").load(file_path)

# COMMAND ----------

display(spark_model.transform(df))

# COMMAND ----------

# MAGIC %md Now, using the signature we logged, we can assert that the input and output schema are as expected.

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, StringType, DoubleType

expected_input_schema = (StructType([
    StructField("accommodates", DoubleType(), True),
    StructField("beds", DoubleType(), True),
    StructField("bedrooms", DoubleType(), True),
    StructField("minimum_nights", DoubleType(), True),
    StructField("number_of_reviews", DoubleType(), True),
    StructField("review_scores_rating", DoubleType(), True),
    StructField("neighbourhood_cleansed", StringType(), True),
    StructField("property_type", StringType(), True),
    StructField("room_type", StringType(), True)                             
]))

# COMMAND ----------

expected_output_schema = StructType([StructField("price", DoubleType(), True)])

input_schema = pyfunc_model.metadata.get_input_schema().as_spark_schema()
output_schema = pyfunc_model.metadata.get_output_schema().as_spark_schema()

# COMMAND ----------

assert expected_input_schema.fields.sort(key=lambda x: x.name) == input_schema.fields.sort(key=lambda x: x.name)
assert expected_output_schema.fields.sort(key=lambda x: x.name) == output_schema.fields.sort(key=lambda x: x.name)

# COMMAND ----------

# MAGIC %md Now that the model passed our example checks, let's promote it to the Production stage.

# COMMAND ----------

client = MlflowClient()
model_version = client.get_latest_versions(registry_model_name, stages=["Staging"])[0].version

# COMMAND ----------

client.transition_model_version_stage(name=registry_model_name, version=model_version, stage="Production", archive_existing_versions=True)

