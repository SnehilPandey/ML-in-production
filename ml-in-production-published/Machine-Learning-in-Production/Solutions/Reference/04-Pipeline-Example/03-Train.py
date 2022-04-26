# Databricks notebook source
# MAGIC %md ### Train
# MAGIC 
# MAGIC This notebook is called by Orchestrate to train a model on our first time period of featurized data. 
# MAGIC 
# MAGIC Let's first import the packages we need.

# COMMAND ----------

from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature

# COMMAND ----------

# MAGIC %md Now, let's load the data from Orchestrate. 

# COMMAND ----------

dbutils.widgets.text("file_path", "Default")
dbutils.widgets.text("experiment_path", "Default")
dbutils.widgets.text("registry_model_name", "Default")

file_path = dbutils.widgets.get("file_path")
experiment_path = dbutils.widgets.get("experiment_path")
registry_model_name = dbutils.widgets.get("registry_model_name")

# COMMAND ----------

# MAGIC %md Set the experiment from the data we loaded. 

# COMMAND ----------

mlflow.set_experiment(experiment_path)

# COMMAND ----------

# MAGIC %md Load the data to train. 

# COMMAND ----------

df = spark.read.format("delta").load(file_path)
train_df, test_df = df.randomSplit([0.9, 0.1], seed=42)

# COMMAND ----------

# MAGIC %md Create our example model, making sure to log signature, input example, train mse, and test mse. 

# COMMAND ----------

with mlflow.start_run() as run:
    params = {
        "numTrees": 100,
        "cacheNodeIds": True,
        "maxDepth": 30,
        "minInstancesPerNode": 100,
        "maxBins": 40
      }

    # Need get all columns that are not price, Index
    cols = [field for (field, data_type) in train_df.dtypes if (field != "price") and ("_index" not in field) and (data_type != "vector")]
    input_example = train_df.select(cols).head(3)
    sig = infer_signature(train_df.select(cols), train_df.select("price"))

    regressor = RandomForestRegressor(featuresCol="features", labelCol="price", **params)

    rf_model = regressor.fit(train_df)

    evaluator = RegressionEvaluator(metricName="mse", labelCol="price")

    train_mse = evaluator.evaluate(rf_model.transform(train_df))
    test_mse = evaluator.evaluate(rf_model.transform(test_df))

    mlflow.log_metric("test_mse", test_mse)
    mlflow.log_metric("train_mse", train_mse)
    mlflow.spark.log_model(rf_model, "model", input_example=input_example, signature=sig)

# COMMAND ----------

# MAGIC %md Finally, we'll push our model to the Staging branch in the model registry. 

# COMMAND ----------

client = MlflowClient()
model_version = mlflow.register_model(model_uri=f"runs:/{run.info.run_id}/model", name=registry_model_name)
model_version = client.transition_model_version_stage(
    name=model_version.name, 
    version=model_version.version, 
    stage="Staging", 
    archive_existing_versions=True
)

