# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC # Lab: Deploying a Model in Batch
# MAGIC Deploying a model via batch is the preferred solution for most machine learning applications. In this lab, you will scale the deployment of a single-node model via Spark UDFs and MLflow's **`pyfunc`**. 
# MAGIC 
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lab you:<br>
# MAGIC  - Develop and register an MLflow model
# MAGIC  - Deploy the model as a Spark UDF
# MAGIC  - Optimize the predictions for reading using Delta

# COMMAND ----------

# MAGIC %run ../../Includes/Classroom-Setup

# COMMAND ----------

df = (spark
      .read
      .parquet("/mnt/training/airbnb/sf-listings/airbnb-cleaned-mlflow.parquet/")
      .select("price", "bathrooms", "bedrooms", "number_of_reviews")
     )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Develop and Register an MLflow Model
# MAGIC 
# MAGIC In this exercise, you will build, log, and register an XGBoost model using Scikit-learn and MLflow.
# MAGIC 
# MAGIC This model will predict the **`price`** variable using **`bathrooms`**, **`bedrooms`**, and **`number_of_reviews`** as features.

# COMMAND ----------

import xgboost as xgb
import mlflow
import mlflow.xgboost
from sklearn.metrics import mean_squared_error
import uuid

# Start run
with mlflow.start_run(run_name="xgboost_model") as run:
    train_pdf = df.toPandas()
    X_train = train_pdf.drop(["price"], axis=1)
    y_train = train_pdf["price"]

    # Train model
    n_estimators = 10
    max_depth = 5
    regressor = xgb.XGBRegressor(n_estimators=n_estimators, max_depth=max_depth)
    regressor.fit(X_train, y_train)

    # Evaluate model
    predictions = regressor.predict(X_train)
    rmse = mean_squared_error(predictions, y_train, squared=False)

    # Log params and metric
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_metric("rmse", rmse)

    # Log model
    mlflow.xgboost.log_model(regressor, "xgboost-model")
    
# Register model
model_name = f"xgboost_model_{uuid.uuid4().hex[:6]}"
model_uri = f"runs:/{run.info.run_id}/xgboost-model"
model_details = mlflow.register_model(model_uri=model_uri, name=model_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Deploy Model as a Spark UDF
# MAGIC 
# MAGIC Next, you will compute predictions for your model using a Spark UDF.

# COMMAND ----------

# TODO
# Create the prediction UDF
predict = #FILL_IN

# Compute the predictions
prediction_df = df.withColumn("prediction", #FILL_IN)
             

# View the results
display(prediction_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) Next Lesson<br>
# MAGIC 
# MAGIC Start the next lesson, [Real Time]($../02-Real-Time )

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
