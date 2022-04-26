# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md # Pipeline Example
# MAGIC 
# MAGIC Let's try putting what we have learned into a simple example pipeline! We will go through and set up a system of notebooks to deploy a model into production and run drift monitoring techniques. 
# MAGIC 
# MAGIC For this example we will need two time periods of data, so that we can demonstrate drift monitoring. We will then create a production model based off of the first time period of data in the pipeline and then finally run our drift monitoring techniques on our simulated second period of data to determine if our production model is still valid. 
# MAGIC 
# MAGIC **Changes between the two datasets:**
# MAGIC * ***The demand for Airbnbs skyrockted, so the prices of Airbnbs doubled***.
# MAGIC   * *Type of Drift*: Concept, Label 
# MAGIC * ***An upstream data management error resulted in null values for `neighbourhood_cleansed`***
# MAGIC   * *Type of Drift*: Feature
# MAGIC * ***An upstream data change resulted in `review_score_rating` move to a 5 star rating system, instead of the previous 100 point system. ***
# MAGIC   * *Type of Drift*: Feature
# MAGIC   
# MAGIC **Note:** There can be permissions issues in some of these notebooks if you are not an admin.

# COMMAND ----------

# MAGIC %md We cannot pass raw data in between notebooks because it is too large. Instead, we will store the data we want at certain paths and then pass the path to the notebook we call instead. 
# MAGIC 
# MAGIC Run the cell below to generate the following variables we will use in this demo:
# MAGIC 
# MAGIC * **`df`**: original time period of airbnb data we will create our production model on.
# MAGIC * **`df2`**: the second time period of data we will drift detect against later, drift simulated as described above.
# MAGIC * **`data_path1`**: The delta path that we can access **`df1`** from.
# MAGIC * **`data_path2`**: The delta path that we can access **`df2`** from. 

# COMMAND ----------

# MAGIC %run "../../Includes/Pipeline-Example-Setup"

# COMMAND ----------

# MAGIC %md ### Data Validation
# MAGIC 
# MAGIC The first step will be to validate the incoming data before we pass it to the rest of the pipeline. 
# MAGIC 
# MAGIC **Note:** We can only pass string data between notebooks, so the result we get out of the notebook is a string casted boolean.

# COMMAND ----------

params = {
    "file_path": data_path1
}
result = dbutils.notebook.run("./01-Data-Validate", 0, params)

result

# COMMAND ----------

# MAGIC %md ### Featurize
# MAGIC 
# MAGIC In this example, we will simply be String Encoding the categorical features and creating our feature vectors.
# MAGIC 
# MAGIC First, we need a new path to store the featurized version of the data, since we do not want to overwrite our original data. 

# COMMAND ----------

data_featurized_path = f"{working_dir}/driftexample/data_featurized"
dbutils.fs.rm(data_featurized_path, True)
dbutils.fs.mkdirs(data_featurized_path)

# COMMAND ----------

# MAGIC %md Now run the featurize notebook. After running this we will have a featurized dataset in the **`data_featurized_path`** ready for training. 

# COMMAND ----------

params = {
    "file_path": data_path1, 
    "save_path": data_featurized_path
}
dbutils.notebook.run("./02-Featurize", 0, params)

# COMMAND ----------

# MAGIC %md ### Train
# MAGIC 
# MAGIC For this notebook, we will pass in the path to the featurized data we want to train on, an experiment path for MLflow, and a registry model name.
# MAGIC 
# MAGIC The notebook will train a model on the data, log the run under the experiment provided, and push the model to the Staging branch in the MLflow model registry.  
# MAGIC 
# MAGIC **Note:** This code involves creating a new MLflow experiment, which could error if this is run in a repo.

# COMMAND ----------

import uuid

experiment_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
registry_model_name = f"pipeline_example_{uuid.uuid4().hex[:6]}"

params = {
    "file_path": data_featurized_path,  
    "experiment_path": experiment_path, 
    "registry_model_name": registry_model_name
}
dbutils.notebook.run("./03-Train", 0, params)

# COMMAND ----------

# MAGIC %md If you look at the MLflow model registry, you can see our model in Staging. 

# COMMAND ----------

# MAGIC %md ### Model Validation
# MAGIC 
# MAGIC This notebook will run some basic checks to make sure it is working well, and then it will push the model into production. In this example, we will make sure the model signature is as expected and that we can successfully generate predictions with the model. 

# COMMAND ----------

params = {
    "file_path": data_featurized_path, 
    "registry_model_name": registry_model_name
}
dbutils.notebook.run("./04-Model-Validate", 0, params)

# COMMAND ----------

# MAGIC %md If you look at the MLflow model registry, our model is now in Production. 

# COMMAND ----------

# MAGIC %md ### Score 
# MAGIC 
# MAGIC Now we want to score on unseen data.

# COMMAND ----------

store_scored_path = f"{working_dir}/driftexample/scored_data"
dbutils.fs.rm(store_scored_path, True)
dbutils.fs.mkdirs(store_scored_path)

params = {
    "file_path": data_featurized_path,
    "save_path": store_scored_path, 
    "registry_model_name": registry_model_name  
}
dbutils.notebook.run("./05-Score", 0, params)

# COMMAND ----------

# MAGIC %md Now we can see our scored featurized dataset with predictions!

# COMMAND ----------

display(spark.read.format("delta").load(store_scored_path))

# COMMAND ----------

# MAGIC %md ### Drift Monitor
# MAGIC 
# MAGIC Now that we have a current production model in the model registry, we would be concerned about the model going stale over time. We will have to be careful to monitor for drift in future datasets. 
# MAGIC 
# MAGIC For this example, let's run this notebook to compare the first time period of data to the more recent one we simulated earlier. 
# MAGIC 
# MAGIC **Note:** We will want the column types, so those are passed in here as well. 

# COMMAND ----------

drift_path = f"{working_dir}/driftexample/data_featurized_drift"

params = {
    "file_path_1": data_featurized_path, 
    "file_path_2": data_path2,
    "drift_path": drift_path
}
dbutils.notebook.run("./06-Monitor", 0, params)

# COMMAND ----------

# MAGIC %md Click on the Notebook job to see the results of the Drift Monitoring!

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
