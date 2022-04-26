# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md # Data Management
# MAGIC 
# MAGIC Production machine learning solutions start with reproducible data management. Strategies that we'll cover in this notebook include <a href="https://docs.databricks.com/delta/versioning.html" target="_blank">Delta Table Versioning</a> and working with the <a href="https://docs.databricks.com/applications/machine-learning/feature-store.html" target="_blank">Feature Store</a>.
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lesson you will:<br>
# MAGIC  - Version tables with Delta
# MAGIC  - Programmatically log Feature Tables

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Management and Reproducibility
# MAGIC 
# MAGIC Managing the machine learning lifecycle means...<br>
# MAGIC 
# MAGIC * Reproducibility of data
# MAGIC * Reproducibility of code
# MAGIC * Reproducibility of models
# MAGIC * Automated integration with production systems
# MAGIC 
# MAGIC **We'll begin with data management,** which can be accomplished in a number of ways including:<br>
# MAGIC 
# MAGIC - Saving a snapshot of your data
# MAGIC - Table versioning and time travel using Delta
# MAGIC - Using a feature table

# COMMAND ----------

# MAGIC %md
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) Classroom-Setup
# MAGIC 
# MAGIC For each lesson to execute correctly, please make sure to run the **`Classroom-Setup`** cell at the start of each lesson.

# COMMAND ----------

# MAGIC %run "../Includes/Classroom-Setup"

# COMMAND ----------

# MAGIC %md Let's load in our data and generate a unique ID for each listing.

# COMMAND ----------

from pyspark.sql.functions import monotonically_increasing_id 

airbnb_df = (spark
             .read
             .format("delta")
             .load("/mnt/training/airbnb/sf-listings/sf-listings-2019-03-06-clean.delta/")
             .withColumn("index", monotonically_increasing_id())
            )

display(airbnb_df)

# COMMAND ----------

# MAGIC %md ## Versioning with Delta Tables
# MAGIC 
# MAGIC Let's start by writing to a new Delta Table.

# COMMAND ----------

dbutils.fs.rm(working_dir, recurse=True) # In case run more than once

airbnb_df.write.format("delta").save(working_dir)

# COMMAND ----------

# MAGIC %md Now let's read our Delta Table and modify it, dropping the **`cancellation_policy`** and **`instant_bookable`** columns.

# COMMAND ----------

delta_df = (spark
            .read
            .format("delta")
            .load(working_dir)
            .drop("cancellation_policy", "instant_bookable")
           )

display(delta_df)

# COMMAND ----------

# MAGIC %md Now we can **`overwrite`** our Delta Table using the **`mode`** parameter.

# COMMAND ----------

delta_df.write.format("delta").mode("overwrite").save(working_dir)

# COMMAND ----------

# MAGIC %md Whoops! We actually wanted to keep the **`cancellation_policy`** column. Luckily we can use data versioning to return to an older version of this table. 
# MAGIC 
# MAGIC Start by using the **`DESCRIBE HISTORY`** SQL command.

# COMMAND ----------

display(spark.sql(f"DESCRIBE HISTORY delta.`{working_dir}`"))

# COMMAND ----------

# MAGIC %md As we can see in the **`operationParameters`** column in version 1, we overwrote the table. We now need to travel back in time to load in version 0 to get all the original columns, then we can delete just the **`instant_bookable`** column.

# COMMAND ----------

delta_df = spark.read.format("delta").option("versionAsOf", 0).load(working_dir)

display(delta_df)

# COMMAND ----------

# MAGIC %md You can also query based upon timestamp.  
# MAGIC 
# MAGIC **Note that the ability to query an older snapshot of a table (time travel) is lost after running <a href="https://docs.databricks.com/delta/delta-batch.html#deltatimetravel" target="_blank">a VACUUM command.</a>**

# COMMAND ----------

timestamp = spark.sql(f"DESCRIBE HISTORY delta.`{working_dir}`").orderBy("version").first().timestamp

display(spark
        .read
        .format("delta")
        .option("timestampAsOf", timestamp)
        .load(working_dir)
       )

# COMMAND ----------

# MAGIC %md Now we can drop **`instant_bookable`** and overwrite the table.

# COMMAND ----------

delta_df.drop("instant_bookable").write.format("delta").mode("overwrite").save(working_dir)

# COMMAND ----------

# MAGIC %md Version 2 is our latest and most accurate table version.

# COMMAND ----------

display(spark.sql(f"DESCRIBE HISTORY delta.`{working_dir}`"))

# COMMAND ----------

# MAGIC %md ## Feature Store
# MAGIC 
# MAGIC A <a href="https://docs.databricks.com/applications/machine-learning/feature-store.html#databricks-feature-store" target="_blank">feature store</a> is a **centralized repository of features.** It enables feature **sharing and discovery across** your organization and also ensures that the same feature computation code is used for model training and inference.

# COMMAND ----------

# MAGIC %md %md Create a new database and unique table name (in case you re-run the notebook multiple times)

# COMMAND ----------

import uuid

spark.sql(f"CREATE DATABASE IF NOT EXISTS {clean_username}")
table_name = f"{clean_username}.airbnb_" + str(uuid.uuid4())[:6]

print(table_name)

# COMMAND ----------

# MAGIC %md Let's start creating a <a href="https://docs.databricks.com/applications/machine-learning/feature-store.html#create-a-feature-table-in-databricks-feature-store" target="_blank">Feature Store Client</a> so we can populate our feature store.

# COMMAND ----------

from databricks import feature_store

fs = feature_store.FeatureStoreClient()

help(fs.create_table)

# COMMAND ----------

# MAGIC %md #### Create Feature Table
# MAGIC 
# MAGIC Next, we can create the Feature Table using the **`create_table`** method.
# MAGIC 
# MAGIC This method takes a few parameters as inputs:
# MAGIC * **`name`** - A feature table name of the form **<database_name>.<table_name>`**
# MAGIC * **`primary_keys`** - The primary key(s). If multiple columns are required, specify a list of column names.
# MAGIC * **`df`** - Data to insert into this feature table.  The schema of **`features_df`** will be used as the feature table schema.
# MAGIC * **`schema`** - Feature table schema. Note that either **`schema`** or **`features_df`** must be provided.
# MAGIC * **`description`** - Description of the feature table
# MAGIC * **`partition_columns`** - Column(s) used to partition the feature table.

# COMMAND ----------

fs.create_table(
    name=table_name,
    primary_keys=["index"],
    df=airbnb_df,
    partition_columns=["neighbourhood_cleansed"],
    description="Original Airbnb data"
)

# COMMAND ----------

# MAGIC %md
# MAGIC Alternatively, you can **`create_table`** with schema only (without **`df`**), and populate data to the feature table with **`fs.write_table`**, **`fs.write_table`** has both **`overwrite`** and **`merge`** mode.
# MAGIC 
# MAGIC Example:
# MAGIC 
# MAGIC ```
# MAGIC fs.create_table(
# MAGIC     name=table_name,
# MAGIC     primary_keys=["index"],
# MAGIC     schema=airbnb_df.schema,
# MAGIC     description="Original Airbnb data"
# MAGIC )
# MAGIC 
# MAGIC fs.write_table(
# MAGIC     name=table_name,
# MAGIC     df=airbnb_df,
# MAGIC     mode="overwrite"
# MAGIC )
# MAGIC ```

# COMMAND ----------

# MAGIC %md Now let's explore the UI and see how it tracks the tables that we created. Navigate to the UI by first ensuring that you are in the Machine Learning workspace, and then clicking on the Feature Store icon on the bottom-left of the navigation bar.
# MAGIC 
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/mlflow/FS_Nav.png" alt="step12" width="150"/>

# COMMAND ----------

# MAGIC %md In this screenshot, we can see the feature table that we created.
# MAGIC <br>
# MAGIC <br>
# MAGIC Note the section of **`Producers`**. This section indicates which notebook produces the feature table.
# MAGIC <br>
# MAGIC <br>
# MAGIC <img src="https://s3.us-west-2.amazonaws.com/files.training.databricks.com/images/mlflow/fs_details+(1).png" alt="step12" width="1000"/>

# COMMAND ----------

# MAGIC %md
# MAGIC We can also look at the metadata of the feature store via the FeatureStore client by using **`get_table()`**.

# COMMAND ----------

print(f"Feature table description : {fs.get_table(table_name).description}")
print(f"Feature table data source : {fs.get_table(table_name).path_data_sources}")

# COMMAND ----------

# MAGIC %md Let's reduce the number of review columns by creating an average review score for each listing.

# COMMAND ----------

from pyspark.sql.functions import lit, expr

review_columns = ["review_scores_accuracy", "review_scores_cleanliness", "review_scores_checkin", 
                  "review_scores_communication", "review_scores_location", "review_scores_value"]

airbnb_df_short_reviews = (airbnb_df
                           .withColumn("average_review_score", expr("+".join(review_columns)) / lit(len(review_columns)))
                           .drop(*review_columns)
                          )

display(airbnb_df_short_reviews)

# COMMAND ----------

# MAGIC %md #### Overwrite Features
# MAGIC 
# MAGIC We set the mode to **`overwrite`** to remove the deleted feature columns from the latest table.

# COMMAND ----------

fs.write_table(
    name=table_name,
    df=airbnb_df_short_reviews,
    mode="overwrite"
)

# COMMAND ----------

# MAGIC %md By navigating back to the UI, we can again see that the modified date has changed and a new column has been added to the feature list. 
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/icon_note_32.png"> The deleted columns are still present in the schema of the table and their values have been replaced by **`null`**.
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/mlflow/FS_New_Feature.png" alt="step12" width="800"/>

# COMMAND ----------

# MAGIC %md Let's read in the data from the Feature Store. We can optionally specify a version or timestamp to use Delta Time Travel to read from a snapshot of the feature table. 

# COMMAND ----------

# Display most recent table
feature_df = fs.read_table(name=table_name)
display(feature_df)

# COMMAND ----------

# MAGIC %md If you need to use the features for real-time serving, you can publish your features to an <a href="https://docs.databricks.com/applications/machine-learning/feature-store.html#publish-features-to-an-online-feature-store" target="_blank">online store</a>.
# MAGIC 
# MAGIC We can also control permissions to the feature table. To delete the table, use the **`delete`** button on the UI. **You need to delete the delta table from database as well.**
# MAGIC 
# MAGIC <img src="https://s3.us-west-2.amazonaws.com/files.training.databricks.com/images/mlflow/fs_permissions+(1).png" alt="step12" width="700"/>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Review
# MAGIC **Question:** Why do we care about Data Management?
# MAGIC **Answer:** Data Management is an oftentimes overlooked aspect of end-to-end reproducibility.
# MAGIC 
# MAGIC **Question:** How do we version data with Delta Tables?
# MAGIC **Answer:** Delta Tables are automatically versioned everytime a new data is written. Accessing a previous version of the table is as simple as using **`display(spark.sql(f"DESCRIBE HISTORY delta.{delta_path}"))`** to find the version to revert to and loading it in.  You can also revert to previous version using timestamps.
# MAGIC 
# MAGIC **Question:** What challenges does the Feature Store help solve?
# MAGIC **Answer:** A key issue many ML pipelines struggle with is feature reproducibility and data sharing. The Feature Store lets different users across the same organization utilize the same feature computation code.
# MAGIC 
# MAGIC **Question:** What does hashing a dataset help me do?
# MAGIC **Answer:** It can help confirm whether a dataset is or is not the same as another.  This is helpful in data reproducibility.  It cannot, however, tell you the full diff between two datasets and is not a scalable solution.

# COMMAND ----------

# MAGIC %md
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) Next Steps
# MAGIC 
# MAGIC Start the next lesson, [Experiment Tracking]($./02-Experiment-Tracking)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Additional Topics & Resources
# MAGIC 
# MAGIC **Q:** Where can I learn more about Delta Tables?
# MAGIC **A:** Check out this <a href="https://databricks.com/session_na21/intro-to-delta-lake" target="_blank"> talk </a> by Himanshu Raj at the Data+AI Summit 2021.
# MAGIC 
# MAGIC **Q:** Where can I learn more about the Feature Store?
# MAGIC **A:** The <a href="https://docs.databricks.com/applications/machine-learning/feature-store.html" target="_blank"> documentation </a> provides an in-depth look at what the Feature Store can do for your pipeline.
# MAGIC 
# MAGIC **Q:** Where can I learn more about reproducibility and its importance?
# MAGIC **A:** This <a href="https://databricks.com/blog/2021/04/26/reproduce-anything-machine-learning-meets-data-lakehouse.html" target="_blank">blog post</a> by Mary Grace Moesta and Srijith Rajamohan provides a starting point for creating reproducible data and models

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
