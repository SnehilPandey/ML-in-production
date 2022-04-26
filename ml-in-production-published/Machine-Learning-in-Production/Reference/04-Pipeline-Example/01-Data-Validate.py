# Databricks notebook source
# MAGIC %md ### Data Validate 
# MAGIC 
# MAGIC This notebook is called by the Orchestrate notebook to validate data. 
# MAGIC 
# MAGIC Let's first load the paths passed to the notebook by Orchestrate. 
# MAGIC 
# MAGIC When triggered by the Orchestrate notebook, these widgets will be overwritten with the data paths we passed to the notebook.

# COMMAND ----------

# Overwritten by orchestrate
dbutils.widgets.text("file_path", "Default")

# COMMAND ----------

file_path = dbutils.widgets.get("file_path")
df = spark.read.format("delta").load(file_path)

# COMMAND ----------

# MAGIC %md In this simple example of data validation, we will just assert that the schema is as expected. 
# MAGIC 
# MAGIC We define our expected schema and assert that, regardless of the order of the fields, that they are equal. 

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, StringType, DoubleType

expected_schema = (StructType([StructField("accommodates", DoubleType(), True),
                               StructField("beds", DoubleType(), True),
                               StructField("bedrooms", DoubleType(), True),
                               StructField("minimum_nights", DoubleType(), True),
                               StructField("number_of_reviews", DoubleType(), True),
                               StructField("review_scores_rating", DoubleType(), True),
                               StructField("price", DoubleType(), True),
                               StructField("neighbourhood_cleansed", StringType(), True),
                               StructField("property_type", StringType(), True),
                               StructField("room_type", StringType(), True)
                               ])
                  )


expected = expected_schema.fields
expected.sort(key=lambda x: x.name)

observed = df.schema.fields
observed.sort(key=lambda x: x.name)

# COMMAND ----------

# MAGIC %md Now we can return the boolean comparison to the Orchestrate notebook so we can assert it. 
# MAGIC 
# MAGIC **Note:** Data passed between notebooks with **`dbutils`** can only be string, so this will return the string form of the boolean. 

# COMMAND ----------

dbutils.notebook.exit(expected == observed)

