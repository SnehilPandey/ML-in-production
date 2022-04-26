# Databricks notebook source
# MAGIC %md ### Featurize 
# MAGIC 
# MAGIC In this notebook we will featurize our validated data by String Encoding the categorical columns and generating feature vectors. 
# MAGIC 
# MAGIC First, let's load in our data passed from Orchestrate. 

# COMMAND ----------

dbutils.widgets.text("file_path", "Default")
dbutils.widgets.text("save_path", "Default")

file_path = dbutils.widgets.get("file_path")
save_path = dbutils.widgets.get("save_path")

# COMMAND ----------

# MAGIC %md Now, we can load in the validated data.

# COMMAND ----------

df = spark.read.format("delta").load(file_path)

# COMMAND ----------

# MAGIC %md Next, we can String Encode the columns and assemble our feature vectors. 

# COMMAND ----------

from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml import Pipeline

categorical_cols = [field for (field, dataType) in df.dtypes if dataType == "string"]
index_output_cols = [x + "_index" for x in categorical_cols]

string_indexer = StringIndexer(inputCols=categorical_cols, outputCols=index_output_cols, handleInvalid="skip")

numeric_cols = [field for (field, dataType) in df.dtypes if ((dataType == "double") & (field != "price"))]
assembler_inputs = index_output_cols + numeric_cols
vec_assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features")

stages = [string_indexer, vec_assembler]
pipeline = Pipeline(stages=stages)

pipeline_model = pipeline.fit(df)
featurized_df = pipeline_model.transform(df)

# COMMAND ----------

# MAGIC %md Finally, we can save our featurized data to the paths we reserved in Orchestrate.

# COMMAND ----------

featurized_df.write.format("delta").save(save_path)

