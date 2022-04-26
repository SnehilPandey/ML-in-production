# Databricks notebook source
spark.conf.set("com.databricks.training.module-name", "ml-in-production")

None # Suppress Output

# COMMAND ----------

# MAGIC %run "./Class-Utility-Methods"

# COMMAND ----------

# MAGIC %run "./Dataset-Mounts"

# COMMAND ----------

username = getUsername()
clean_username = getCleanUsername()
userhome = getUserhome()
course_dir = getCourseDir()
datasets_dir = f"{course_dir}/datasets"
working_dir = getWorkingDir().replace("_pil", "")

dbutils.fs.mkdirs(userhome)
dbutils.fs.mkdirs(course_dir)
dbutils.fs.mkdirs(working_dir)

None # Suppress output

# COMMAND ----------

# Used to initialize MLflow with the job ID when ran under test
def init_mlflow_as_job():
  import mlflow
  job_experiment_id = sc._jvm.scala.collection.JavaConversions.mapAsJavaMap(
      dbutils.entry_point.getDbutils().notebook().getContext().tags()
    )["jobId"]

  if job_experiment_id:
    mlflow.set_experiment(f"/Curriculum/Test Results/Experiments/{job_experiment_id}")
    
init_mlflow_as_job()

None # Suppress output

