# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md # MLflow Webhooks & Testing
# MAGIC 
# MAGIC Webhooks trigger the execution of code (oftentimes tests) upon some event. This lesson explores how to employ webhooks to trigger automated tests against models in the model registry. 
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lesson you will:<br>
# MAGIC  - Explore the role of webhooks in ML pipelines
# MAGIC  - Create a job to test models in the model registry
# MAGIC  - Automate that job using MLflow webhooks
# MAGIC  - Create a HTTP webhook to send notifications to Slack
# MAGIC  

# COMMAND ----------

# MAGIC %md ## Automated Testing
# MAGIC 
# MAGIC The backbone of the continuous integration, continuous deployment (CI/CD) process is the automated building, testing, and deployment of code. A **webhook or trigger** causes the execution of code based upon some event.  This is commonly when new code is pushed to a code repository.  In the case of machine learning jobs, this could be the arrival of a new model in the model registry.
# MAGIC 
# MAGIC The two types of <a href="https://docs.databricks.com/applications/mlflow/model-registry-webhooks.html" target="_blank">**MLflow Model Registry Webhooks**</a>:
# MAGIC  - Webhooks with Job triggers: Trigger a job in a Databricks workspace
# MAGIC  - Webhooks with HTTP endpoints: Send triggers to any HTTP endpoint
# MAGIC  
# MAGIC This lesson uses:
# MAGIC 1. a **Job webhook** to trigger the execution of a Databricks job 
# MAGIC 2. a **HTTP webhook** to send notifications to Slack 
# MAGIC 
# MAGIC Upon the arrival of a new model version with a given name in the model registry, the function of the Databricks job is to:<br><br>
# MAGIC - Import the new model version
# MAGIC - Test the schema of its inputs and outputs
# MAGIC - Pass example code through the model
# MAGIC 
# MAGIC This covers many of the desired tests for ML models.  However, throughput testing could also be performed using this paradigm. Also, the model could also be promoted to the production stage in an automated fashion.

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Create a Model and Job
# MAGIC 
# MAGIC The following steps will create a Databricks job using another notebook in this directory: **`03b-Webhooks-Job-Demo`**
# MAGIC 
# MAGIC **Note:** 
# MAGIC * Ensure that you are an admin on this workspace and that you're not using Community Edition (which has jobs disabled). 
# MAGIC * If you are not an admin, ask the instructor to share their token with you. 
# MAGIC * Alternatively, you can set **`token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)`**.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create a user access token
# MAGIC 
# MAGIC Create a user access token using the following steps:<br><br>
# MAGIC 
# MAGIC 1. Click the Settings icon
# MAGIC 1. Click User Settings
# MAGIC 1. Go to the Access Tokens tab
# MAGIC 1. Click the Generate New Token button
# MAGIC 1. Optionally enter a description (comment) and expiration period
# MAGIC 1. Click the Generate button
# MAGIC 1. Copy the generated token **and paste it in the following cell**
# MAGIC 
# MAGIC **Note:**
# MAGIC * Ensure that you are an admin on this workspace and that you're not using Community Edition (which has jobs disabled). 
# MAGIC * If you are not an admin, ask the instructor to share their token with you. 
# MAGIC * Alternatively, you can set **`token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)`**.
# MAGIC 
# MAGIC You can find details <a href="https://docs.databricks.com/dev-tools/api/latest/authentication.html" target="_blank">about access tokens here</a>

# COMMAND ----------

# Paste your token below

token = "<insert your token here>" 
# token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)

# COMMAND ----------

# With the token, we can create our authorization header for our subsequent REST calls
headers = {"Authorization": f"Bearer {token}"}

# Next we need an enpoint at which to execute our request which we can get from the Notebook's tags collection
java_tags = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags()
# This ojbect comes from the Java CM - Convert the Java Map opject to a Python dictionary
tags = sc._jvm.scala.collection.JavaConversions.mapAsJavaMap(java_tags)
# Lastly, extract the databricks instance (domain name) from the dictionary
instance = tags["browserHostName"]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Train and Register a Model
# MAGIC 
# MAGIC Build and log your model.

# COMMAND ----------

from mlflow.models.signature import infer_signature
from sklearn.metrics import mean_squared_error
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

with mlflow.start_run(run_name="Webhook RF Experiment") as run:
    # Data prep
    df = pd.read_parquet("/dbfs/mnt/training/airbnb/sf-listings/airbnb-cleaned-mlflow.parquet")
    X_train, X_test, y_train, y_test = train_test_split(df.drop(["price"], axis=1), df["price"], random_state=42)
    signature = infer_signature(X_train, pd.DataFrame(y_train))
    example = X_train.head(3)

    # Train and log model
    rf = RandomForestRegressor(random_state=42)
    rf.fit(X_train, y_train)
    mlflow.sklearn.log_model(rf, "random-forest-model", signature=signature, input_example=example)
    mse = mean_squared_error(y_test, rf.predict(X_test))
    mlflow.log_metric("mse", mse)
    run_id = run.info.run_id
    experiment_id = run.info.experiment_id

# COMMAND ----------

# MAGIC %md Register the model

# COMMAND ----------

import uuid
name = f"webhook_demo_{uuid.uuid4().hex[:6]}"
model_uri = f"runs:/{run_id}/random-forest-model"

model_details = mlflow.register_model(model_uri=model_uri, name=name)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Creating the Job
# MAGIC 
# MAGIC The following steps will create a Databricks job using another notebook in this directory: **`03b-Webhooks-Job-Demo`**

# COMMAND ----------

# MAGIC %md Create a job that executes the notebook **`03b-Webhooks-Job-Demo`** in the same folder as this notebook.<br><br>
# MAGIC 
# MAGIC - Hover over the sidebar in the Databricks UI on the left.
# MAGIC 
# MAGIC <img src="http://files.training.databricks.com/images/ml-deployment/Job_1.png" alt="step12" width="100"/>
# MAGIC <br></br>
# MAGIC 
# MAGIC - Click on Create Job
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/ml-deployment/Job_2.png" alt="step12" width="750"/>
# MAGIC 
# MAGIC <br></br>
# MAGIC - Name your Job
# MAGIC - Select the notebook **`03b-Webhooks-Job-Demo`** 
# MAGIC - Select the current cluster
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/ml-deployment/Job_3_6.png" alt="step12" width="750"/>
# MAGIC 
# MAGIC <br></br>
# MAGIC - Copy the Job ID
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/ml-deployment/Job_7.png" alt="step12" width="750"/>

# COMMAND ----------

# MAGIC %md Alternatively, the code below will programmatically create the job.

# COMMAND ----------

import requests

def find_job_id(instance, headers, job_name, offset_limit=1000):
    params = {"offset": 0}
    uri = f"https://{instance}/api/2.1/jobs/list"
    done = False
    job_id = None
    while not done:
        done = True
        res = requests.get(uri, params=params, headers=headers)
        assert res.status_code == 200, f"Job list not returned; {res.content}"
        
        jobs = res.json().get("jobs", [])
        if len(jobs) > 0:
            for job in jobs:
                if job.get("settings", {}).get("name", None) == job_name:
                    job_id = job.get("job_id", None)
                    break

            # if job_id not found; update the offset and try again
            if job_id is None:
                params["offset"] += len(jobs)
                if params["offset"] < offset_limit:
                    done = False
    
    return job_id

def get_job_parameters(job_name, cluster_id, notebook_path):
    params = {
            "name": job_name,
            "tasks": [{"task_key": "webhook_task", 
                       "existing_cluster_id": cluster_id,
                       "notebook_task": {
                           "notebook_path": notebook_path
                       }
                      }]
        }
    return params

def get_create_parameters(job_name, cluster_id, notebook_path):
    api = "api/2.1/jobs/create"
    return api, get_job_parameters(job_name, cluster_id, notebook_path)

def get_reset_parameters(job_name, cluster_id, notebook_path, job_id):
    api = "api/2.1/jobs/reset"
    params = {"job_id": job_id, "new_settings": get_job_parameters(job_name, cluster_id, notebook_path)}
    return api, params

def get_webhook_job(instance, headers, job_name, cluster_id, notebook_path):
    job_id = find_job_id(instance, headers, job_name)
    if job_id is None:
        api, params = get_create_parameters(job_name, cluster_id, notebook_path)
    else:
        api, params = get_reset_parameters(job_name, cluster_id, notebook_path, job_id)
    
    uri = f"https://{instance}/{api}"
    res = requests.post(uri, headers=headers, json=params)
    assert res.status_code == 200, f"Expected an HTTP 200 response, received {res.status_code}; {res.content}"
    job_id = res.json().get("job_id", job_id)
    return job_id

notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get().replace("03a-Webhooks-and-Testing", "03b-Webhooks-Job-Demo")

# if the Job was created via UI, set it here.
job_id = get_webhook_job(instance, 
                         headers, 
                         f"{clean_username}_webhook_job",
                         spark.conf.get("spark.databricks.clusterUsageTags.clusterId"),
                         notebook_path
                        )

print(job_id)

# COMMAND ----------

# MAGIC %md ### Examine the Job
# MAGIC 
# MAGIC Take a look at [the notebook you just scheduled]($./03b-Webhooks-Job-Demo) to see what it accomplishes.

# COMMAND ----------

# MAGIC %md ## Create a Job Webhook
# MAGIC 
# MAGIC There are a few different events that can trigger a Webhook. In this notebook, we will be experimenting with triggering a job when our model transitions between stages.

# COMMAND ----------

from mlflow.utils.rest_utils import http_request
from mlflow.utils.databricks_utils import get_databricks_host_creds

url = f"https://{instance}"
endpoint = "/api/2.0/mlflow/registry-webhooks/create"
host_creds = get_databricks_host_creds("databricks")

job_json = {"model_name": name,
            "events": ["MODEL_VERSION_TRANSITIONED_STAGE"],
            "description": "Job webhook trigger",
            "status": "Active",
            "job_spec": {"job_id": job_id,
                         "workspace_url": url,
                         "access_token": token}
           }

response = http_request(
    host_creds=host_creds, 
    endpoint=endpoint,
    method="POST",
    json=job_json
)


# COMMAND ----------

# MAGIC %md Now that we have registered the webhook, we can **test it by transitioning our model from stage `None` to `Staging` in the Experiment UI.** We should see in the Jobs tab that our Job has run.

# COMMAND ----------

# MAGIC %md To get a list of active Webhooks, use a GET request with the LIST endpoint. Note that this command will return an error if no Webhooks have been created for the Model.

# COMMAND ----------

endpoint = f"/api/2.0/mlflow/registry-webhooks/list/?model_name={name.replace(' ', '%20')}"

response = http_request(
    host_creds=host_creds, 
    endpoint=endpoint,
    method="GET"
)

print(response.json())

# COMMAND ----------

# MAGIC %md Finally, delete the webhook by copying the webhook ID to the curl or python request. You can confirm that the Webhook was deleted by using the list request.

# COMMAND ----------

delete_hook = "<insert your webhook id here>"
new_json = {"id": delete_hook}
endpoint = f"/api/2.0/mlflow/registry-webhooks/delete"

response = http_request(
    host_creds=host_creds, 
    endpoint=endpoint,
    method="DELETE",
    json=new_json
)

print(response.json())

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Create a HTTP webhook
# MAGIC 
# MAGIC This section requires that you have access to a Slack workspace and permissions to create a webhook. This design pattern also works with Teams or other endpoints that accept HTTP requests.
# MAGIC  
# MAGIC Set a Slack incoming webhook following <a href="https://api.slack.com/messaging/webhooks" target="_blank">this page</a>. Paste your webhook in the code below and uncomment the code. It should look like **`https://hooks.slack.com...`** Upon the arrival of a new model version with a given name in the model registry, it will send notifications to the slack channel.
# MAGIC 
# MAGIC Note that you can find more details on <a href="https://github.com/mlflow/mlflow/blob/master/mlflow/utils/rest_utils.py" target="_blank">the **`mlflow`** REST utility functions here.</a>

# COMMAND ----------

# from mlflow.utils.rest_utils import http_request
# from mlflow.utils.databricks_utils import get_databricks_host_creds
# import json
# import urllib

# slack_incoming_webhook = "<insert your token here>" 

# url = f"https://{instance}"
# endpoint = "/api/2.0/mlflow/registry-webhooks/create"
# host_creds = get_databricks_host_creds("databricks")

# ## specify http url of the slack notification
# http_json = {"model_name": name,
#   "events": ["MODEL_VERSION_TRANSITIONED_STAGE"],
#   "description": "Job webhook trigger",
#   "status": "Active",
#   "http_url_spec": {
#     "url": slack_incoming_webhook,
#     "enable_ssl_verification": "false"}}

# response = http_request(
#   host_creds=host_creds, 
#   endpoint=endpoint,
#   method="POST",
#   json=http_json
# )

# print(response.json())

# COMMAND ----------

# MAGIC %md Now that we have registered the webhook, we can **test it by transitioning our model from stage `None` to `Staging` in the Experiment UI.** We should see an incoming message in the associated slack channel. 
# MAGIC 
# MAGIC <img src="http://files.training.databricks.com/images/ml-deployment/webhook_slack.png" alt="webhook_notification" width="400"/>
# MAGIC <br></br>

# COMMAND ----------

# MAGIC %md ## Resources
# MAGIC 
# MAGIC - <a href="https://databricks.com/blog/2020/11/19/mlflow-model-registry-on-databricks-simplifies-mlops-with-ci-cd-features.html" target="_blank">See this blog for more details on CI/CD and webhooks</a>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
