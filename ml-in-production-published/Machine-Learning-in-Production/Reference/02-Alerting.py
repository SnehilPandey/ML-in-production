# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md # Alerting
# MAGIC 
# MAGIC Alerting allows you to announce the progress of different applications, which becomes increasingly important in automated production systems.  In this lesson, you explore basic alerting strategies using email and REST integration with tools like Slack and Microsoft Teams.
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lesson you will:<br>
# MAGIC  - Explore the alerting landscape
# MAGIC  - Create a basic REST alert integrated with Slack or Microsoft Teams

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup

# COMMAND ----------

# MAGIC %md 
# MAGIC ### The Alerting Landscape
# MAGIC 
# MAGIC There are a number of different alerting tools with various levels of sophistication.<br><br>
# MAGIC * PagerDuty 
# MAGIC  - Has become one of the most popular tools for monitoring production outages
# MAGIC  - Allows for the escalation of issues across a team with alerts including text messages and phone calls
# MAGIC * Slack or Microsoft Teams
# MAGIC * Twilio   
# MAGIC * Email alerts
# MAGIC 
# MAGIC Most alerting frameworks allow for custom alerting done through REST integration.

# COMMAND ----------

# MAGIC %md ### Setting Basic Alerts
# MAGIC 
# MAGIC Create a basic alert using a Slack or Microsoft Teams endpoint.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC To set up a Microsoft Teams endpoint, do the following:<br><br>
# MAGIC 
# MAGIC 1. After setting up teams, click on the **Teams** tab.
# MAGIC <div><img src="https://files.training.databricks.com/images/eLearning/ML-Part-4/teams1.png" style="height: 200px; margin: 20px"/></div>
# MAGIC 2. Click the dropdown next to the team you want to associate the endpoint to (create a new team if you don't have one already).  Then click **Connectors**. <br></br>
# MAGIC <div><img src="https://files.training.databricks.com/images/eLearning/ML-Part-4/teams2.png" style="height: 350px; margin: 20px"/></div>
# MAGIC 3. Choose **Configure** next to **Incoming Webhook**. <br></br>
# MAGIC <div><img src="https://files.training.databricks.com/images/eLearning/ML-Part-4/teams3.png" style="height: 250px; margin: 20px"/></div>
# MAGIC 4. Give the webhook a name and click **Create**. <br></br>
# MAGIC <div><img src="https://files.training.databricks.com/images/eLearning/ML-Part-4/teams4.png" style="height: 250px; margin: 20px"/></div>
# MAGIC 5. Copy the URL and paste it below.
# MAGIC <div><img src="https://files.training.databricks.com/images/eLearning/ML-Part-4/teams5.png" style="height: 250px; margin: 20px"/></div>

# COMMAND ----------

# MAGIC %md 
# MAGIC Define a Slack webhook.  This has **not** been done for you.
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/icon_note_24.png"/> Define your own Slack webhook <a href="https://api.slack.com/incoming-webhooks#getting-started" target="_blank">using these 4 steps. </a><br>
# MAGIC <img src="https://files.training.databricks.com/images/icon_note_24.png"/> This same approach applies to PagerDuty as well.
# MAGIC 
# MAGIC Follow the directions here to enable <a href="https://api.slack.com/incoming-webhooks#" target="_blank">incoming webhooks</a>.

# COMMAND ----------

webhook_ml_production_api_demo = "" # FILL_IN

# COMMAND ----------

# MAGIC %md Send a test message and check Slack.

# COMMAND ----------

def post_api_endpoint(content, webhook=""):
    """
    Post message to Teams to log progress
    """
    import requests
    from requests.exceptions import MissingSchema
    from string import Template

    t = Template("{'text': '${content}'}")

    try:
        response = requests.post(webhook, data=t.substitute(content=content), headers={"Content-Type": "application/json"})
        return response
    except MissingSchema:
        print("Please define an appropriate API endpoint use by defining the `webhook` argument")

post_api_endpoint("This is my post from Python", webhook_ml_production_api_demo)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Review
# MAGIC **Question:** What are the most common alerting tools?  
# MAGIC **Answer:** PagerDuty tends to be the tool most used in production environments.  SMTP servers emailing alerts are also popular, as is Twilio for text message alerts.  Slack webhooks and bots can easily be written as well.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Additional Topics & Resources
# MAGIC 
# MAGIC **Q:** Where can I find the alerting tools mentioned in this lesson?  
# MAGIC **A:** Check out <a href="https://www.twilio.com" target="_blank">Twilio</a> and <a href="https://www.pagerduty.com" target="_blank">PagerDuty</a>.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
