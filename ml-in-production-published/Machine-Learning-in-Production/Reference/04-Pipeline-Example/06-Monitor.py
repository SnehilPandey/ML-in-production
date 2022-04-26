# Databricks notebook source
# MAGIC %md ### Drift Monitor
# MAGIC 
# MAGIC This notebook is called by Orchestrate to determine if drift has occured over time. 
# MAGIC 
# MAGIC It takes in two time periods of data and determines if significant change has occurred in addition to a target directory to write the drift.
# MAGIC 
# MAGIC Let's first load the data passed from Orchestrate. 

# COMMAND ----------

dbutils.widgets.text("file_path_1", "Default")
dbutils.widgets.text("file_path_2", "Default")
dbutils.widgets.text("drift_path", "Default")

file_path_1 = dbutils.widgets.get("file_path_1")
file_path_2 = dbutils.widgets.get("file_path_2")
drift_path = dbutils.widgets.get("drift_path")

# COMMAND ----------

df1_featurized = spark.read.format("delta").load(file_path_1)
df2 = spark.read.format("delta").load(file_path_2)

# COMMAND ----------

# MAGIC %md Right now we have our featurized dataset we trained on for the first time period. 
# MAGIC 
# MAGIC The second time period still needs to be featurized, however. 
# MAGIC 
# MAGIC Let's identify the caterogical and numeric columns, since we will want those for drift monitoring, and featurize the second time window.

# COMMAND ----------

categorical_cols = [field for (field, dataType) in df2.dtypes if dataType == "string"]
numeric_cols = [field for (field, dataType) in df2.dtypes if ((dataType == "double"))]
cols = numeric_cols + categorical_cols

# COMMAND ----------

# MAGIC %md Create a path for the second time window featurized data. 

# COMMAND ----------

dbutils.fs.rm(drift_path, True)
dbutils.fs.mkdirs(drift_path)

# COMMAND ----------

# MAGIC %md Run the Featurize notebook on the time window. 

# COMMAND ----------

params = {
    "file_path": file_path_2, 
    "save_path": drift_path
}
dbutils.notebook.run("./02-Featurize", 0, params)

# COMMAND ----------

# MAGIC %md Load the featurized second time window. 

# COMMAND ----------

df2_featurized = spark.read.format("delta").load(drift_path)

# COMMAND ----------

# MAGIC %md Load the Monitor class from the lesson. 

# COMMAND ----------

# Dependencies for Class Monitor
import pandas as pd
import seaborn as sns
from scipy import stats
import numpy as np 

class Monitor():
  
    def __init__(self, pdf1, pdf2, categorical_cols, numeric_cols, alpha=.05):
        """
        Pass in two pandas dataframes with the same columns for two time windows
        List the categorical and numeric columns, and optionally provide an alpha level
        """
        assert (pdf1.columns == pdf2.columns).all(), "Columns do not match"
        self.pdf1 = pdf1
        self.pdf2 = pdf2
        self.alpha = alpha
        self.categorical_cols = categorical_cols
        self.numeric_cols = numeric_cols
    
    
    def run(self):
        """
        Call to run drift monitoring
        """
        self.handle_numeric()
        self.handle_categorical()
  
    def handle_numeric(self):
        """
        Handle the numeric features with the Two-Sample Kolmogorov-Smirnov (KS) Test with Bonferroni Correction 
        """
        corrected_alpha = self.alpha / len(self.numeric_cols)

        for num in self.numeric_cols:
            ks_stat, ks_pval = stats.ks_2samp(self.pdf1[num], self.pdf2[num], mode="asymp")
            if ks_pval <= corrected_alpha:
                self.on_drift(num)
 
    def handle_categorical(self):
        """
        Handle the Categorical features with Two-Way Chi-Squared Test with Bonferroni Correction
        """
        corrected_alpha = self.alpha / len(self.categorical_cols)

        for feature in self.categorical_cols:
            pdf_count1 = pd.DataFrame(self.pdf1[feature].value_counts()).sort_index().rename(columns={feature:"pdf1"})
            pdf_count2 = pd.DataFrame(self.pdf2[feature].value_counts()).sort_index().rename(columns={feature:"pdf2"})
            pdf_counts = pdf_count1.join(pdf_count2, how="outer").fillna(0)
            obs = np.array([pdf_counts["pdf1"], pdf_counts["pdf2"]])
            _, p, _, _ = stats.chi2_contingency(obs)
            if p < corrected_alpha:
                self.on_drift(feature)
  
    def generate_null_counts(self, palette="#2ecc71"):
        """
        Generate the visualization of percent null counts of all features
        Optionally provide a color palette for the visual
        """
        cm = sns.light_palette(palette, as_cmap=True)
        return pd.concat([100 * self.pdf1.isnull().sum() / len(self.pdf1), 
                          100 * self.pdf2.isnull().sum() / len(self.pdf2)], axis=1, 
                          keys=["pdf1", "pdf2"]).style.background_gradient(cmap=cm, text_color_threshold=0.5, axis=1)

    def generate_percent_change(self, palette="#2ecc71"):
        """
        Generate visualization of percent change in summary statistics of numeric features
        Optionally provide a color palette for the visual
        """
        cm = sns.light_palette(palette, as_cmap=True)
        summary1_pdf = self.pdf1.describe()[self.numeric_cols]
        summary2_pdf = self.pdf2.describe()[self.numeric_cols]
        percent_change = 100 * abs((summary1_pdf - summary2_pdf) / (summary1_pdf + 1e-100))
        return percent_change.style.background_gradient(cmap=cm, text_color_threshold=0.5, axis=1)
    
    def on_drift(self, feature):
        """
        Complete this method with your response to drift.  Options include:
          - raise an alert
          - automatically retrain model
        """
        print(f"Drift found in {feature}!")

# COMMAND ----------

# MAGIC %md Finally, let's load in our data and run drift monitoring. 

# COMMAND ----------

drift_monitor = Monitor(df1_featurized.select(cols).toPandas(), df2_featurized.select(cols).toPandas(), categorical_cols, numeric_cols)
drift_monitor.run()

# COMMAND ----------

drift_monitor.generate_percent_change()

