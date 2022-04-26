# Databricks notebook source
def get_cloud():
  with open("/databricks/common/conf/deploy.conf") as f:
    for line in f:
      if "databricks.instance.metadata.cloudProvider" in line and "\"GCP\"" in line: return "GCP"
      elif "databricks.instance.metadata.cloudProvider" in line and "\"AWS\"" in line: return "AWS"
      elif "databricks.instance.metadata.cloudProvider" in line and "\"Azure\"" in line: return "MSA"

#############################################
# TAG API FUNCTIONS
#############################################

# Get all tags
def getTags() -> dict: 
  return sc._jvm.scala.collection.JavaConversions.mapAsJavaMap(
    dbutils.entry_point.getDbutils().notebook().getContext().tags()
  )

# Get a single tag's value
def getTag(tagName: str, defaultValue: str = None) -> str:
  values = getTags()[tagName]
  try:
    if len(values) > 0:
      return values
  except:
    return defaultValue

#############################################
# Get Databricks runtime major and minor versions
#############################################

def getDbrMajorAndMinorVersions() -> (int, int):
  import os
  dbrVersion = os.environ["DATABRICKS_RUNTIME_VERSION"]
  dbrVersion = dbrVersion.split(".")
  return (int(dbrVersion[0]), int(dbrVersion[1]))

# Get Python version
def getPythonVersion() -> str:
  import sys
  pythonVersion = sys.version[0:sys.version.index(" ")]
  spark.conf.set("com.databricks.training.python-version", pythonVersion)
  return pythonVersion

#############################################
# USER, USERNAME, AND USERHOME FUNCTIONS
#############################################

# Get the user's username
def getUsername() -> str:
  import uuid
  try:
    return dbutils.widgets.get("databricksUsername")
  except:
    return getTag("user", str(uuid.uuid1()).replace("-", ""))

def getCleanUsername() -> str:
  import re
  return re.sub(r"[^a-zA-Z0-9]", "_", getUsername().split("@")[0]).lower()

# Get the user's userhome
def getUserhome() -> str:
  username = getUsername()
  return "dbfs:/user/{}".format(username)

def getModuleName() -> str: 
  # This will/should fail if module-name is not defined in the Classroom-Setup notebook
  return spark.conf.get("com.databricks.training.module-name")

def getLessonName() -> str:
  # If not specified, use the notebook's name.
  return dbutils.entry_point.getDbutils().notebook().getContext().notebookPath().getOrElse(None).split("/")[-1]

def getCourseDir() -> str:
  import re
  moduleName = re.sub(r"[^a-zA-Z0-9]", "_", getModuleName()).lower()
  courseDir = f"{getUserhome()}/{moduleName}"
  return courseDir.replace("__", "_").replace("__", "_").replace("__", "_").replace("__", "_")

def getWorkingDir() -> str:
  import re
  course_dir = getCourseDir()
  lessonName = re.sub(r"[^a-zA-Z0-9]", "_", getLessonName()).lower()
  working_dir = f"{getCourseDir()}/{lessonName}"
  return working_dir.replace("__", "_").replace("__", "_").replace("__", "_").replace("__", "_")


#############################################
# VERSION ASSERTION FUNCTIONS
#############################################

# When migrating DBR versions this should be one
# of the only two places that needs to be updated
latestDbrMajor = 9
latestDbrMinor = 1

# Assert an appropriate Databricks Runtime version
# def assertDbrVersion(expected:str, latestMajor:int=latestDbrMajor, latestMinor:int=latestDbrMinor, display:bool = True):
  
#   expMajor = latestMajor
#   expMinor = latestMinor
  
#   if expected and expected != UNSUPPORTED:
#     expMajor = int(expected.split(".")[0])
#     expMinor = int(expected.split(".")[1])

#   (major, minor) = getDbrMajorAndMinorVersions()

#   if (major < expMajor) or (major == expMajor and minor < expMinor):
#     msg = f"This notebook must be run on DBR {expMajor}.{expMinor} or newer. Your cluster is using {major}.{minor}. You must update your cluster configuration before proceeding."

#     raise AssertionError(msg)
    
#   if major != expMajor or minor != expMinor:
#     html = f"""
#       <div style="color:red; font-weight:bold">WARNING: This notebook was tested on DBR {expMajor}.{expMinor}, but we found DBR {major}.{minor}.</div>
#       <div style="font-weight:bold">Using an untested DBR may yield unexpected results and/or various errors</div>
#       <div style="font-weight:bold">Please update your cluster configuration and/or <a href="https://academy.databricks.com/" target="_blank">download a newer version of this course</a> before proceeding.</div>
#     """

#   else:
#     html = f"Running on <b>DBR {major}.{minor}</b>"
  
#   if display:
#     displayHTML(html)
#   else:
#     print(html)
  
#   return f"{major}.{minor}"

displayHTML("Defining courseware-specific utility methods...")


