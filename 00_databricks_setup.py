# Databricks notebook source
# ------------
# Packages
# ------------

!pip install 'google-cloud-bigquery[pandas]'
from google.cloud import bigquery
from google.oauth2 import service_account
import pandas as pd
import numpy as np

# Authenticate
credentials = service_account.Credentials.from_service_account_file('grav-waves-358320-34ebfeae2689.json', scopes=["https://www.googleapis.com/auth/cloud-platform"],)
client = bigquery.Client(credentials=credentials, project=credentials.project_id,)

# COMMAND ----------

# ------------------------
# Pull in data from GBQ
# ------------------------

sql = """
SELECT *
FROM mergers.background
WHERE event_id BETWEEN 2001 AND 4001;
"""

gw_events = client.query(sql).to_dataframe()

# COMMAND ----------

spark.createDataFrame(gw_events).write.saveAsTable('user_nsulliv3.gw_bg_4000')

# COMMAND ----------

# --------------------------------
# Stack df's to create one table
# --------------------------------

spark.sql('''
CREATE OR REPLACE TABLE user_nsulliv3.gw_bns AS

SELECT *
FROM gw_bns1000 

UNION ALL 
SELECT * 
FROM gw_bns2000 

UNION ALL 
SELECT *
FROM gw_bns3000

''')


