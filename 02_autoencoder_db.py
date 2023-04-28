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

!pip install sequitur
import torch
from sequitur.models import LINEAR_AE
from sequitur import quick_train

# sequitur automatically uses GPU if it's available, so we don't need to worry about this
# torch.cuda.is_available()

!pip install torchinfo
from torchinfo import summary

# COMMAND ----------

# ------------
# Data
# ------------

# Simulated data that's been bandpassed and whitened
sql = """
SELECT *
FROM mergers.bns_generated
WHERE event_id BETWEEN 400 AND 410
"""

event_403 = client.query(sql).to_dataframe()

# ---------------------------------
# Data pre-processing for torch
# ---------------------------------

train_seqs = []
test_seqs = []

for i in np.arange(400, 411):

  event_pd = event_403.query(f'event_id == {i}')
  event_vals = event_pd.strain.values
  event_tensor = torch.from_numpy(event_vals).float()

  if i in np.arange(400, 408):
    train_seqs.append(event_tensor)

  else:
    test_seqs.append(event_tensor)

# --------------------
# Train autoencoder
# --------------------

# Length of examples must be a multiple of encoding_dim (size of z)
encoder, decoder, encodings, losses = quick_train(LINEAR_AE, train_seqs, encoding_dim = 2048, denoise = True, verbose = False)


# --------------------------------------------------
# Generate latent space vectors & reconstructions
# --------------------------------------------------

z_ls = []
x_prime_ls = []

for i in test_seqs:
  z = encoder(i) # Pass through new observation
  z_np = z.detach().cpu().numpy()
  z_ls.append(z_np)

  x_prime = decoder(z)
  x_prime_np = x_prime.detach().cpu().numpy()
  x_prime_ls.append(x_prime_np)

# ------------------------------
# Create dataframe of z vectors
# ------------------------------

z_pd_ls = []

for i, ele in enumerate(z_ls):
  pd_i = pd.DataFrame({'z_vals': ele,
                      'event_id': 407+i})
  z_pd_ls.append(pd_i)

z_pd = pd.concat(z_pd_ls)

# Create a spark dataframe to visualize
spark.createDataFrame(z_pd).createOrReplaceTempView('z_test')


# --------------------------------
# Create dataframe of x' vectors
# --------------------------------

x_prime_pd_ls = []

for i, ele in enumerate(x_prime_ls):
  pd_i = pd.DataFrame({'x_prime': ele,
                      'event_id': 407+i})
  x_prime_pd_ls.append(pd_i)

x_prime_pd = pd.concat(x_prime_pd_ls)
x_prime_pd.head()


# COMMAND ----------



# COMMAND ----------

# MAGIC %r
# MAGIC options(repr.plot.width = 2200, repr.plot.height = 1000)

# COMMAND ----------

# MAGIC %r
# MAGIC # library(tidyverse)
# MAGIC # library(SparkR)
# MAGIC
# MAGIC z_df <- collect(sql('SELECT * FROM z_test;'))
# MAGIC
# MAGIC z_df %>%
# MAGIC dplyr::group_by(event_id) %>%
# MAGIC dplyr::mutate(row_n = dplyr::row_number()) %>%
# MAGIC ungroup() %>%
# MAGIC ggplot(aes(row_n, z_vals, color = factor(event_id))) +
# MAGIC geom_line(alpha = 0.7) +
# MAGIC ylim(-2, 2) +
# MAGIC facet_wrap(~factor(event_id))
