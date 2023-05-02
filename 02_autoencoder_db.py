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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

!pip install torchinfo
from torchinfo import summary

# COMMAND ----------

# MAGIC %r
# MAGIC
# MAGIC # Packages
# MAGIC install.packages("sysfonts")
# MAGIC library(sysfonts)
# MAGIC
# MAGIC # Fonts
# MAGIC # font_add_google("Didact Gothic", "dg")
# MAGIC font_add_google("IBM Plex Sans", "ibm")
# MAGIC
# MAGIC # showtext_auto()
# MAGIC font_fam = "ibm"
# MAGIC
# MAGIC # Aesthetics
# MAGIC purple_light <- "#d8d5f7"
# MAGIC purple <- "#9992DA" 
# MAGIC purple_dark <- "#706BA4"
# MAGIC grey_dark <- "#585959"
# MAGIC
# MAGIC theme_light_beige <- theme(plot.background = element_rect(fill = "#F0F1EA", color = "transparent"),
# MAGIC       panel.background = element_rect(fill = "#F0F1EA", color = "transparent"),
# MAGIC       plot.margin = margin(t = "1.5", r = "1.5", b = "1.5", l = "1.5", unit = "in"),
# MAGIC       plot.caption = element_text(size = 12, color = "#343A41", family = font_fam),
# MAGIC       panel.grid = element_blank(),
# MAGIC       plot.title = element_text(size = 40, color = "#343A41", family = font_fam, face = "bold"),
# MAGIC       axis.text = element_text(size = 15, color = "#343A41", family = font_fam),
# MAGIC       axis.title = element_text(size = 19, color = "#343A41", family = font_fam),
# MAGIC       axis.ticks = element_blank(),
# MAGIC       legend.background = element_blank(),
# MAGIC       legend.position = "bottom",
# MAGIC       legend.title = element_text(color = "#343A41", family = font_fam),
# MAGIC       legend.text = element_text(color = "#343A41", family = font_fam),
# MAGIC       strip.background = element_rect(fill = "#343A41"),
# MAGIC       strip.text = element_text(color = "white", family = font_fam, face = "bold", size = 13))
# MAGIC
# MAGIC
# MAGIC theme_tech <- theme(panel.background = element_rect(fill = purple_light, color = purple_light),
# MAGIC         panel.grid = element_line(color = purple_light),
# MAGIC         strip.background = element_rect(fill = purple_dark),
# MAGIC         strip.text = element_text(color = "white", size = 15, face = "bold"),
# MAGIC         axis.ticks = element_blank(),
# MAGIC         axis.title = element_text(face = "bold", size = 19, color = grey_dark),
# MAGIC         axis.text.y = element_text(size = 15, color = grey_dark),
# MAGIC         # axis.text.y = element_blank(),
# MAGIC         axis.text.x = element_text(size = 15, color = grey_dark, face = 'bold'),
# MAGIC         plot.title = element_text(size = 40, color = "#343A41", family = font_fam, face = "bold"),
# MAGIC         legend.position = "bottom",
# MAGIC         legend.text = element_text(size = 15, color = grey_dark))

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # 1. Proof-of-concept (part 2)
# MAGIC
# MAGIC Let's just ensure that the architecture I'm using can handle the full length of a BNS signal.

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

gw_events = client.query(sql).to_dataframe()

# ---------------------------------
# Data pre-processing for torch
# ---------------------------------

train_seqs = []
test_seqs = []

for i in np.arange(400, 411):

  event_pd = gw_events.query(f'event_id == {i}')
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
spark.createDataFrame(z_pd).createOrReplaceTempView('test_z')


# --------------------------------
# Create dataframe of x' vectors
# --------------------------------

x_prime_pd_ls = []

for i, ele in enumerate(x_prime_ls):
  pd_i = pd.DataFrame({'x_prime': ele,
                      'event_id': 407+i})
  x_prime_pd_ls.append(pd_i)

x_prime_pd = pd.concat(x_prime_pd_ls)
x_prime_pd.reset_index(inplace = True)
x_prime_df = x_prime_pd.rename(columns = {'index':'time_seq'})
x_prime_df['time_seq'] = x_prime_df['time_seq']+1

# COMMAND ----------

spark.createDataFrame(x_prime_df).createOrReplaceTempView('test_x_prime')
spark.createDataFrame(gw_events).createOrReplaceTempView('gw_main')

# Add back in original strain
spark.sql('''
CREATE OR REPLACE TEMPORARY VIEW gw_reconstr_eval AS
SELECT a.*, b.strain AS x_actual, pow((b.strain - a.x_prime), 2) AS MSE
FROM test_x_prime a 
LEFT JOIN gw_main b  
ON a.event_id = b.event_id 
AND a.time_seq = b.time_seq
''')

# COMMAND ----------

# MAGIC %r
# MAGIC options(repr.plot.width = 2200, repr.plot.height = 1000)

# COMMAND ----------

# MAGIC %r
# MAGIC # library(tidyverse)
# MAGIC # library(SparkR)
# MAGIC
# MAGIC # Bottleneck vector plot
# MAGIC
# MAGIC z_df <- collect(sql('SELECT * FROM test_z;'))
# MAGIC
# MAGIC z_df %>%
# MAGIC dplyr::group_by(event_id) %>%
# MAGIC dplyr::mutate(row_n = dplyr::row_number()) %>%
# MAGIC ungroup() %>%
# MAGIC ggplot(aes(row_n, z_vals, color = factor(event_id))) +
# MAGIC geom_line(alpha = 0.7) +
# MAGIC ylim(-2, 2) +
# MAGIC facet_wrap(~factor(event_id))

# COMMAND ----------

# MAGIC %r
# MAGIC
# MAGIC # Actual vs. reconstructed
# MAGIC
# MAGIC gw_reconstr <- collect(sql('SELECT * FROM gw_reconstr_eval;'))
# MAGIC
# MAGIC ggplot(gw_reconstr, aes(x_actual, x_prime, color = factor(event_id))) +
# MAGIC geom_point()

# COMMAND ----------

# MAGIC %r
# MAGIC
# MAGIC gw_reconstr %>%
# MAGIC gather(x_actual, x_prime, key = "x", value = "strain") %>%
# MAGIC ggplot(aes(time_seq, strain, color = x, group = x)) +
# MAGIC geom_line()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # 2. Switch to LSTM
# MAGIC
# MAGIC In #1 (above) in this notebook, I just use linear architecture and train it on a few samples from the BNS data.

# COMMAND ----------

from torch import nn, optim
import torch.nn.functional as F
import copy

# COMMAND ----------

sql = """
SELECT *
FROM mergers.bns_generated
WHERE event_id BETWEEN 400 AND 410
"""

gw_events = client.query(sql).to_dataframe()

# COMMAND ----------

gw_wide = gw_events.pivot(index = 'event_id', columns = 'time_seq', values = 'strain')
gw_np = gw_wide.to_numpy() # 11 x 32K (11 examples of length 32K timesteps)

train_np = gw_np[:8, :]
val_np = gw_np[8:, :]


train_set = [torch.tensor(x).unsqueeze(1).float() for x in train_np] # 11 x 32k x 1
val_set = [torch.tensor(x).unsqueeze(1).float() for x in val_np] 

# COMMAND ----------

class Encoder(nn.Module):

  def __init__(self, seq_len, n_features, embedding_dim=64):
    super(Encoder, self).__init__()

    self.seq_len, self.n_features = seq_len, n_features
    self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim

    self.rnn1 = nn.LSTM(
      input_size=n_features,
      hidden_size=self.hidden_dim,
      num_layers=1,
      batch_first=True
    )
    
    self.rnn2 = nn.LSTM(
      input_size=self.hidden_dim,
      hidden_size=embedding_dim,
      num_layers=1,
      batch_first=True
    )

  def forward(self, x):
    x = x.reshape((1, self.seq_len, self.n_features))

    x, (_, _) = self.rnn1(x)
    x, (hidden_n, _) = self.rnn2(x)

    return hidden_n.reshape((self.n_features, self.embedding_dim))

# COMMAND ----------

class Decoder(nn.Module):

  def __init__(self, seq_len, input_dim=64, n_features=1):
    super(Decoder, self).__init__()

    self.seq_len, self.input_dim = seq_len, input_dim
    self.hidden_dim, self.n_features = 2 * input_dim, n_features

    self.rnn1 = nn.LSTM(
      input_size=input_dim,
      hidden_size=input_dim,
      num_layers=1,
      batch_first=True
    )

    self.rnn2 = nn.LSTM(
      input_size=input_dim,
      hidden_size=self.hidden_dim,
      num_layers=1,
      batch_first=True
    )

    self.output_layer = nn.Linear(self.hidden_dim, n_features)

  def forward(self, x):
    x = x.repeat(self.seq_len, self.n_features)
    x = x.reshape((self.n_features, self.seq_len, self.input_dim))

    x, (hidden_n, cell_n) = self.rnn1(x)
    x, (hidden_n, cell_n) = self.rnn2(x)
    x = x.reshape((self.seq_len, self.hidden_dim))

    return self.output_layer(x)

# COMMAND ----------

class RecurrentAutoencoder(nn.Module):

  def __init__(self, seq_len, n_features, embedding_dim=64):
    super(RecurrentAutoencoder, self).__init__()

    self.encoder = Encoder(seq_len, n_features, embedding_dim).to(device)
    self.decoder = Decoder(seq_len, embedding_dim, n_features).to(device)

  def forward(self, x):
    x = self.encoder(x)
    x = self.decoder(x)

    return x

# COMMAND ----------

model = RecurrentAutoencoder(seq_len = 32768, n_features = 1, embedding_dim = 128)
model = model.to(device)

# COMMAND ----------

def train_model(model, train_dataset, val_dataset, n_epochs):
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
  criterion = nn.L1Loss(reduction='sum').to(device)
  history = dict(train=[], val=[])

  best_model_wts = copy.deepcopy(model.state_dict())
  best_loss = 10000.0
  
  for epoch in range(1, n_epochs + 1):
    model = model.train()

    train_losses = []
    for seq_true in train_dataset:
      optimizer.zero_grad()

      seq_true = seq_true.to(device)
      seq_pred = model(seq_true)

      loss = criterion(seq_pred, seq_true)

      loss.backward()
      optimizer.step()

      train_losses.append(loss.item())

    val_losses = []
    model = model.eval()
    with torch.no_grad():
      for seq_true in val_dataset:

        seq_true = seq_true.to(device)
        seq_pred = model(seq_true)

        loss = criterion(seq_pred, seq_true)
        val_losses.append(loss.item())

    train_loss = np.mean(train_losses)
    val_loss = np.mean(val_losses)

    history['train'].append(train_loss)
    history['val'].append(val_loss)

    if val_loss < best_loss:
      best_loss = val_loss
      best_model_wts = copy.deepcopy(model.state_dict())

    print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss}')

  model.load_state_dict(best_model_wts)
  return model.eval(), history

# COMMAND ----------

summary(model)

# COMMAND ----------

model, history = train_model(model, train_set, val_set, n_epochs = 50)

# COMMAND ----------

x = torch.randn(10, 3)

# COMMAND ----------

train_set[0].shape

# COMMAND ----------

from sequitur.models import LSTM_AE

model = LSTM_AE(
  input_dim=4000,
  encoding_dim = 1000,
  h_dims=[64],
  h_activ=None,
  out_activ=None
)

x = torch.randn(10, 4000) # Sequence of 32K 1D vectors
z = model.encoder(x) # z.shape = [1000]
x_prime = model.decoder(z, seq_len=10) # x_prime.shape = [10, 1]


# COMMAND ----------

model = LSTM_AE(
  input_dim = 32000,
  encoding_dim = 1000,
  h_dims = [64],
  h_activ = None,
  out_activ = None
)

x = torch.randn(1015, 32000) # Sequence of 32K 1D vectors
z = model.encoder(x) # z.shape = [1000]
x_prime = model.decoder(z, seq_len=10) # x_prime.shape = [10, 1]


# COMMAND ----------

summary(model.encoder)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # 3. Train on background
# MAGIC
# MAGIC I'm also going to train on background data rather than signal, so that (hopefully) we get higher reconstruction error when passing in vectors with BNS signal mixed in with the noise (fingers crossed)! This then would allow us to classify these higher-error segments as "anomalous" and probably containing signal (ie a binary neutron star merger!).

# COMMAND ----------

# ------------
# Data
# ------------

# Simulated data that's been bandpassed and whitened
sql = """
SELECT *
FROM mergers.background
WHERE event_id < 100
"""

gw_events = client.query(sql).to_dataframe()

sql = """
SELECT *
FROM mergers.bns_generated
WHERE event_id < 100
"""

bns_gen = client.query(sql).to_dataframe()

# ---------------------------------
# Data pre-processing for torch
# ---------------------------------

from sklearn.model_selection import train_test_split

gw_wide = gw_events.pivot(index = 'event_id', columns = 'time_seq', values = 'strain')
gw_train, gw_val = train_test_split(gw_wide, test_size = 0.1, random_state = 23)

train_np = gw_train.to_numpy() # 11 x 32K (11 examples of length 32K timesteps)
val_np = gw_val.to_numpy()

# Split background into train/val
train_set = [torch.tensor(x).float() for x in train_np]
val_set = [torch.tensor(x).float() for x in val_np]

# Convert BNS signal into tensor data
bns_wide = bns_gen.pivot(index = 'event_id', columns = 'time_seq', values = 'strain')
bns_np = bns_wide.to_numpy()
bns_val = [torch.tensor(x).float() for x in bns_np]

# --------------------
# Train autoencoder
# --------------------

# Length of examples must be a multiple of encoding_dim (size of z)
encoder, decoder, encodings, losses = quick_train(LINEAR_AE, train_set, encoding_dim = 2048, denoise = True, verbose = False)

# COMMAND ----------

# Get x' for BNS validation, keep original
bns_xprime = []
bns_z = []

for i, ele in enumerate(bns_val):
  z = encoder(ele) # Pass through new observation
  z_np = z.detach().cpu().numpy()
  bns_z.append(z_np)

  x_prime = decoder(z)
  x_prime_np = x_prime.detach().cpu().numpy()
  bns_xprime.append({'event_id': i, 'pred': x_prime_np, 'actual': ele.detach().cpu().numpy(), 'target': 'BNS'})

# BNS validation x' df
bns_xprime_pd = [pd.DataFrame(x) for x in bns_xprime]
bns_xprime_df = pd.concat(bns_xprime_pd) 

# Get x' for background validation set, keep original
bgd_xprime = []
bgd_z = []

for i, ele in enumerate(val_set):
  z = encoder(ele) # Pass through new observation
  z_np = z.detach().cpu().numpy()
  bgd_z.append(z_np)

  x_prime = decoder(z)
  x_prime_np = x_prime.detach().cpu().numpy()
  bgd_xprime.append({'event_id': i, 'pred': x_prime_np, 'actual': ele.detach().cpu().numpy(), 'target': 'background'})

# Background validation x' df
bgd_xprime_pd = [pd.DataFrame(x) for x in bgd_xprime]
bgd_xprime_df = pd.concat(bgd_xprime_pd)

# Stack validation sets
spark.createDataFrame(bns_xprime_df).createOrReplaceTempView('bns_xprime')
spark.createDataFrame(bgd_xprime_df).createOrReplaceTempView('bgd_xprime')

# COMMAND ----------

# MAGIC %r
# MAGIC # library(SparkR)
# MAGIC # library(tidyverse)
# MAGIC
# MAGIC
# MAGIC ae_preds <- SparkR::collect(SparkR::sql('SELECT * FROM bns_xprime UNION SELECT * FROM bgd_xprime;'))

# COMMAND ----------

# MAGIC %r
# MAGIC options(repr.plot.width = 1800, repr.plot.height = 1000)

# COMMAND ----------

# MAGIC %r
# MAGIC
# MAGIC # Avg MSE for each window
# MAGIC ae_preds %>%
# MAGIC mutate(mse = (actual - pred)^2) %>%
# MAGIC group_by(event_id, target) %>%
# MAGIC summarize(mean_mse = mean(mse),
# MAGIC           sum_mse = sum(mse)) %>%
# MAGIC ggplot(aes(mean_mse, fill = target, color = target)) +
# MAGIC geom_density(alpha = 0.3, size = 1.7) +
# MAGIC scale_fill_manual(values = c(purple_dark, "white")) +
# MAGIC scale_color_manual(values = c(purple_dark, "white")) +
# MAGIC labs(x = 'MSE',
# MAGIC fill = '',
# MAGIC color = '') +
# MAGIC theme_tech

# COMMAND ----------

# MAGIC %r
# MAGIC
# MAGIC # Avg MSE for each window
# MAGIC ggplot(ae_preds, aes(actual, pred, color = target)) +
# MAGIC geom_point(size = 1.3, alpha = 0.3) +
# MAGIC geom_smooth(method = "lm", se = F) +
# MAGIC scale_color_manual(values = c(purple_dark, "white")) +
# MAGIC labs(x = 'MSE',
# MAGIC fill = '',
# MAGIC color = '') +
# MAGIC theme_tech
