# Databricks notebook source
# ------------
# Packages
# ------------

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

!pip install sequitur
import torch
from sequitur.models import LINEAR_AE
from sequitur import quick_train

# sequitur automatically uses GPU if it's available, so we don't need to worry about this
# torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

!pip install torchinfo
from torchinfo import summary

# Raise pivot limits
spark.conf.set("spark.sql.pivotMaxValues", 32768)

# COMMAND ----------

# MAGIC %r
# MAGIC
# MAGIC # Packages
# MAGIC install.packages("sysfonts")
# MAGIC library(sysfonts)
# MAGIC library(tidyverse)
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
# MAGIC # 0. Custom functions

# COMMAND ----------

class GravWaveData:

  def __init__(self):
    self.gw_df = None

  def get_gw_df(self, event_id_range: list, input_schema: str):

    min_event_id, max_event_id = event_id_range[0], event_id_range[1]

    # Bring in background grav-wave data
    gw_spk = spark.sql(f'''SELECT * --except(time_seq), CONCAT('t_', time_seq) AS time_seq 
                      FROM {input_schema}
                      WHERE event_id BETWEEN {min_event_id} AND {max_event_id};''')

    gw_pvt = gw_spk.groupBy('event_id').pivot('time_seq').sum('strain')

    # Convert to pandas
    gw_pd = gw_pvt.toPandas()
    self.gw_df = gw_pd.drop(columns = ['event_id'])

    return self.gw_df

  def split_gw_df(self, test_size: float = 0.1, random_state: int = 23):

    if self.gw_df is None:
      raise Exception('self.gw_df is None. Run the get_gw_df() method first to pull in data from schema and pivot.')

    # Split into train/test
    self.train_df, self.val_df = train_test_split(self.gw_df, test_size = test_size, random_state = random_state)

    return self.train_df, self.val_df

  def convert_df_to_tensor(self, dfs_to_convert: list):

    self.data_np = [x.to_numpy() for x in dfs_to_convert]

    self.data_tensor = []
    for i in range(len(self.data_np)):
      self.data_tensor.append([torch.tensor(x).float() for x in self.data_np[i]])

    return self.data_tensor

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # 1a. Train a linear AE on background, latent dims = 2048

# COMMAND ----------

# ----------------------------
# Preprocess background data
# ----------------------------

background = GravWaveData()
bgd_pd = background.get_gw_df(event_id_range = [0, 1500], input_schema = 'user_nsulliv3.gw_bg')
bgd_train_df, bgd_val_df = background.split_gw_df(test_size = 0.1)
bgd_tensors = background.convert_df_to_tensor([bgd_train_df, bgd_val_df])
bgd_train, bgd_val = [x for x in bgd_tensors]

# --------------------
# Train autoencoder
# --------------------

# Length of examples must be a multiple of encoding_dim (size of z)
encoder, decoder, encodings, losses = quick_train(LINEAR_AE, bgd_train, encoding_dim = 2048, denoise = True, verbose = False)

torch.save(encoder, "/dbfs/models/gw_bgd_2048_encoder.pt")
torch.save(decoder, "/dbfs/models/gw_bgd_2048_decoder.pt")


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # 1b. Evaluate linear AE, latent dims = 2048
# MAGIC
# MAGIC * Runtime: 6.49 hours
# MAGIC * Epochs: 50
# MAGIC * LR: 1e-3

# COMMAND ----------

# ---------------------------------------
# Load in saved models, get BNS val data
# ---------------------------------------

encoder = torch.load('/dbfs/models/gw_bgd_2048_encoder.pt')
decoder = torch.load('/dbfs/models/gw_bgd_2048_decoder.pt')

# Pull some BNS data, and use all of it for validation (along with background val set)
bns = GravWaveData()
bns_pd = bns.get_gw_df(event_id_range = [1501, 2000], input_schema = 'user_nsulliv3.gw_bns')
bns_tensors = bns.convert_df_to_tensor([bns_pd])
bns_val = bns_tensors[0]

# -----------------------
# Predict on val sets
# -----------------------

# Get x' and z for validation set
xprime_ls = []
z_ls = []

for i, ele in enumerate(bns_val+bgd_val):
  z = encoder(ele) # Pass through new observation
  z_np = z.detach().cpu().numpy()
  z_ls.append(z_np)

  x_prime = decoder(z)
  x_prime_np = x_prime.detach().cpu().numpy()
  xprime_ls.append({'event_num': i, 'pred': x_prime_np, 'actual': ele.detach().cpu().numpy(), 'target': 'BNS' if i <= len(bns_val) else 'background'})

# Create x' spark view for visualization in R
xprime_pd = [pd.DataFrame(x) for x in xprime_ls]
xprime_df = pd.concat(xprime_pd)
spark.createDataFrame(xprime_df).write.mode('overwrite').option('overwriteSchema', 'true').saveAsTable('user_nsulliv3.gw_model_bgd_2048_xprime')

# COMMAND ----------

# MAGIC %r
# MAGIC library(SparkR)
# MAGIC library(tidyverse)
# MAGIC
# MAGIC
# MAGIC ae_preds <- SparkR::collect(SparkR::sql('SELECT * FROM xprime_2048;'))

# COMMAND ----------

# MAGIC %r
# MAGIC
# MAGIC # Set graph 
# MAGIC options(repr.plot.width = 1800, repr.plot.height = 1000)

# COMMAND ----------

# MAGIC %r
# MAGIC
# MAGIC # Avg MSE for each window
# MAGIC ae_preds %>%
# MAGIC mutate(mse = (actual - pred)^2) %>%
# MAGIC group_by(event_num, target) %>%
# MAGIC summarize(mean_mse = mean(mse),
# MAGIC           sum_mse = sum(mse)) %>%
# MAGIC ggplot(aes(mean_mse, fill = target, color = target)) +
# MAGIC geom_density(alpha = 0.3, size = 1.7) +
# MAGIC scale_fill_manual(values = c(purple_dark, "white")) +
# MAGIC scale_color_manual(values = c(purple_dark, "white")) +
# MAGIC labs(x = "MSE",
# MAGIC fill = '',
# MAGIC color = '') +
# MAGIC theme_tech +
# MAGIC xlim(0, 1500)

# COMMAND ----------

# MAGIC %r
# MAGIC
# MAGIC # Avg MSE for each window
# MAGIC ae_preds %>%
# MAGIC mutate(mse = (actual - pred)^2) %>%
# MAGIC group_by(event_num, target) %>%
# MAGIC summarize(mean_mse = mean(mse),
# MAGIC           sum_mse = sum(mse)) %>%
# MAGIC ungroup() %>%
# MAGIC mutate(mean_mse = ifelse(target == "BNS", mean_mse+100, mean_mse)) %>%
# MAGIC ggplot(aes(mean_mse, fill = target, color = target)) +
# MAGIC geom_histogram(alpha=0.4, position="identity", size = 0.2, bins = 100) +
# MAGIC scale_fill_manual(values = c(purple_dark, "white")) +
# MAGIC scale_color_manual(values = c(purple_dark, "white")) +
# MAGIC labs(x = "MSE",
# MAGIC fill = '',
# MAGIC color = '') +
# MAGIC theme_tech +
# MAGIC xlim(0, 2000)

# COMMAND ----------

# MAGIC %r
# MAGIC
# MAGIC # Avg MSE for each window
# MAGIC ggplot(ae_preds, aes(actual, pred, color = target)) +
# MAGIC geom_point(size = 1.8, alpha = 0.3) +
# MAGIC geom_smooth(method = "lm", se = F) +
# MAGIC scale_color_manual(values = c(purple_dark, "white")) +
# MAGIC labs(x = 'MSE',
# MAGIC fill = '',
# MAGIC color = '') +
# MAGIC theme_tech

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # 2a. Train a linear AE, latent dims = 8192
# MAGIC * Runtime: 2.09 hours
# MAGIC * Epochs: 50
# MAGIC * LR: 1x10^-3

# COMMAND ----------

# For efficiency (training set of 1200 examples took 6.5 hrs to train), test out on small train set
bgd_train_sm = bgd_train[:100]

# --------------------
# Train autoencoder
# --------------------

latent_dims = 8192

# Length of examples must be a multiple of encoding_dim (size of z)
encoder, decoder, encodings, losses = quick_train(LINEAR_AE, bgd_train_sm, encoding_dim = latent_dims, denoise = True, verbose = False)

torch.save(encoder, f'/dbfs/models/gw_bgd_{latent_dims}_sm_encoder.pt')
torch.save(decoder, f'/dbfs/models/gw_bgd_{latent_dims}_sm_decoder.pt')

spark.createDataFrame(pd.DataFrame({'loss': losses, 'epoch': np.arange(1, len(losses)+1)})).write.mode('overwrite').option('overwriteSchema', 'true').saveAsTable('user_nsulliv3.gw_model_bgd_8192_loss')

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # 2b. Evaluate linear AE, latent dims = 8192

# COMMAND ----------

# MAGIC %r 
# MAGIC
# MAGIC loss_8192 <- SparkR::collect(SparkR::sql('SELECT * FROM user_nsulliv3.gw_model_bgd_8192_loss;'))

# COMMAND ----------

# MAGIC %r
# MAGIC
# MAGIC ggplot(loss_8192, aes(epoch, loss)) +
# MAGIC geom_line()

# COMMAND ----------

# Pull some BNS data, and use all of it for validation (along with background val set)
bns = GravWaveData()
bns_pd = bns.get_gw_df(event_id_range = [1501, 2000], input_schema = 'user_nsulliv3.gw_bns')
bns_tensors = bns.convert_df_to_tensor([bns_pd])
bns_val = bns_tensors[0]

# -----------------------
# Predict on val sets
# -----------------------

# Get x' and z for validation set
xprime_ls = []
z_ls = []

for i, ele in enumerate(bns_val+bgd_val):
  z = encoder(ele) # Pass through new observation
  z_np = z.detach().cpu().numpy()
  z_ls.append(z_np)

  x_prime = decoder(z)
  x_prime_np = x_prime.detach().cpu().numpy()
  xprime_ls.append({'event_num': i, 'pred': x_prime_np, 'actual': ele.detach().cpu().numpy(), 'target': 'BNS' if i <= len(bns_val) else 'background'})

# Create x' spark view for visualization in R
xprime_pd = [pd.DataFrame(x) for x in xprime_ls]
xprime_df = pd.concat(xprime_pd)
spark.createDataFrame(xprime_df).write.mode('overwrite').option('overwriteSchema', 'true').saveAsTable('user_nsulliv3.gw_model_bgd_8192_xprime')

# COMMAND ----------

# MAGIC %r 
# MAGIC
# MAGIC ae_preds <- SparkR::collect(SparkR::sql('SELECT * FROM user_nsulliv3.gw_model_bgd_8192_xprime;'))

# COMMAND ----------

# MAGIC %r
# MAGIC options(repr.plot.width = 2200, repr.plot.height = 1000)

# COMMAND ----------

# MAGIC %r
# MAGIC
# MAGIC # Avg MSE for each window
# MAGIC ae_preds %>%
# MAGIC mutate(mse = (actual - pred)^2) %>%
# MAGIC group_by(event_num, target) %>%
# MAGIC summarize(med_mse = median(mse),
# MAGIC           sum_mse = sum(mse)) %>%
# MAGIC # ungroup() %>%
# MAGIC mutate(med_mse = ifelse(target == "BNS", med_mse+100, med_mse)) %>%
# MAGIC ggplot(aes(med_mse, fill = target, color = target)) +
# MAGIC # geom_density(alpha = 0.3, size = 1.2) +
# MAGIC geom_histogram(alpha=0.4, position="identity", size = 0.2, bins = 100) +
# MAGIC scale_fill_manual(values = c(purple_dark, "white")) +
# MAGIC scale_color_manual(values = c(purple_dark, "white")) +
# MAGIC labs(x = "MSE",
# MAGIC fill = '',
# MAGIC color = '') +
# MAGIC theme_tech +
# MAGIC xlim(0, 2000)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # 3a. Train a linear AE on scaled data, latent dims = 2048
# MAGIC * Epochs: 10
# MAGIC * LR: 5e-4
# MAGIC * Runtime: 17 min

# COMMAND ----------

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

# COMMAND ----------

bg_train_sm = background.data_np[0][:50]
X_train = scaler.fit_transform(bg_train_sm)

train_scaled_tensor = [torch.tensor(x).float() for x in X_train]

encoder, decoder, encodings, losses = quick_train(LINEAR_AE, train_scaled_tensor, encoding_dim = 2048, denoise = True, verbose = False, lr = 5e-4, epochs = 10)

spark.createDataFrame(pd.DataFrame({'loss': losses, 'epoch': np.arange(1, 51)})).createOrReplaceTempView('ae_scaled_2048')

# COMMAND ----------

# MAGIC %r 
# MAGIC
# MAGIC ae_loss <- SparkR::collect(SparkR::sql('SELECT * FROM ae_scaled_2048'))
# MAGIC
# MAGIC ggplot(ae_loss, aes(epoch, loss)) +
# MAGIC geom_line()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # 3b. Evaluate linear AE trained on scaled data, latent dims = 2048

# COMMAND ----------

bg_val_np = background.data_np[1][11:100] # 2nd index is validation set
bg_val_scaled = scaler.transform(bg_val_np)
bg_val_tensor = [torch.tensor(x).float() for x in bg_val_scaled]

bns_val_np = bns.data_np[0][11:100]
bns_val_scaled = scaler.transform(bns_val_np)
bns_val_tensor = [torch.tensor(x).float() for x in bns_val_scaled]

# -----------------------
# Predict on val sets
# -----------------------

# Get x' and z for validation set
xprime_ls = []
z_ls = []

for i, ele in enumerate(bns_val_tensor+bg_val_tensor):
  z = encoder(ele) # Pass through new observation
  z_np = z.detach().cpu().numpy()
  z_ls.append(z_np)

  x_prime = decoder(z)
  x_prime_np = x_prime.detach().cpu().numpy()
  xprime_ls.append({'event_num': i, 'pred': x_prime_np, 'actual': ele.detach().cpu().numpy(), 'target': 'BNS' if i <= len(bns_val_tensor) else 'background'})

# Create x' spark view for visualization in R
xprime_pd = [pd.DataFrame(x) for x in xprime_ls]
xprime_df = pd.concat(xprime_pd)
spark.createDataFrame(xprime_df).createOrReplaceTempView('ae_scaled_2048_preds')

# COMMAND ----------

# MAGIC %r 
# MAGIC
# MAGIC # ae_preds <- SparkR::collect(SparkR::sql('SELECT * FROM ae_scaled_2048_preds;'))
# MAGIC
# MAGIC # Avg MSE for each window
# MAGIC ae_preds %>%
# MAGIC mutate(mse = (actual - pred)^2) %>%
# MAGIC group_by(event_num, target) %>%
# MAGIC summarize(mean_mse = mean(mse),
# MAGIC           sum_mse = sum(mse)) %>%
# MAGIC # ungroup() %>%
# MAGIC # mutate(med_mse = ifelse(target == "BNS", med_mse+100, med_mse)) %>%
# MAGIC ggplot(aes(mean_mse, fill = target, color = target)) +
# MAGIC # geom_density(alpha = 0.3, size = 1.2) +
# MAGIC geom_histogram(alpha=0.4, position="identity", size = 0.2, bins = 100) +
# MAGIC scale_fill_manual(values = c(purple_dark, "white")) +
# MAGIC scale_color_manual(values = c(purple_dark, "white")) +
# MAGIC labs(x = "MSE",
# MAGIC fill = '',
# MAGIC color = '') +
# MAGIC theme_tech +
# MAGIC xlim(0, 1)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # 4a. Train a linear AE, latent dims = 16384
# MAGIC * Trained on: 100 samples
# MAGIC * Epochs: 15
# MAGIC * LR: 5e-4
# MAGIC * Runtime: 1.15 hours

# COMMAND ----------

# --------------------
# Train autoencoder
# --------------------

latent_dims = 16384

# Length of examples must be a multiple of encoding_dim (size of z)
encoder, decoder, encodings, losses = quick_train(LINEAR_AE, bgd_train_sm, epochs = 15, lr = 5e-4, encoding_dim = latent_dims, denoise = True, verbose = True)

torch.save(encoder, f'/dbfs/models/gw_bgd_{latent_dims}_encoder.pt')
torch.save(decoder, f'/dbfs/models/gw_bgd_{latent_dims}_decoder.pt')

spark.createDataFrame(pd.DataFrame({'loss': losses, 'epoch': np.arange(1, len(losses)+1)})).write.mode('overwrite').option('overwriteSchema', 'true').saveAsTable('user_nsulliv3.gw_model_bgd_16384_loss')

# COMMAND ----------

# -----------------------
# Predict on val sets
# -----------------------

# Get x' and z for validation set
xprime_ls = []
z_ls = []

for i, ele in enumerate(bns_val+bgd_val):
  z = encoder(ele) # Pass through new observation
  z_np = z.detach().cpu().numpy()
  z_ls.append(z_np)

  x_prime = decoder(z)
  x_prime_np = x_prime.detach().cpu().numpy()
  xprime_ls.append({'event_num': i, 'pred': x_prime_np, 'actual': ele.detach().cpu().numpy(), 'target': 'BNS' if i <= len(bns_val) else 'background'})

# Create x' spark view for visualization in R
xprime_pd = [pd.DataFrame(x) for x in xprime_ls]
xprime_df = pd.concat(xprime_pd)
spark.createDataFrame(xprime_df).write.mode('overwrite').option('overwriteSchema', 'true').saveAsTable('user_nsulliv3.gw_model_bgd_16384_xprime')

# COMMAND ----------

# MAGIC %r 
# MAGIC
# MAGIC ae_preds <- SparkR::collect(SparkR::sql('SELECT * FROM user_nsulliv3.gw_model_bgd_16384_xprime;'))

# COMMAND ----------

# MAGIC %r
# MAGIC ae_preds %>%
# MAGIC mutate(mse = (actual - pred)^2) %>%
# MAGIC group_by(event_num, target) %>%
# MAGIC summarize(mean_mse = mean(mse),
# MAGIC           sum_mse = sum(mse))

# COMMAND ----------

# MAGIC %r
# MAGIC
# MAGIC ae16k_preds <- SparkR::collect(SparkR::sql("
# MAGIC WITH mse AS (SELECT target, event_num, pow((actual-pred), 2) as pred_mse
# MAGIC FROM user_nsulliv3.gw_model_bgd_16384_xprime)
# MAGIC
# MAGIC SELECT target, event_num, AVG(pred_mse) AS mean_mse 
# MAGIC FROM mse 
# MAGIC GROUP BY 1, 2;
# MAGIC
# MAGIC "))

# COMMAND ----------

# MAGIC %r  
# MAGIC ae16k_preds

# COMMAND ----------

# MAGIC %r 
# MAGIC
# MAGIC # Avg MSE for each window
# MAGIC ae16k_preds %>%
# MAGIC filter(mean_mse < 2000) %>%
# MAGIC # ungroup() %>%
# MAGIC mutate(mean_mse = ifelse(target == "BNS", mean_mse+20, mean_mse)) %>%
# MAGIC ggplot(aes(mean_mse, fill = target, color = target)) +
# MAGIC geom_density(alpha = 0.3, size = 1.2) +
# MAGIC # geom_histogram(alpha=0.4, position="identity", bins = 100) +
# MAGIC scale_fill_manual(values = c(purple_dark, "white")) +
# MAGIC scale_color_manual(values = c(purple_dark, "white")) +
# MAGIC labs(x = "MSE",
# MAGIC fill = '',
# MAGIC color = '') +
# MAGIC theme_tech

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # 5. Create plot of 2048, 8192, and 16,384 latent dims all together

# COMMAND ----------

spark.sql('''
WITH mse AS (SELECT target, event_num, pow((actual-pred), 2) as pred_mse
FROM user_nsulliv3.gw_model_bgd_2048_xprime),

av_mse AS (SELECT target, event_num, AVG(pred_mse) AS mean_mse 
FROM mse 
GROUP BY 1, 2),

metrics_prep AS (SELECT target, 
CASE WHEN target = 'background' THEN 0 ELSE 1 END AS target_binary,
CASE WHEN target = 'BNS' THEN mean_mse+10 ELSE mean_mse END AS mean_mse
FROM av_mse),

binary_preds AS (
  SELECT *,
CASE WHEN mean_mse BETWEEN 500 AND 2000 THEN 1 ELSE 0 END AS pred_binary
FROM metrics_prep
)


  SELECT CASE WHEN target_binary = 0 AND pred_binary = 0 THEN 'tn'
WHEN target_binary = 1 AND pred_binary = 1 THEN 'tp'
WHEN target_binary = 0 AND pred_binary = 1 THEN 'fp'
ELSE 'fn' END AS pred_category, count(*) AS n_obs
FROM binary_preds
GROUP BY 1;
''')
