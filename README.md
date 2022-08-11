# Background

This repo contains code that can be run from Google Colab to analyze gravitational wave merger detections since 2015 from the [Gravitational Wave Open Science Center (GWOSC)](https://www.gw-openscience.org/).

# Steps

## 00: stand up a Google BigQuery database

This notebook stands up a BigQuery database and fills it with GWOSC data.

![](img/gbq_q.png)

This notebook also adds metadata to the database so those accessing it know the type of data contained therein and from where the data originated.

![](img/gbq_desc.png)

Why stand up a database on Google Cloud rather than simply using the `gwpy` package or API calls? There are a few motivations for this:

1. SQL is an industry-standard querying language, which means that professionals across domains know and use SQL to pull data, and having a SQL database available makes exploration quick and convenient. Using `gwpy`, however, requires learning niche functions that are only applicable for pulling this particular set of gravitational wave data. The GBQ database makes this data accessible to a diverse set of backgrounds, thus opening it up to new discovery from cross-functional eyes.

2. Using SQL is generally more concise than calling APIs (even when applying complex logic) and since it's stored and pulled in the same format needed for exploration and modeling, it doesn't require any restructuring after pulling.

# 01: exploration and visualization

This notebook explores some descriptive statistics of the dataset, visualizes the data in different ways, and pulls data in using gwpy with the dataset's GPS as the primary key for the package.

![](img/01_final_masses.png)


