#%%
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sn

#%%
df = pd.read_csv("combined.csv")

print(df.head())
print(df.info())
# %%
print(f"Dataset has {df.shape[0]} rows and {df.shape[1]} columns.")

print(df.isnull().sum())

print(df.describe())
# %%
data = df.drop_duplicates()

#%%
# Convert all boolean columns to integers
bool_cols = data.select_dtypes(include='bool').columns
data[bool_cols] = data[bool_cols].astype('Int64')

print(data.info())

#%%
# Date Format
data['upload_date'] = pd.to_datetime(data['upload_date'], format='%Y%m%d', errors='coerce')

data['upload_date'] = data['upload_date'].dt.strftime('%Y/%m/%d')

print(data['upload_date'].head())

# %%
print(data.head())

# %%
