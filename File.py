#%%
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

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

data = data.sort_values(by='upload_date', ascending=True)

print(data['upload_date'].head())

# %%
print(data.head())

print(data.info())

#%%
data['uploader_sub_count'] = data['uploader_sub_count'].apply(lambda x: np.nan if x < 0 else x)
data = data.dropna(subset=['uploader_sub_count'])

#%%
# Removing the outliers 

numeric_columns = ['uploader_sub_count', 'view_count', 'like_count', 'dislike_count']
Q1 = data[numeric_columns].quantile(0.25)
Q3 = data[numeric_columns].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

data_cleaned = data[~((data[numeric_columns] < lower_bound) | (data[numeric_columns] > upper_bound)).any(axis=1)]
print(data_cleaned.info())

print(data_cleaned.head())

# %%
data_without_date = data.drop(columns=['upload_date'])

# Correlation plot 
plt.figure(figsize=(12, 8))
sns.heatmap(data_without_date.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# %%

data['upload_date'] = pd.to_datetime(data['upload_date'], errors='coerce')

plt.figure(figsize=(12, 6))
plt.scatter(data['upload_date'], data['view_count'], color='b', marker='o')
plt.xlabel('Date')
plt.ylabel('View Count')
plt.title('View Count Over Time (Scatter Plot)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %%
