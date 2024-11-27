# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, PowerTransformer

# Load data
df = pd.read_csv("C:/Users/g/OneDrive/Desktop/GWU/202408/DATS 6103 Intro to Data Mining/Mod 3/Project/combined.csv")

# Display basic info
print(df.head())
print(df.info())

# %%
print(f"Dataset has {df.shape[0]} rows and {df.shape[1]} columns.")
print(df.isnull().sum())
print(df.describe())

#%%

# Remove duplicates
data = df.drop_duplicates()

#%%

# Convert boolean columns to integers
bool_cols = data.select_dtypes(include='bool').columns
data[bool_cols] = data[bool_cols].astype('Int64')

#%%
# Format and sort dates
data['upload_date'] = pd.to_datetime(data['upload_date'], format='%Y%m%d', errors='coerce')
data['upload_date'] = data['upload_date'].dt.strftime('%Y/%m/%d')
data = data.sort_values(by='upload_date', ascending=True)
data = data.dropna()

data['upload_date'].unique()

#%%

# Handle invalid `uploader_sub_count` values
data['uploader_sub_count'] = data['uploader_sub_count'].apply(lambda x: np.nan if x < 0 else x)
data = data.dropna(subset=['uploader_sub_count'])

#%%
# Removing outliers using IQR
numeric_columns = ['uploader_sub_count', 'view_count', 'like_count', 'dislike_count']
Q1 = data[numeric_columns].quantile(0.25)
Q3 = data[numeric_columns].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

data_cleaned = data[~((data[numeric_columns] < lower_bound) | (data[numeric_columns] > upper_bound)).any(axis=1)]

#%%
# Rescale numeric features to handle scaling issues
scaler = MinMaxScaler()
data_cleaned[numeric_columns] = scaler.fit_transform(data_cleaned[numeric_columns])

# Feature engineering: Add a feature for engagement rate
data_cleaned['engagement_rate'] = (data_cleaned['like_count'] + data_cleaned['dislike_count']) / data_cleaned['view_count']

# Address non-linearity using transformations

# Can do log transformations
# Can change for all the columns
pt = PowerTransformer(method='yeo-johnson')
data_cleaned['view_count_transformed'] = pt.fit_transform(data_cleaned[['view_count']])

#%%
# Updated info
print(data_cleaned.info())

#%%
# Recheck correlations
data_without_date = data_cleaned.drop(columns=['upload_date'])

plt.figure(figsize=(12, 8))
sns.heatmap(data_without_date.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Updated Correlation Heatmap with Feature Engineering")
plt.show()

#%%
# Plot to explore binary features distribution
binary_features = ['has_subtitles', 'is_ads_enabled', 'is_comments_enabled', 'is_age_limit', 'is_live_content']
plt.figure(figsize=(14, 10))
for i, col in enumerate(binary_features, start=1):
    plt.subplot(2, 3, i)
    sns.countplot(x=data_cleaned[col])
    plt.title(f"Distribution of {col}")
plt.tight_layout()
plt.show()

# %%
