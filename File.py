# %% Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# %% Load and inspect the dataset
df = pd.read_csv("C:/Users/g/OneDrive/Desktop/GWU/202408/DATS 6103 Intro to Data Mining/Mod 3/Project/combined.csv")

print("Preview of the dataset:")
print(df.head())
print("\nDataset Information:")
print(df.info())

# %% Basic data overview
print(f"\nThe dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")
print("\nMissing values per column:")
print(df.isnull().sum())
print("\nSummary statistics for numeric columns:")
print(df.describe())

# %% Remove duplicate rows to ensure data consistency
data = df.drop_duplicates()
print("\nDuplicate rows removed.")

# %% Convert boolean columns to integers for easier handling
bool_cols = data.select_dtypes(include='bool').columns
data[bool_cols] = data[bool_cols].astype('Int64')
print("\nBoolean columns converted to integers.")

# %% Format and sort the `upload_date` column
data['upload_date'] = pd.to_datetime(data['upload_date'], format='%Y%m%d', errors='coerce')
data['upload_date'] = data['upload_date'].dt.strftime('%Y/%m/%d')
data = data.sort_values(by='upload_date', ascending=True).dropna(subset=['upload_date'])
print("\nDates formatted and sorted.")

# %% Handle invalid uploader_sub_count values (e.g., negative values)
data['uploader_sub_count'] = data['uploader_sub_count'].apply(lambda x: np.nan if x < 0 else x)
data = data.dropna(subset=['uploader_sub_count'])
print("\nInvalid uploader subscriber counts handled.")

# %% Remove outliers using the IQR method
numeric_columns = ['uploader_sub_count', 'view_count', 'like_count', 'dislike_count']
Q1 = data[numeric_columns].quantile(0.25)
Q3 = data[numeric_columns].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

data_cleaned = data[~((data[numeric_columns] < lower_bound) | (data[numeric_columns] > upper_bound)).any(axis=1)]
print("\nOutliers removed using the IQR method.")

# %% Scale numeric features to bring all values to a uniform range
scaler = MinMaxScaler()
data_cleaned[numeric_columns] = scaler.fit_transform(data_cleaned[numeric_columns])
print("\nNumeric features scaled.")

# Feature engineering: Add a new column for engagement rate
data_cleaned['engagement_rate'] = (data_cleaned['like_count'] + data_cleaned['dislike_count']) / data_cleaned['view_count']
print("\nFeature 'engagement_rate' added.")

# %% Apply log transformations and drop original columns
log_transform_columns = [
    'uploader_sub_count', 'view_count', 'engagement_rate', 
    'has_subtitles', 'is_ads_enabled', 'is_comments_enabled', 
    'is_age_limit', 'is_live_content'
]

plt.figure(figsize=(12, len(log_transform_columns) * 4))  # Dynamically adjust figure size
for i, col in enumerate(log_transform_columns):
    # Plot original distribution
    plt.subplot(len(log_transform_columns), 2, 2 * i + 1)
    sns.histplot(data_cleaned[col], kde=True, color='blue', bins=30)
    plt.title(f"Original Distribution of {col}")
    plt.xlabel(col)

    # Apply log transformation (add 1 to avoid log(0))
    data_cleaned[f'{col}_log'] = np.log1p(data_cleaned[col])

    # Plot transformed distribution
    plt.subplot(len(log_transform_columns), 2, 2 * i + 2)
    sns.histplot(data_cleaned[f'{col}_log'], kde=True, color='green', bins=30)
    plt.title(f"Log Transformed Distribution of {col}")
    plt.xlabel(f'{col}_log')

# Drop original columns after log transformation
data_cleaned.drop(columns=log_transform_columns, inplace=True)
print(f"\nOriginal columns dropped after log transformation: {log_transform_columns}")

plt.tight_layout()
plt.show()

# %% Recheck correlations with transformed features
log_columns = [f'{col}_log' for col in log_transform_columns]
plt.figure(figsize=(12, 8))
sns.heatmap(data_cleaned[log_columns].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap with Log-Transformed Features")
plt.show()

# %%
