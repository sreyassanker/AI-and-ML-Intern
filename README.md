# AI-and-ML-Intern
import pandas as pd
from google.colab import files # Keep this import just in case you need file upload later in other cells

# Step 1: Load the CSV file into a DataFrame (assuming the file already exists from a previous upload)
# If the file does not exist, you will need to uncomment the following lines:
# from google.colab import files
# uploaded = files.upload('Titanic-Dataset.csv')
# filename = list(uploaded.keys())[0]

filename = 'Titanic-Dataset.csv' # Define the filename since we are not using files.upload()
df = pd.read_csv(filename)

# Step 4: Show first 5 rows (to get a peek at the data)
print("First 5 rows:")
print(df.head())

# Step 5: Check for null values per column
print("\nNumber of nulls in each column:")
print(df.isnull().sum())

# Step 6: Check data types of each column
print("\nData types of each column:")
print(df.dtypes)

# Optional: Print shape of the DataFrame (rows, columns)
print("\nShape of the dataset:")
print(df.shape)

# prompt: Handle missing values using mean/median/imputation.

# Identify columns with missing values
cols_with_missing = df.columns[df.isnull().any()].tolist()
print(f"\nColumns with missing values: {cols_with_missing}")

# Strategy 1: Impute missing values with the mean (for numerical columns)
numerical_cols = df[cols_with_missing].select_dtypes(include=['number']).columns
for col in numerical_cols:
  mean_value = df[col].mean()
  df[col].fillna(mean_value, inplace=True)
  print(f"Imputed missing values in column '{col}' with the mean ({mean_value:.2f})")

# Strategy 2: Impute missing values with the median (for numerical columns)
# You can choose either mean or median based on the data distribution (e.g., use median if data is skewed)
# For example, to use median for 'Age':
# if 'Age' in cols_with_missing:
#   median_value = df['Age'].median()
#   df['Age'].fillna(median_value, inplace=True)
#   print(f"Imputed missing values in column 'Age' with the median ({median_value:.2f})")

# Strategy 3: Impute missing values with the mode (for categorical columns)
categorical_cols = df[cols_with_missing].select_dtypes(include=['object', 'category']).columns
for col in categorical_cols:
  # Calculate the mode, which might return multiple values if there's a tie
  mode_value = df[col].mode()[0] # Take the first mode if there are multiple
  df[col].fillna(mode_value, inplace=True)
  print(f"Imputed missing values in column '{col}' with the mode ('{mode_value}')")

# Verify that missing values have been handled
print("\nNumber of nulls in each column after imputation:")
print(df.isnull().sum())

# prompt: Convert categorical features into numerical using encoding

# Step 8: Identify categorical columns that are not numerical (object or category dtype)
categorical_cols_for_encoding = df.select_dtypes(include=['object', 'category']).columns.tolist()

# Remove columns that might have been handled differently (e.g., 'Cabin' if you planned to drop it or use a different strategy)
# For this example, let's assume we want to encode 'Sex', 'Embarked', and potentially other relevant categorical columns.
# We will exclude 'Name' as it's likely unique and not useful for encoding.
# We will also exclude 'Ticket' as it might be complex to encode effectively without further feature engineering.
columns_to_encode = ['Sex', 'Embarked'] # Specify the columns you want to encode

print(f"\nColumns identified for encoding: {columns_to_encode}")

# Apply one-hot encoding using pd.get_dummies
df = pd.get_dummies(df, columns=columns_to_encode, drop_first=True) # drop_first=True avoids multicollinearity

print("\nDataFrame after one-hot encoding:")
print(df.head())

print("\nData types after encoding:")
df.dtypes

# prompt: Normalize/standardize the numerical features

from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Identify numerical columns for normalization/standardization
# Exclude dummy variables created during one-hot encoding and original non-numeric columns
numerical_features = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

# Remove columns that are IDs, labels, or were handled differently (like original PassengerId or dummy variables)
# We will keep 'Age', 'Fare', 'Pclass' as examples of numerical features to scale.
# Pclass is technically categorical, but sometimes treated as ordinal/numerical for scaling.
# Adjust this list based on the actual numerical features you want to scale.
features_to_scale = ['Age', 'Fare', 'Pclass']
numerical_features_to_scale = [col for col in features_to_scale if col in numerical_features]

print(f"\nNumerical features identified for scaling: {numerical_features_to_scale}")

# Option 1: Standardization (Z-score normalization)
# Scales data to have a mean of 0 and standard deviation of 1
scaler_standard = StandardScaler()
df[numerical_features_to_scale] = scaler_standard.fit_transform(df[numerical_features_to_scale])

print("\nDataFrame after Standardization (first 5 rows of scaled features):")
print(df[numerical_features_to_scale].head())

# Option 2: Min-Max Scaling
# Scales data to a fixed range, usually 0 to 1
# To use this, you would comment out the Standardization part above and uncomment this:
# scaler_minmax = MinMaxScaler()
# df[numerical_features_to_scale] = scaler_minmax.fit_transform(df[numerical_features_to_scale])

# print("\nDataFrame after Min-Max Scaling (first 5 rows of scaled features):")
# print(df[numerical_features_to_scale].head())

# Verify the scaling by looking at the distribution of the scaled columns
# print("\nDescription of scaled numerical features:")
# print(df[numerical_features_to_scale].describe())

# prompt: Visualize outliers using boxplots and remove them

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Visualize outliers using boxplots
numerical_cols_for_outliers = ['Age', 'Fare'] # Select numerical columns to check for outliers

print("\nVisualizing outliers using boxplots:")
for col in numerical_cols_for_outliers:
  plt.figure(figsize=(8, 4))
  sns.boxplot(x=df[col])
  plt.title(f'Boxplot of {col}')
  plt.show()

# Remove outliers using the Interquartile Range (IQR) method
# Function to remove outliers using IQR
def remove_outliers_iqr(df, column):
  Q1 = df[column].quantile(0.25)
  Q3 = df[column].quantile(0.75)
  IQR = Q3 - Q1
  lower_bound = Q1 - 1.5 * IQR
  upper_bound = Q3 + 1.5 * IQR
  df_cleaned = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)].copy()
  print(f"Original shape: {df.shape}")
  print(f"Shape after removing outliers in {column}: {df_cleaned.shape}")
  return df_cleaned

# Apply outlier removal to the selected numerical columns
df_cleaned = df.copy() # Create a copy to work with
for col in numerical_cols_for_outliers:
  df_cleaned = remove_outliers_iqr(df_cleaned, col)

# Update the main DataFrame reference if you want to use the cleaned data moving forward
df = df_cleaned

print("\nDataFrame after removing outliers:")
print(df.head())

# Verify that outliers have been removed by visualizing again (optional)
# print("\nVisualizing data after outlier removal:")
# for col in numerical_cols_for_outliers:
#   plt.figure(figsize=(8, 4))
#   sns.boxplot(x=df[col])
#   plt.title(f'Boxplot of {col} (After Outlier Removal)')
#   plt.show()
