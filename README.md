🚢 Titanic Dataset – Data Preprocessing Pipeline (AI & ML Internship)
This project demonstrates a complete data preprocessing workflow on the classic Titanic dataset, covering data loading, inspection, missing value handling, encoding, scaling, outlier detection, and removal.
The pipeline is implemented using Python, Pandas, Scikit-learn, Seaborn, and Matplotlib, and is suitable as a foundation for machine learning model development.
📌 Project Objectives
Load and explore the Titanic dataset
Handle missing values using appropriate statistical strategies
Convert categorical variables into numerical form
Normalize / standardize numerical features
Detect and remove outliers using statistical methods
Prepare a clean dataset for downstream ML modeling
📂 Dataset
File name: Titanic-Dataset.csv
Source: Commonly used Titanic survival dataset
Description: Contains passenger information such as age, sex, fare, class, and survival status
🧰 Libraries Used
pandas
numpy
scikit-learn
seaborn
matplotlib
(Optionally uses google.colab.files for file uploads if required.)
🔹 Step 1: Load the Dataset
import pandas as pd
from google.colab import files

filename = 'Titanic-Dataset.csv'
df = pd.read_csv(filename)
🔹 Step 2: Initial Data Exploration
View first 5 rows
Identify missing values
Check data types
Inspect dataset shape
print(df.head())
print(df.isnull().sum())
print(df.dtypes)
print(df.shape)
🔹 Step 3: Handling Missing Values
✔ Identify columns with missing data
cols_with_missing = df.columns[df.isnull().any()].tolist()
✔ Numerical Columns – Mean Imputation
Missing values in numerical columns are filled using the mean value.
for col in numerical_cols:
    df[col].fillna(df[col].mean(), inplace=True)
✔ Categorical Columns – Mode Imputation
Missing values in categorical columns are filled using the most frequent value (mode).
for col in categorical_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)
✔ Verification
print(df.isnull().sum())
🔹 Step 4: Encoding Categorical Features
Selected categorical columns are converted into numerical format using One-Hot Encoding.
Encoded columns:
Sex
Embarked
Excluded columns:
Name (unique identifiers)
Ticket (requires advanced feature engineering)
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)
🔹 Step 5: Feature Scaling
Numerical features are normalized to ensure consistent scale.
Selected Features:
Age
Fare
Pclass (treated as ordinal)
✔ Standardization (Z-Score Normalization)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df[['Age', 'Fare', 'Pclass']] = scaler.fit_transform(df[['Age', 'Fare', 'Pclass']])
(Min-Max scaling option is also provided but commented out.)
🔹 Step 6: Outlier Detection (Visualization)
Outliers are visualized using boxplots for key numerical features.
import seaborn as sns
import matplotlib.pyplot as plt

sns.boxplot(x=df['Age'])
sns.boxplot(x=df['Fare'])
🔹 Step 7: Outlier Removal (IQR Method)
Outliers are removed using the Interquartile Range (IQR) method.
✔ IQR Formula
Lower Bound = Q1 − 1.5 × IQR
Upper Bound = Q3 + 1.5 × IQR
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    return df[(df[column] >= Q1 - 1.5*IQR) & (df[column] <= Q3 + 1.5*IQR)]
Applied sequentially to:
Age
Fare
📊 Final Output
Cleaned dataset with:
No missing values
Encoded categorical variables
Scaled numerical features
Outliers removed
Ready for:
Classification models
Survival prediction
Feature importance analysis
