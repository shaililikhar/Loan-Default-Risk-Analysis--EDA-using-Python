import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
application_data_path = 'application_data.csv'
previous_application_path = 'previous_application.csv'

# Read data with additional parameters to ensure complete load
application_data = pd.read_csv(application_data_path, low_memory=False)
previous_application = pd.read_csv(previous_application_path, low_memory=False)

# Factorize categorical columns in both datasets to convert them to numeric values
for column in application_data.select_dtypes(include=['object']).columns:
    application_data[column], _ = pd.factorize(application_data[column])

for column in previous_application.select_dtypes(include=['object']).columns:
    previous_application[column], _ = pd.factorize(previous_application[column])

print(f"\nOriginal shape of application data: {application_data.shape}")
print(f"\nOriginal shape of previous application data: {previous_application.shape}")


# Data Cleaning: Drop irrelevant columns
application_columns_to_drop = [
    'WEEKDAY_APPR_PROCESS_START', 'HOUR_APPR_PROCESS_START', 'APARTMENTS_AVG', 'BASEMENTAREA_AVG', 
    'YEARS_BEGINEXPLUATATION_AVG', 'YEARS_BUILD_AVG', 'COMMONAREA_AVG', 'ELEVATORS_AVG', 'ENTRANCES_AVG', 
    'FLOORSMAX_AVG', 'FLOORSMIN_AVG', 'LANDAREA_AVG', 'LIVINGAPARTMENTS_AVG', 'LIVINGAREA_AVG',
    'NONLIVINGAPARTMENTS_AVG', 'NONLIVINGAREA_AVG', 'APARTMENTS_MODE', 'BASEMENTAREA_MODE', 
    'YEARS_BEGINEXPLUATATION_MODE', 'YEARS_BUILD_MODE', 'COMMONAREA_MODE', 'ELEVATORS_MODE',
    'ENTRANCES_MODE', 'FLOORSMAX_MODE', 'FLOORSMIN_MODE', 'LANDAREA_MODE', 'LIVINGAPARTMENTS_MODE', 
    'LIVINGAREA_MODE', 'NONLIVINGAPARTMENTS_MODE', 'NONLIVINGAREA_MODE', 'APARTMENTS_MEDI', 
    'BASEMENTAREA_MEDI', 'YEARS_BEGINEXPLUATATION_MEDI', 'YEARS_BUILD_MEDI', 'COMMONAREA_MEDI',
    'ELEVATORS_MEDI', 'ENTRANCES_MEDI', 'FLOORSMAX_MEDI', 'FLOORSMIN_MEDI', 'LANDAREA_MEDI',
    'LIVINGAPARTMENTS_MEDI', 'LIVINGAREA_MEDI', 'NONLIVINGAPARTMENTS_MEDI', 'NONLIVINGAREA_MEDI',
    'FONDKAPREMONT_MODE', 'HOUSETYPE_MODE', 'TOTALAREA_MODE', 'WALLSMATERIAL_MODE',
    'EMERGENCYSTATE_MODE','FLAG_DOCUMENT_2','FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_6',
    'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9', 'FLAG_DOCUMENT_10', 
    'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14',
    'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18',
    'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21'
]

previous_application_columns_to_drop = [
    'WEEKDAY_APPR_PROCESS_START', 'HOUR_APPR_PROCESS_START','RATE_INTEREST_PRIMARY', 
    'RATE_INTEREST_PRIVILEGED', 'SELLERPLACE_AREA'
]
# Drop the irrelevant columns from each dataset and update the original datasets
application_data.drop(columns=application_columns_to_drop, errors='ignore', inplace=True)
previous_application.drop(columns=previous_application_columns_to_drop, errors='ignore', inplace=True)

# Merge the datasets on the common identifier 'SK_ID_CURR'
merged_data = pd.merge(application_data, previous_application, on='SK_ID_CURR', how='left')
print(f"\nPrevious shape of merged and cleaned data: {merged_data.shape}")

# Identify rows with more than 50% missing values before dropping
threshold = merged_data.shape[1] / 2
rows_with_missing_values = merged_data[merged_data.isnull().sum(axis=1) > threshold]

# Drop rows with more than 50% missing values
merged_data = merged_data.dropna(thresh=int(threshold), axis=0)

# Print dropped rows with more than 50% missing values
print(f"\nRows dropped:\n{rows_with_missing_values}")

# Handle missing values
# Fill missing values with mean for numerical columns and mode for categorical columns
for column in merged_data.columns:
    if merged_data[column].dtype == 'object':
        mode_value = merged_data[column].mode()[0]
        merged_data[column] = merged_data[column].fillna(mode_value)
    else:
        mean_value = merged_data[column].mean()
        merged_data[column] = merged_data[column].fillna(mean_value)

# Ensure consistent data types
# Convert numerical columns to appropriate types (float or int)
for column in merged_data.columns:
    if merged_data[column].dtype == 'object':
        try:
            merged_data[column] = pd.to_numeric(merged_data[column])
        except ValueError:
            pass  # Ignore columns that can't be converted

# Final shape after additional cleaning
print(f"\nFinal shape of merged and cleaned data: {merged_data.shape}")

# Save the cleaned data to a new CSV file
merged_data.to_csv('merged_cleaned_data.csv', index=False)

# Visual Analysis

# Calculate the correlation matrix with the correct column names
correlation_matrix = merged_data.corr()

# Extract correlations with the target variable
target_correlation = correlation_matrix['TARGET'].sort_values(ascending=False)
top_features = target_correlation.head(6).index.tolist()  # top 5 features + target

# Display top correlated features with target
print("\nTop 5 features correlated with loan default (TARGET):")
print(target_correlation.head(6))

# Plot the correlation matrix of top features
plt.figure(figsize=(10, 8))
sns.heatmap(merged_data[top_features].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.xticks(rotation=45) 
plt.yticks(rotation=0)
plt.title('Correlation Matrix of Top Significant Variables')
plt.show()

# Bar chart for top significant variables and target
plt.figure(figsize=(10, 6))
sns.barplot(x=target_correlation.head(6).index[1:], y=target_correlation.head(6).values[1:], palette="viridis")
plt.xticks(rotation=45)  
plt.title('Top Significant Variables Correlated with Loan Default')
plt.ylabel('Correlation with TARGET')
plt.xlabel('Features')
plt.show()

# Plot individual plots to show relationship of top variable and target 
# Box plots for continuous features
continuous_features = ['DAYS_BIRTH', 'DAYS_LAST_PHONE_CHANGE', 'DAYS_ID_PUBLISH']

for feature in continuous_features:
    plt.figure(figsize=(8, 5))
    sns.boxplot(x=merged_data['TARGET'], y=merged_data[feature], palette="viridis")
    plt.title(f'Relationship between {feature} and Loan Default (TARGET)')
    plt.ylabel(feature)
    plt.xlabel('TARGET')
    plt.show()

# Bar charts for categorical features
categorical_features = ['REGION_RATING_CLIENT_W_CITY', 'REGION_RATING_CLIENT']

for feature in categorical_features:
    plt.figure(figsize=(8, 5))
    sns.barplot(x=merged_data[feature], y=merged_data['TARGET'], palette="viridis")
    plt.title(f'Relationship between {feature} and Loan Default (TARGET)')
    plt.ylabel('Average TARGET Value')
    plt.xlabel(feature)
    plt.show()