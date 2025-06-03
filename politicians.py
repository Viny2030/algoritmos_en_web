import random
from faker import Faker
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Set a random seed for reproducibility
np.random.seed(42)
random.seed(42)

# --- 1. Load the dataset for political corruption ---
# Using the raw URL for the CSV file on GitHub
url_politicians = 'https://raw.githubusercontent.com/Viny2030/algorithms_fraud_corruption/main/df_politicians.csv'

try:
    df_politicians = pd.read_csv(url_politicians)
    print("Successfully loaded df_politicians.csv from GitHub.")
    print("First 5 rows of the loaded dataset:")
    print(df_politicians.head())
except Exception as e:
    print(f"Error loading df_politicians.csv: {e}")
    print("Please ensure the URL is correct and the file is accessible.")
    # Exit or handle the error if the dataset cannot be loaded
    exit() # Or create a dummy DataFrame to allow the script to continue if desired

# --- Data Cleaning and Type Conversion (based on loaded data) ---
# Ensure 'Last_Year_Asset_Declaration' is numeric
df_politicians['Last_Year_Asset_Declaration'] = pd.to_numeric(df_politicians['Last_Year_Asset_Declaration'], errors='coerce')
# Fill any NaNs that might result from 'coerce' (if non-numeric values were present)
df_politicians['Last_Year_Asset_Declaration'].fillna(0, inplace=True)

# Ensure 'Last_Year_Asset_Increase_Percentage' is numeric
df_politicians['Last_Year_Asset_Increase_Percentage'] = pd.to_numeric(df_politicians['Last_Year_Asset_Increase_Percentage'], errors='coerce')
df_politicians['Last_Year_Asset_Increase_Percentage'].fillna(0, inplace=True) # Fill NaNs

# Ensure 'Donations_Received' and 'Campaign_Expenses' are numeric
df_politicians['Donations_Received'] = pd.to_numeric(df_politicians['Donations_Received'], errors='coerce').fillna(0)
df_politicians['Campaign_Expenses'] = pd.to_numeric(df_politicians['Campaign_Expenses'], errors='coerce').fillna(0)
df_politicians['Previous_Complaints'] = pd.to_numeric(df_politicians['Previous_Complaints'], errors='coerce').fillna(0)

# Check if 'Is_Suspicious' column exists, if not, create it and mark some as suspicious (as your original script did)
if 'Is_Suspicious' not in df_politicians.columns:
    print("Column 'Is_Suspicious' not found. Initializing and simulating fraud.")
    df_politicians['Is_Suspicious'] = 0 # Initialize

    # Simulate additional Suspicious Cases if the column was missing
    num_politicians = len(df_politicians)
    num_suspicious_to_add = int(num_politicians * 0.15) # Target 15% of total politicians to be suspicious

    non_suspicious_indices = df_politicians[df_politicians['Is_Suspicious'] == 0].index
    num_suspicious_to_add = min(num_suspicious_to_add, len(non_suspicious_indices))

    suspicious_indices = np.random.choice(non_suspicious_indices, num_suspicious_to_add, replace=False)
    df_politicians.loc[suspicious_indices, 'Is_Suspicious'] = 1

    # Apply modifications to simulate common corruption patterns for the newly marked suspicious cases
    for idx in suspicious_indices:
        if random.random() < 0.4: # 40% of cases: significant asset increase
            df_politicians.loc[idx, 'Last_Year_Asset_Increase_Percentage'] = np.random.uniform(0.20, 0.60)
            df_politicians.loc[idx, 'Last_Year_Asset_Declaration'] = df_politicians.loc[idx, 'Last_Year_Asset_Declaration'] * np.random.uniform(1.2, 1.5)

        if random.random() < 0.3: # 30% of cases: high donations and relatively low campaign expenses
            df_politicians.loc[idx, 'Donations_Received'] = np.random.uniform(100000, 300000)
            df_politicians.loc[idx, 'Campaign_Expenses'] = np.random.uniform(5000, 50000)

        if random.random() < 0.3: # 30% of cases: Undisclosed business ties
            df_politicians.loc[idx, 'Business_Ties'] = 'Undisclosed'

        if random.random() < 0.2: # 20% of cases: Multiple previous complaints
            df_politicians.loc[idx, 'Previous_Complaints'] = random.randint(2, 5)

        if random.random() < 0.1: # 10% of cases: High-level position and extraordinary asset increase
            if df_politicians.loc[idx, 'Position'] in ['Minister', 'Senator', 'Mayor']:
                df_politicians.loc[idx, 'Last_Year_Asset_Increase_Percentage'] *= random.uniform(2.5, 4.0)
else:
    print("Column 'Is_Suspicious' already exists. Using existing labels.")


# 3. Feature Engineering
# a) Ratio of Asset Increase Percentage to Total Assets (e.g., disproportionate increase relative to total wealth)
df_politicians['Asset_Increase_Ratio'] = df_politicians['Last_Year_Asset_Increase_Percentage'] / (df_politicians['Last_Year_Asset_Declaration'] + 1e-6)

# b) Ratio of Campaign Expenses to Donations (e.g., very low expenses despite high donations could be suspicious)
df_politicians['Expenses_Donations_Ratio'] = df_politicians['Campaign_Expenses'] / (df_politicians['Donations_Received'] + 1e-6)

# c) Binary flag: Is there a High Asset Increase (based on a threshold)?
high_increase_amount_threshold = 150000 # Define a threshold for what constitutes a "high" amount increase
df_politicians['Is_High_Asset_Increase_Amount'] = (df_politicians['Last_Year_Asset_Increase_Percentage'] * df_politicians['Last_Year_Asset_Declaration'] > high_increase_amount_threshold).astype(int)

# d) Binary flag: Are there Many Previous Complaints (based on a threshold)?
many_complaints_threshold = 1 # More than 1 complaint is considered "many"
df_politicians['Has_Many_Complaints'] = (df_politicians['Previous_Complaints'] > many_complaints_threshold).astype(int)

# e) Encode categorical variables using LabelEncoder
# 'Position'
le_position = LabelEncoder()
df_politicians['Position_Encoded'] = le_position.fit_transform(df_politicians['Position'])

# 'Political_Party'
le_party = LabelEncoder()
df_politicians['Political_Party_Encoded'] = le_party.fit_transform(df_politicians['Political_Party'])

# 'Business_Ties' - ensure to handle potential new values or existing ones correctly
# First, ensure all unique values are present in categories. If 'Undisclosed' might not be in original, add it.
if 'Undisclosed' not in df_politicians['Business_Ties'].unique():
    df_politicians['Business_Ties'] = df_politicians['Business_Ties'].replace('Your_Simulated_Undisclosed_Value', 'Undisclosed') # If you had a placeholder
df_politicians['Business_Ties'].fillna('No', inplace=True) # Fill any NaNs in Business_Ties with 'No' or a suitable default
le_business_ties = LabelEncoder()
df_politicians['Business_Ties_Encoded'] = le_business_ties.fit_transform(df_politicians['Business_Ties'])


# 4. Feature Selection and Data Preparation
# Define the features (independent variables, X) that the model will use for prediction
features = ['Last_Year_Asset_Declaration', 'Last_Year_Asset_Increase_Percentage',
            'Donations_Received', 'Campaign_Expenses', 'Asset_Increase_Ratio',
            'Expenses_Donations_Ratio', 'Is_High_Asset_Increase_Amount', 'Has_Many_Complaints',
            'Position_Encoded', 'Political_Party_Encoded', 'Business_Ties_Encoded']

# Check if all features exist in the DataFrame
missing_features = [f for f in features if f not in df_politicians.columns]
if missing_features:
    print(f"Warning: The following features are missing from the DataFrame and will be excluded: {missing_features}")
    features = [f for f in features if f not in missing_features] # Update features list

X = df_politicians[features]
# Define the target variable (dependent variable, y) which is 'Is_Suspicious'
y = df_politicians['Is_Suspicious']

# Handle any potential missing values by filling them with 0 (or a suitable strategy)
X = X.fillna(0)

# Check for single class in target variable
if len(y.unique()) < 2:
    print("\nError: The 'Is_Suspicious' column contains only one unique class. Cannot perform classification.")
    print("Please ensure your dataset or simulation generates both suspicious and non-suspicious cases.")
    exit() # Exit if classification is not possible

# 5. Split Data into Training and Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# 6. Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 7. Train the Classification Model (Random Forest)
print("\n7. Training the Model for Political Corruption Detection (Random Forest):")
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# 8. Evaluate the Model
print("\n8. Model Evaluation:")
print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=['Not Suspicious', 'Suspicious'], zero_division=0))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 9. Analyze Politicians Detected as Suspicious
df_test_results = df_politicians.loc[X_test.index].copy()
df_test_results['Prediction_Suspicious'] = y_pred

suspicious_politicians = df_test_results[df_test_results['Prediction_Suspicious'] == 1][
    ['Politician_ID', 'Full_Name', 'Position', 'Political_Party', 'Last_Year_Asset_Declaration',
     'Last_Year_Asset_Increase_Percentage', 'Donations_Received', 'Campaign_Expenses',
     'Business_Ties', 'Previous_Complaints', 'Is_Suspicious', 'Prediction_Suspicious']
]
print("\n9. Politicians Detected as Potentially Suspicious by the System:")
if not suspicious_politicians.empty:
    print(suspicious_politicians)
else:
    print("No politicians were detected as suspicious by the model in the test set.")


# 10. Feature Importance Analysis
print("\n10. Feature Importance (from RandomForestClassifier):")
if hasattr(model, 'feature_importances_'):
    feature_importances = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_})
    feature_importances = feature_importances.sort_values(by='Importance', ascending=False)
    print(feature_importances)

    # --- Plotting Feature Importance ---
    plt.figure(figsize=(10, 7))
    sns.barplot(x='Importance', y='Feature', data=feature_importances, palette='viridis')
    plt.title('Feature Importance for Political Corruption Detection', fontsize=16)
    plt.xlabel('Importance Score', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# 11. Visualizations for Data Exploration and Model Insights

# --- Plot 1: Distribution of 'Is_Suspicious' (Target Variable) ---
plt.figure(figsize=(7, 6))
sns.countplot(x='Is_Suspicious', data=df_politicians, palette='cividis')
plt.title('Distribution of Suspicious vs. Non-Suspicious Cases', fontsize=16)
plt.xlabel('Suspicious Status (0: Not Suspicious, 1: Suspicious)', fontsize=12)
plt.ylabel('Number of Politicians', fontsize=12)
plt.xticks([0, 1], ['Not Suspicious', 'Suspicious'], fontsize=10)
plt.yticks(fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# --- Plot 2: Distribution of key numerical features by 'Is_Suspicious' ---
numerical_features_to_plot = [
    'Last_Year_Asset_Increase_Percentage',
    'Donations_Received',
    'Campaign_Expenses',
    'Previous_Complaints',
    'Asset_Increase_Ratio',
    'Expenses_Donations_Ratio'
]

for feature in numerical_features_to_plot:
    if feature in df_politicians.columns: # Ensure feature exists before plotting
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df_politicians, x=feature, hue='Is_Suspicious', kde=True,
                     palette={0: 'skyblue', 1: 'salmon'},
                     stat='density', common_norm=False, bins=20)
        plt.title(f'Distribution of {feature} by Suspicious Status', fontsize=16)
        plt.xlabel(feature, fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.legend(title='Is Suspicious', labels=['Not Suspicious', 'Suspicious'])
        plt.tight_layout()
        plt.show()
    else:
        print(f"Skipping plot for missing feature: {feature}")

# --- Plot 3: Confusion Matrix Heatmap ---
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, linewidths=.5, linecolor='black',
            xticklabels=['Predicted Not Suspicious', 'Predicted Suspicious'],
            yticklabels=['True Not Suspicious', 'True Suspicious'])
plt.title('Confusion Matrix of Political Corruption Detection', fontsize=16)
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10, rotation=0)
plt.tight_layout()
plt.show()

# Save the final DataFrame to a CSV file (optional)
csv_file_name = 'df_politicians_processed.csv'
df_politicians.to_csv(csv_file_name, index=False)
print(f"\nFinal processed DataFrame saved to '{csv_file_name}'")

# Display the first few rows of the processed DataFrame to check features
print("\nFirst 5 rows of the processed DataFrame:")
print(df_politicians.head())
