import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from faker import Faker
import random
import datetime

# Initialize Faker for data generation
fake = Faker()
num_records = 30

# Generate synthetic e-commerce data
url = 'https://raw.githubusercontent.com/Viny2030/algorithms_fraud_corruption/main/df_ecommerce.csv'
df_ecommerce = pd.read_csv(url)

df_ecommerce['Date_Time'] = pd.to_datetime(df_ecommerce['Date_Time']) # Convert the column to datetime

# Mark nighttime transactions as fraudulent (example rule)
# Ensure 'Is_Fraudulent' column exists before assigning values
if 'Is_Fraudulent' not in df_ecommerce.columns:
    df_ecommerce['Is_Fraudulent'] = 0 # Initialize with 0 (not fraudulent)

for index, row in df_ecommerce.iterrows():
    if row['Date_Time'].hour >= 21:
        df_ecommerce.at[index, 'Is_Fraudulent'] = 1

# 4.1 Online Fraudulent Transaction Identification
print("\n---")
print("## 4.1 Online Fraudulent Transaction Identification (Logistic Regression Example)")
print("---")
if 'Amount' in df_ecommerce.columns and 'Date_Time' in df_ecommerce.columns and 'Is_Fraudulent' in df_ecommerce.columns:
    # Extract temporal features
    df_ecommerce['Hour'] = df_ecommerce['Date_Time'].dt.hour
    df_ecommerce['Day_of_Week'] = df_ecommerce['Date_Time'].dt.dayofweek

    # Encode categorical variables
    # Only encode 'Product' and 'IP_Address' for features, 'User_ID' is used for behavioral analysis later
    df_encoded_transactions = pd.get_dummies(df_ecommerce, columns=['Product'], prefix='Prod', dummy_na=False)
    df_encoded_transactions = pd.get_dummies(df_encoded_transactions, columns=['IP_Address'], prefix='IP', dummy_na=False, prefix_sep='_')

    # Select features for the model
    # Filter out columns that might not exist after one-hot encoding if a category is missing
    features_transactions = ['Amount', 'Hour', 'Day_of_Week'] + \
                            [col for col in df_encoded_transactions.columns if col.startswith('Prod_')] + \
                            [col for col in df_encoded_transactions.columns if col.startswith('IP_')]

    # Ensure all selected features are actually in the DataFrame
    features_transactions = [col for col in features_transactions if col in df_encoded_transactions.columns]

    if 'Is_Fraudulent' in df_encoded_transactions.columns and all(feature in df_encoded_transactions.columns for feature in features_transactions):
        X_trans = df_encoded_transactions[features_transactions]
        y_trans = df_encoded_transactions['Is_Fraudulent']

        # Handle cases where there's only one class in y_trans
        if len(np.unique(y_trans)) < 2:
            print("\nCannot perform logistic regression: 'Is_Fraudulent' column has only one unique class.")
        else:
            X_train_trans, X_test_trans, y_train_trans, y_test_trans = train_test_split(X_trans, y_trans, test_size=0.3, random_state=42, stratify=y_trans)

            scaler_trans = StandardScaler()
            X_train_scaled_trans = scaler_trans.fit_transform(X_train_trans)
            X_test_scaled_trans = scaler_trans.transform(X_test_trans)

            model_ecommerce = LogisticRegression(random_state=42, solver='liblinear') # Use liblinear for smaller datasets
            model_ecommerce.fit(X_train_scaled_trans, y_train_trans)
            y_pred_ecommerce = model_ecommerce.predict(X_test_scaled_trans)

            print("\nLogistic Regression Predictions:", y_pred_ecommerce)
            print("Actual Values:", y_test_trans.values)
            print("Model Accuracy:", accuracy_score(y_test_trans, y_pred_ecommerce))
            print("\nClassification Report:\n", classification_report(y_test_trans, y_pred_ecommerce, target_names=list(map(str, np.unique(y_trans))), zero_division=0))

            # Plotting actual vs predicted for Logistic Regression
            plt.figure(figsize=(8, 5))
            sns.scatterplot(x=range(len(y_test_trans)), y=y_test_trans, label='Actual', marker='o', s=100)
            sns.scatterplot(x=range(len(y_pred_ecommerce)), y=y_pred_ecommerce, label='Predicted', marker='x', s=100)
            plt.title('Logistic Regression: Actual vs. Predicted Fraudulent Transactions')
            plt.xlabel('Transaction Index')
            plt.ylabel('Is Fraudulent (0=No, 1=Yes)')
            plt.yticks([0, 1])
            plt.legend()
            plt.grid(True)
            plt.show()

    else:
        print("\nCannot perform analysis for 4.1 due to missing necessary columns or features after encoding.")
else:
    print("\nCannot perform analysis for 4.1 due to missing necessary initial columns.")

# 4.2 Detection of Fake Accounts and Malicious Activities
print("\n---")
print("## 4.2 Detection of Fake Accounts and Malicious Activities (DBSCAN on IPs)")
print("---")
if 'IP_Address' in df_ecommerce.columns:
    le_ip = LabelEncoder()
    df_ecommerce['IP_Encoded'] = le_ip.fit_transform(df_ecommerce['IP_Address'])
    ip_array = df_ecommerce[['IP_Encoded']].values
    scaler_ip = StandardScaler()
    ip_scaled = scaler_ip.fit_transform(ip_array)

    # Adjust eps based on data scale to get meaningful clusters
    # A smaller eps might result in more noise, larger eps in fewer, larger clusters
    # For scaled data, 0.5 is a common starting point, but it's often tuned.
    dbscan_ip = DBSCAN(eps=0.5, min_samples=2)
    df_ecommerce['IP_Group'] = dbscan_ip.fit_predict(ip_scaled)

    print("\nIP Clustering (DBSCAN):")
    print(df_ecommerce[['User_ID', 'IP_Address', 'IP_Group']])
    print("\nIP Groups:", df_ecommerce['IP_Group'].unique())

    # Count fraudulent transactions per IP group, excluding noise (-1)
    fraudulent_per_group = df_ecommerce[df_ecommerce['IP_Group'] != -1].groupby('IP_Group')['Is_Fraudulent'].sum()
    print("\nNumber of Fraudulent Transactions per IP Group (excluding noise):")
    print(fraudulent_per_group)

    # Plotting DBSCAN results
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=df_ecommerce.index, y=df_ecommerce['IP_Encoded'], hue=df_ecommerce['IP_Group'], palette='viridis', legend='full', s=100)
    plt.title('DBSCAN Clustering of IP Addresses')
    plt.xlabel('Transaction Index')
    plt.ylabel('Encoded IP Address')
    plt.grid(True)
    plt.show()

else:
    print("\nCannot perform analysis for 4.2 because the 'IP_Address' column is missing.")

# 4.3 User Behavior Analysis for Fraud Detection (Conceptual Example)
print("\n---")
print("## 4.3 User Behavior Analysis for Fraud Detection (Conceptual Example)")
print("---")
if 'User_ID' in df_ecommerce.columns and 'Date_Time' in df_ecommerce.columns:
    user_frequency = df_ecommerce.groupby('User_ID')['Date_Time'].count().reset_index(name='Num_Transactions')
    print("\nTransaction Frequency per User:")
    print(user_frequency)

    frequency_threshold = 3
    high_activity_users = user_frequency[user_frequency['Num_Transactions'] >= frequency_threshold]['User_ID'].tolist()
    if high_activity_users:
        print(f"\nHigh Activity Users ({frequency_threshold} or more transactions): {high_activity_users}")
        high_activity_transactions = df_ecommerce[df_ecommerce['User_ID'].isin(high_activity_users)]
        print("\nTransactions of High Activity Users:")
        print(high_activity_transactions[['User_ID', 'Date_Time', 'Amount', 'Is_Fraudulent']])

        # Plotting transaction frequency per user
        plt.figure(figsize=(12, 6))
        sns.barplot(x='User_ID', y='Num_Transactions', hue='Num_Transactions', data=user_frequency, palette='coolwarm', dodge=False, legend=False)
        plt.title('Transaction Frequency per User')
        plt.xlabel('User ID')
        plt.ylabel('Number of Transactions')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
    else:
        print("\nNo users with high transaction frequency were found in this example.")
else:
    print("\nCannot perform analysis for 4.3 due to missing necessary columns.")
