import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

def load_data(filepath):
    return pd.read_csv(filepath)

def preprocess_data(data):
    # Handling missing values
    data = data.dropna(subset=['Class'])
    data['Amount'] = data['Amount'].fillna(data['Amount'].median())

    # Separating the features and the target
    X = data.drop(columns='Class', axis=1)
    y = data['Class']

    # Feature scaling for 'Amount' and 'Time'
    scaler = StandardScaler()
    X[['Amount', 'Time']] = scaler.fit_transform(X[['Amount', 'Time']])

    return X, y

def balance_data(X, y, undersample_ratio=0.5):
    # Undersample the majority class to a higher threshold to retain more data
    rus = RandomUnderSampler(sampling_strategy=undersample_ratio, random_state=42)
    X_undersampled, y_undersampled = rus.fit_resample(X, y)

    # Use SMOTE to further balance the dataset
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X_undersampled, y_undersampled)

    return X_balanced, y_balanced

def save_data(X, y, filepath):
    balanced_data = pd.DataFrame(X, columns=X.columns)
    balanced_data['Class'] = y
    balanced_data.to_csv(filepath, index=False)

if __name__ == "__main__":
    # Example usage
    raw_data_path = r"C:\Users\91952\Desktop\Credit card fraud detection\Data\creditcard.csv"
    processed_data_path = r"C:\Users\91952\Desktop\Credit card fraud detection\Data\credit_card_data_processed.csv"

    data = load_data(raw_data_path)
    X, y = preprocess_data(data)
    X_balanced, y_balanced = balance_data(X, y)
    save_data(X_balanced, y_balanced, processed_data_path)
