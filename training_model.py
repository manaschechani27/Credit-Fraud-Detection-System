import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd

# Load the processed dataset
data = pd.read_csv(r'C:\Users\91952\Desktop\Credit card fraud detection\Data\credit_card_data_processed.csv')

# Separate features and target
X = data.drop(columns=['Class'])
y = data['Class']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Initialize and train the Random Forest model
rf_model = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Perform GridSearchCV with 5-fold cross-validation
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Print the best parameters found by GridSearchCV
print("Best Parameters:", grid_search.best_params_)

# Retrieve the best model from GridSearchCV
best_rf_model = grid_search.best_estimator_

# Predictions
y_pred_train_rf = best_rf_model.predict(X_train)
y_pred_test_rf = best_rf_model.predict(X_test)

# Evaluation
print("Classification Report:\n", classification_report(y_test, y_pred_test_rf))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_test_rf))
print(f"Random Forest Model - Training Accuracy: {accuracy_score(y_train, y_pred_train_rf):.4f}")
print(f"Random Forest Model - Test Accuracy: {accuracy_score(y_test, y_pred_test_rf):.4f}")

# Serialize and save the best model
model_filename = r'C:\Users\91952\Desktop\Credit card fraud detection\Data\random_forest_model.pkl'
joblib.dump(best_rf_model, model_filename)
print(f"Saved the trained model at: {model_filename}")
