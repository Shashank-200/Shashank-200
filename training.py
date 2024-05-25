import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load the datasets
train_df = pd.read_excel(r'D:\py-tasks\data\train.xlsx')
test_df = pd.read_excel(r'D:\py-tasks\data\test.xlsx')

# Assume 'target' is the column name for the target variable in train_df
X_train = train_df.drop(columns=['target'])
y_train = train_df['target']
X_test = test_df

# Handle missing values
X_train.fillna(X_train.mean(), inplace=True)
X_test.fillna(X_test.mean(), inplace=True)

# Encode categorical features
for column in X_train.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X_train[column] = le.fit_transform(X_train[column])
    X_test[column] = le.transform(X_test[column])

# Scale numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression
logreg = LogisticRegression(random_state=42)
logreg.fit(X_train_scaled, y_train)
logreg_train_accuracy = accuracy_score(y_train, logreg.predict(X_train_scaled))
logreg_predictions = logreg.predict(X_test_scaled)
pd.DataFrame({'Id': test_df.index, 'Logistic_Regression_Predictions': logreg_predictions}).to_csv('logreg_predictions.csv', index=False)

# Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_scaled, y_train)
rf_train_accuracy = accuracy_score(y_train, rf.predict(X_train_scaled))
rf_predictions = rf.predict(X_test_scaled)
pd.DataFrame({'Id': test_df.index, 'Random_Forest_Predictions': rf_predictions}).to_csv('rf_predictions.csv', index=False)

# SVM
svm = SVC(random_state=42)
svm.fit(X_train_scaled, y_train)
svm_train_accuracy = accuracy_score(y_train, svm.predict(X_train_scaled))
svm_predictions = svm.predict(X_test_scaled)
pd.DataFrame({'Id': test_df.index, 'SVM_Predictions': svm_predictions}).to_csv('svm_predictions.csv', index=False)

# Summary of training accuracies
print(f"Logistic Regression Training Accuracy: {logreg_train_accuracy}")
print(f"Random Forest Training Accuracy: {rf_train_accuracy}")
print(f"SVM Training Accuracy: {svm_train_accuracy}")
