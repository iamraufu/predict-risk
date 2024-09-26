import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

file_path = './dataset.csv'
dataset = pd.read_csv(file_path)


cleaned_dataset = dataset.dropna()

mean_charges = cleaned_dataset['charges'].mean()
cleaned_dataset['charges_category'] = cleaned_dataset['charges'].apply(lambda x: 'Risky' if x > mean_charges else 'Not Risky')


cleaned_dataset['sex'] = cleaned_dataset['sex'].astype('category')
cleaned_dataset['smoker'] = cleaned_dataset['smoker'].astype('category')
cleaned_dataset['region'] = cleaned_dataset['region'].astype('category')
cleaned_dataset['charges_category'] = cleaned_dataset['charges_category'].astype('category')

cleaned_dataset = cleaned_dataset.drop(columns=['charges'])

X = cleaned_dataset.drop(columns=['charges_category'])
y = cleaned_dataset['charges_category']


X = pd.get_dummies(X, drop_first=True)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)


y_pred = log_reg.predict(X_test)
print(f"Logistic Regression Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Classification Report:\n{classification_report(y_test, y_pred)}")

conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Risky', 'Risky'], yticklabels=['Not Risky', 'Risky'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Logistic Regression')
plt.show()

rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train, y_train)


y_pred_rf = rf_classifier.predict(X_test)
print(f"Random Forest Accuracy: {accuracy_score(y_test, y_pred_rf)}")
print(f"Classification Report:\n{classification_report(y_test, y_pred_rf)}")


conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(conf_matrix_rf, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Risky', 'Risky'], yticklabels=['Not Risky', 'Risky'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Random Forest')
plt.show()


# Save the trained model
joblib.dump(rf_classifier, 'rf_classifier_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(list(X.columns), 'column_names.pkl')

# new_data = {
#     'age': [25, 40, 55],
#     'sex': ['female', 'male', 'female'],
#     'bmi': [22.5, 30.0, 27.0],
#     'children': [1, 2, 3],
#     'smoker': ['no', 'yes', 'no'],
#     'region': ['southwest', 'southeast', 'northwest']
# }
#
# test_df = pd.DataFrame(new_data)
#
#
# test_df = pd.get_dummies(test_df, drop_first=True)
#
# missing_cols = set(X.columns) - set(test_df.columns)
# for col in missing_cols:
#     test_df[col] = 0
#
#
# test_df = test_df[X.columns]
#
# test_df = scaler.transform(test_df)
#
#
# predictions = rf_classifier.predict(test_df)
# print("Predictions:", predictions)