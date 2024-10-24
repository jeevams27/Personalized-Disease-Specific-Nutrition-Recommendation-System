# Personalized-Disease-Specific-Nutrition-Recommendation-System

The Personalized Disease-Specific Nutrition Recommendation System aims to develop tailored nutrition plans for individuals with chronic diseases, including diabetes, chronic kidney disease, cardiovascular disease, liver disease, and asthma. Utilizing machine learning algorithms, the system analyzes key features such as age, weight, height, and food preferences to create effective dietary strategies that minimize disease vulnerability.

## Features

1. **Tailored Nutrition Plans:** Provides personalized dietary recommendations based on individual health profiles and chronic disease conditions.
2. **Machine Learning Algorithms:** Utilizes advanced machine learning techniques to analyze data and generate effective nutrition strategies.
3. **User-Friendly Interface:** Ensures easy navigation and interaction for users seeking dietary guidance.
4. **Data Visualization:** Includes visual aids to help users understand their nutritional needs and dietary modifications.

## Requirements
1. Python 3.x: For the development of the project.
2. Essential Python Packages: Includes libraries such as pandas, numpy, scikit-learn for data handling and machine learning functionalities.

## Flow chart:
![image](https://github.com/user-attachments/assets/a84f8890-59f3-463d-b493-f0182dad097b)


## Installation:
1. Clone the repository :
```
git clone https://github.com/Paul-Andrew-15/Personalized-Disease-Specific-Nutrition-Recommendation-System.git
```
2. Install the required packages.
3. Download the pre-trained Nutrition recommendation model and label mappings.

## Usage:
1. Open a new Google Colab notebook.
2. Upload the project files in Google Drive.
3. Load the pre-trained Nutrition Recommendation model and label mappings. Ensure the model files are correctly placed in the Colab working directory.
4. Execute the Nutrition Recommendation script in the Colab notebook, which may involve adapting the script to run within a notebook environment.
5. Follow the on-screen instructions or customize input cells in the notebook for Nutrition Recommendations with uploaded patient data.
6. View and analyze the results directly within the Colab notebook.
7. Repeat the process for additional Diseases as needed.

## Program:
```python
# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, make_scorer
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt
import joblib
import warnings

warnings.filterwarnings("ignore")

# Loading dataset
dataset_path = 'Chronic_disease_less.csv'
df = pd.read_csv(dataset_path)
df.head(10)

# Fix the column names
df.rename(columns={'Blood Glucose(mg/dL)': 'Blood_Glucose',
                   'Cholesterol Levels(mg/dL)': 'Cholesterol_Levels',
                   'Blood Pressure(mmHg)': 'Blood_Pressure',
                   'Kidney Function (mg/dL)': 'Kidney_Function'}, inplace=True)
df.columns

# Exploratory Data Analysis
def numeric_univariate_analysis(num_data, col_name):
    print('*' * 5, col_name, '*' * 5)
    print(num_data.agg(['min', 'max', 'mean', 'median', 'std', 'skew', 'kurt']))
    print()

numeric_univariate_analysis(df['Age'], 'Age')

df['BMI'].plot(kind='kde')
plt.figure(figsize=(10, 6))
plt.title('Class Distribution in Training Data')
plt.xlabel('Class Labels')
plt.ylabel('Frequency')
df['Meal'].value_counts().plot(kind='bar', color='green')
plt.show()

df.boxplot(column='Age', by='Food Preference', grid=False)
plt.title('Age Distribution by Food Preference')
plt.xlabel('Food Preference')
plt.ylabel('Age')
plt.show()

tab = pd.crosstab(df['Gender'], df['Disease Name'], normalize='index')
tab.plot(kind='bar')
plt.show()

# Data Preprocessing
df.isnull().sum()
df['Kidney_Function'] = df['Kidney_Function'].fillna(-1)
df['Blood_Glucose'] = df['Blood_Glucose'].fillna(-1)

le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])
df['Smoking'] = le.fit_transform(df['Smoking'])
df['Alcohol Consumption'] = le.fit_transform(df['Alcohol Consumption'])
df['Food Preference'] = le.fit_transform(df['Food Preference'])
df['Nutritional Requirement'] = le.fit_transform(df['Nutritional Requirement'])
df['Blood_Pressure_Category'] = le.fit_transform(df['Blood_Pressure_Category'])
df['Hydration Level'] = le.fit_transform(df['Hydration Level'])
df['Meal'] = le.fit_transform(df['Meal'])
df['Disease Name'] = le.fit_transform(df['Disease Name'])
df['Food Allergies'] = le.fit_transform(df['Food Allergies'])
df['Symptoms'] = le.fit_transform(df['Symptoms'])

# Feature selection
X = df.drop(['Meal', 'Patient ID', 'Symptoms', 'Nutritional Requirement', 'Gender'], axis=1)
y = df['Meal']

# Model selection
def get_params(model, param_dist, loss_function, X_train, y_train, n_iter=100, random_state=42):
    warnings.filterwarnings("ignore")
    scorer = make_scorer(loss_function, greater_is_better=True)
    print(f'Getting {model.__class__.__name__} params using RandomizedSearchCV:\n')
    random_search = RandomizedSearchCV(model, param_distributions=param_dist, scoring=scorer, cv=5, n_iter=n_iter, random_state=random_state)
    random_search.fit(X_train, y_train)
    best_model = random_search.best_estimator_
    best_params = best_model.get_params()
    param_string = " | ".join([f"{key}: {value}" for key, value in best_params.items() if value is not None])
    print(param_string)
    print('\n')
    return best_params

rf_param_dist = {
    'n_estimators': np.arange(50, 200, 10),
    'max_depth': [None] + list(np.arange(5, 20, 5)),
    'min_samples_split': np.arange(2, 11),
    'min_samples_leaf': np.arange(1, 11),
    'bootstrap': [True, False]
}

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_accuracy_params = get_params(RandomForestClassifier(), rf_param_dist, accuracy_score, X_train, y_train, n_iter=100, random_state=42)

model = RandomForestClassifier(**rf_accuracy_params)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, zero_division=0)

print("Confusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)

# Save the model
joblib.dump(model, 'nutrition_recommendation_model.pkl')

```

## Output:
Nutrition Recommendation System
![image](https://github.com/user-attachments/assets/37cd6b0a-da60-4a25-9a86-1f4c11e9477f)

![image](https://github.com/user-attachments/assets/000eb12b-9fef-4b93-b192-0e2070c1333c)


## Result:
The nutrition recommendation model, built using the Random Forest algorithm, demonstrates strong performance on both training and testing set:
1. The model achieved an accuracy of 92% showcasing its ability to provide accurate dietary recommendations.
2. Precision, reflecting the model's ability to predict specific meal types correctly, is 82%, indicating good identification of relevant nutrition plans.
3. Recall stands at 80%, demonstrating the model's effectiveness in capturing relevant dietary suggestions.

These results indicate the model's strong accuracy, precision, and recall, making it a reliable tool for personalized nutrition recommendations.
