import pandas as pd
import numpy as np
import seaborn as sns
import sklearn
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Loading the dataset
df = pd.read_csv('Data/diabetes.csv')
df.head()

# Renaming a column
df.rename(columns={'DiabetesPedigreeFunction': 'DPF'}, inplace=True)

# Noticed a minimum value of 0 for the following columns which is not possible, hence reassigning them as missing values.
df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = df[[
    'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.nan)

# Filling missing values
df['Glucose'].fillna(df['Glucose'].mean(), inplace=True)
df['BloodPressure'].fillna(df['BloodPressure'].mean(), inplace=True)
df['SkinThickness'].fillna(df['SkinThickness'].mean(), inplace=True)
df['Insulin'].fillna(df['Insulin'].mean(), inplace=True)
df['BMI'].fillna(df['BMI'].median(), inplace=True)

# Splitting the dataset
x = np.array(df.drop(['Outcome'], 1))
y = np.array(df['Outcome'])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
    x, y, test_size=0.20, random_state=0)


# Logistic Regression model
logreg = LogisticRegression()
logreg.fit(x_train, y_train)
acc = logreg.score(x_test, y_test)
print(acc)


# KNN Classification

# Scaling variables
scaler = StandardScaler()
scaler.fit(x_train)
scaler.fit(x_test)

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# Training the model
model = KNeighborsClassifier(n_neighbors=11)
model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
print(acc)
y_pred = model.predict(x_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
