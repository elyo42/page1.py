import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support,matthews_corrcoef

pd.options.display.max_columns = None
pd.options.display.max_rows = None

df = pd.read_csv('dataset_diabetes/diabetic_data.csv',sep=',', header=0)

# print(df.head().to_string())
df = df.replace('?',np.nan)
# print(df.head().to_string())
# print(df.dtypes)
# print(df.describe())
age_mapping = {
    '[0-10)': 1,
    '[10-20)': 2,
    '[20-30)': 3,
    '[30-40)': 4,
    '[40-50)': 5,
    '[50-60)': 6,
    '[60-70)': 7,
    '[70-80)': 8,
    '[80-90)': 9,
    '[90-100)': 10
}
df['race'] = df['race'].fillna('Other')
race_mapping = {'Caucasian': 1,
                'AfricanAmerican': 2,
                'Asian': 3,
                'Hispanic': 4,
                'Other': 5
                }
readmitted_mapping = {'>30': 1,
                      '<30': 2,
                      'NO': 3}
medication_mapping = {'No': 0, 'Steady': 1}
medication_columns = df.iloc[:,24:47]
medication_columns = medication_columns.columns
df[medication_columns] = df[medication_columns].replace(medication_mapping)

df['age'] = df['age'].map(age_mapping)
df['male'] = (df['gender'] == 'Male').astype(int)
df['female'] = (df['gender'] == 'Female').astype(int)
df['race'] = df['race'].map(race_mapping)
df['readmitted']= df['readmitted'].map(readmitted_mapping)
df = df.drop(['gender','weight'], axis=1)



    # Print the column index

print(df.head())
# X = df.drop(columns='readmitted', axis=1)
# y = df.iloc[:,-1]
#
# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.75,random_state=42,stratify=y)

# plt.figure()
# plt.hist(y_train)
# plt.ylabel('#instances')
# plt.xlabel('class')
# plt.show()

# classifiers = [KNeighborsClassifier(3),
#                KNeighborsClassifier(5),
#                KNeighborsClassifier(10),
#                DecisionTreeClassifier(max_depth=5),
#                DecisionTreeClassifier(max_depth=10),
#                DecisionTreeClassifier(max_depth=20),
#                RandomForestClassifier(n_estimators=1000, max_depth=3),
#                RandomForestClassifier(n_estimators=1000, max_depth=5),
#                LogisticRegression(max_iter=100000),
#                GaussianNB(),
#                MLPClassifier(hidden_layer_sizes=(100, 100, 100),
#                              max_iter=10000, activation='relu'),
#                MLPClassifier(hidden_layer_sizes=(100, 100, 100),
#                              max_iter=10000, activation='tanh')]
#
# clf_names = ["Nearest Neighbors (k=3)",
#              "Nearest Neighbors (k=5)",
#              "Nearest Neighbors (k=10)",
#              "Decision Tree (Max Depth=5)",
#              "Decision Tree (Max Depth=10)",
#              "Decision Tree (Max Depth=20)",
#              "Random Forest (Max Depth=3)",
#              "Random Forest (Max Depth=5)",
#              "Logistic Regression",
#              "Gaussian Naive Bayes",
#              "MLP (RelU)",
#              "MLP (tanh)"]
#
# scores_micro = dict()
# scores_macro = dict()
# scores_mcc = dict()
# for name, clf in zip(clf_names, classifiers):
#     print("fitting classifier", name)
#     clf.fit(X_train, y_train)
#     print("predicting labels for classifier", name)
#     y_pred = clf.predict(X_test)
#     scores_micro[name] = precision_recall_fscore_support(
#         y_test, y_pred, average="micro")
#     scores_macro[name] = precision_recall_fscore_support(
#         y_test, y_pred, average="macro")
#     scores_mcc[name] = matthews_corrcoef(Y_test, y_pred)
#
# scores_micro_df = pd.DataFrame(scores_micro, index=[
#                                'precision (micro)', 'recall (micro)', 'fscore (micro)', 'support'])
# scores_micro_df = scores_micro_df[0:3]  # drop support
# scores_macro_df = pd.DataFrame(scores_macro, index=[
#                                'precision (macro)', 'recall (macro)', 'fscore (macro)', 'support'])
# scores_macro_df = scores_macro_df[0:3]
# scores_df = scores_macro_df._append(scores_micro_df)._append(
#     pd.Series(scores_mcc, name='MCC'))
