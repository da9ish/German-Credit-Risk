import pandas as pd
import numpy as np
from pandas import DataFrame

import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer

from sklearn.base import TransformerMixin

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm, tree

from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.pipeline import Pipeline, FeatureUnion

def get_data():
    columns = ["checking_ac", "duration", "credit_history", "purpose", "amount", "saving_ac",
           "employment_status", "installment_rate", 'personal_status_sex', "debtor_guarantor", "residence_since",
          "property", "age", "installment_plan", "housing", "existing_credits", "job", "liable_count", "telephone",
          "foreign_worker", "target"]
    df = pd.read_csv("data/german.data2.csv", delimiter=' ', index_col=False, names=columns)
    return df

data = get_data()
data.target.replace({1:0, 2: 1}, inplace=True)

enc_data = pd.get_dummies(data)
# print(enc_data.columns.values)

# sns.distplot(enc_data.amount)
# plt.show()

plt.subplot('211')
sns.violinplot(x=data.credit_history, y=data.amount, hue='target', data=data, palette='muted', split=True)
plt.subplot('212')
sns.boxplot(x=data.credit_history, y=data.amount, hue='target', data=data, palette='muted')
plt.show()

enc_copy = enc_data.copy()
X = enc_copy.drop(['target'], axis = 1)
y = enc_data.target

# print(X.columns)

print(enc_data.shape, X.shape, y.shape)
# print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=2)

print("X Train: " + str(X_train.shape))
print("Y Train: " + str(y_train.shape))

print("X Test: " + str(X_test.shape))
print("Y Test: " + str(y_test.shape))

model = LogisticRegression()
model.fit(X_train, y_train.values.ravel())

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)
acc = accuracy_score(y_test, y_pred)
cnf_matrix = confusion_matrix(y_test, y_pred)
print(acc)
print(cnf_matrix)

svm = svm.SVC()
log_reg = LogisticRegression()
decision_tree = tree.DecisionTreeClassifier()
random_forest = RandomForestClassifier()
ada_boost = AdaBoostClassifier()
gradient_boost = GradientBoostingClassifier()

std_sclr = StandardScaler()

pipeline = Pipeline([
    ('transform', Pipeline([
        # ('std_sclr', std_sclr),
        ('log_transform', FunctionTransformer(np.log1p)),
    ])),
    
    # ('log', LogTransform()),
    ('estimator', decision_tree)
])

pipeline.fit(X_train, y_train)
# pipeline.transform(X_train, y_train)
y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cnf_matrix = confusion_matrix(y_test, y_pred)
print(acc)
print(cnf_matrix)