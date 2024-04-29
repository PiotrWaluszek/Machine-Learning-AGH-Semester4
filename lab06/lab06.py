# Importing necessary libraries
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets

# Loading the breast cancer dataset
data_breast_cancer = datasets.load_breast_cancer(as_frame=True)

# Selecting features and targets from the dataset
bc_features = data_breast_cancer['data'][['mean texture', 'mean symmetry']]
bc_labels = data_breast_cancer['target'].astype(np.uint8)
X_bc_train, X_bc_test, y_bc_train, y_bc_test = train_test_split(bc_features, bc_labels, test_size=0.2)

# Instantiating classifiers
decision_tree_C = DecisionTreeClassifier()
logistic_regression_C = LogisticRegression()
knn_C = KNeighborsClassifier()

# Training classifiers
decision_tree_C.fit(X_bc_train, y_bc_train)
logistic_regression_C.fit(X_bc_train, y_bc_train)
knn_C.fit(X_bc_train, y_bc_train)

# Configuring voting classifiers
voting_clf_hard = VotingClassifier(
    estimators=[('dt', decision_tree_C),
                ('lr', logistic_regression_C),
                ('knn', knn_C)], voting='hard')
voting_clf_soft = VotingClassifier(
    estimators=[('dt', decision_tree_C),
                ('lr', logistic_regression_C),
                ('knn', knn_C)], voting='soft')

# Training voting classifiers
voting_clf_hard.fit(X_bc_train, y_bc_train)
voting_clf_soft.fit(X_bc_train, y_bc_train)

# Evaluating all classifiers
acc_list = []
classifiers = [decision_tree_C, logistic_regression_C, knn_C, voting_clf_hard, voting_clf_soft]
for i in classifiers:
    accuracy_bc_train = accuracy_score(y_bc_train, i.predict(X_bc_train))
    accuracy_bc_test = accuracy_score(y_bc_test, i.predict(X_bc_test))
    acc_list.append((accuracy_bc_train, accuracy_bc_test))

# Saving accuracy data and models
with open('acc_vote.pkl', 'wb') as f:
    pickle.dump(acc_list, f)
with open('vote.pkl', 'wb') as f:
    pickle.dump(classifiers, f)

# Applying Bagging and Pasting
bagging_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=30)
bagging_clf.fit(X_bc_train, y_bc_train)

bagging_clf_50pct = BaggingClassifier(DecisionTreeClassifier(), n_estimators=30, max_samples=0.5)
bagging_clf_50pct.fit(X_bc_train, y_bc_train)

pasting_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=30, bootstrap=False)
pasting_clf.fit(X_bc_train, y_bc_train)

pasting_clf_50pct = BaggingClassifier(DecisionTreeClassifier(), n_estimators=30, bootstrap=False, max_samples=0.5)
pasting_clf_50pct.fit(X_bc_train, y_bc_train)

# Random Forest and Boosting methods
rnd_clf = RandomForestClassifier(n_estimators=30)
rnd_clf.fit(X_bc_train, y_bc_train)

ada_clf = AdaBoostClassifier(n_estimators=30)
ada_clf.fit(X_bc_train, y_bc_train)

gbrt_clf = GradientBoostingClassifier(n_estimators=30)
gbrt_clf.fit(X_bc_train, y_bc_train)

# Evaluating ensemble methods
classifiers_6 = [bagging_clf, bagging_clf_50pct, pasting_clf, pasting_clf_50pct, rnd_clf, ada_clf, gbrt_clf]
acc_list_6 = []
for i in classifiers_6:
    train_acc = accuracy_score(y_bc_train, i.predict(X_bc_train))
    test_acc = accuracy_score(y_bc_test, i.predict(X_bc_test))
    acc_list_6.append((train_acc, test_acc))

# Saving ensemble methods' data
with open('acc_bag.pkl', 'wb') as f:
    pickle.dump(acc_list_6, f)
with open('bag.pkl', 'wb') as f:
    pickle.dump(classifiers_6, f)

# Re-splitting the dataset for a randomized feature approach
X_bc_train, X_bc_test, y_bc_train, y_bc_test = train_test_split(data_breast_cancer['data'], data_breast_cancer['target'].astype(np.uint8), test_size=0.2)

# Randomized feature selection
rnd_clf_random = BaggingClassifier(n_estimators=30, max_features=2, bootstrap=True, max_samples=0.5)
rnd_clf_random.fit(X_bc_train, y_bc_train)
train_acc = accuracy_score(y_bc_train, rnd_clf_random.predict(X_bc_train))
test_acc = accuracy_score(y_bc_test, rnd_clf_random.predict(X_bc_test))
acc_random = [train_acc, test_acc]

# Saving randomized feature model and data
rnd_clf_random_to_list = [rnd_clf_random]
with open('acc_fea.pkl', 'wb') as f:
    pickle.dump(acc_random, f)
with open('fea.pkl', 'wb') as f:
    pickle.dump(rnd_clf_random_to_list, f)

# Analyzing individual tree performance in the random feature model
results = []
for i, estimator in enumerate(rnd_clf_random.estimators_):
    train_accuracy = estimator.score(X_bc_train.iloc[:, rnd_clf_random.estimators_features_[i]].values, y_bc_train)
    test_accuracy = estimator.score(X_bc_test.iloc[:, rnd_clf_random.estimators_features_[i]].values, y_bc_test)
    selected_features = list(X_bc_train.columns[rnd_clf_random.estimators_features_[i]])
    results.append((train_accuracy, test_accuracy, selected_features))

# Storing ranked results based on performance
df = pd.DataFrame(results, columns=['Train Accuracy', 'Test Accuracy', 'Selected Features'])
df = df.sort_values(by=['Train Accuracy', 'Test Accuracy'], ascending=False)
df.to_pickle('acc_fea_rank.pkl')
