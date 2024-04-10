from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix
import numpy as np
import pickle

mnist = fetch_openml('mnist_784', version=1)

X, y = mnist['data'], mnist['target']

y_0 = (y == '0')

X_train, X_test, y_train_0, y_test_0 = train_test_split(X, y_0, test_size=0.2, random_state=42)

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_0)

acc_train = sgd_clf.score(X_train, y_train_0)
acc_test = sgd_clf.score(X_test, y_test_0)

with open('sgd_acc.pkl', 'wb') as f:
    pickle.dump([acc_train, acc_test], f)

scores = cross_val_score(sgd_clf, X_train, y_train_0, cv=3, scoring='accuracy', n_jobs=-1)

with open('sgd_cva.pkl', 'wb') as f:
    pickle.dump(scores, f)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

sgd_clf_multi = SGDClassifier(random_state=42)
sgd_clf_multi.fit(X_train, y_train)

y_pred = sgd_clf_multi.predict(X_test)

conf_mx = confusion_matrix(y_test, y_pred)

with open('sgd_cmx.pkl', 'wb') as f:
    pickle.dump(conf_mx, f)