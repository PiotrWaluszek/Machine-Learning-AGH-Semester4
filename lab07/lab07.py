#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import fetch_openml
import numpy as np
import pickle

mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
mnist.target = mnist.target.astype(np.uint8)
X = mnist["data"]
y = mnist["target"]


# In[2]:


# ex. 3.1 up to ex 3.4
from statistics import mode
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, silhouette_score

silhouette_list = [] # lista wskaznikow
confusion_matrix_10_clusters = {}

for i in range(8, 13):
    kmeans = KMeans(n_clusters=i, n_init=10)
    kmeans.fit(X)
    y_pred = kmeans.predict(X)
    
    temp_silhouette_score = silhouette_score(X, y_pred)
    silhouette_list.append(silhouette_score(X, kmeans.labels_))
    
    if i == 10:
        confusion_matrix_10_clusters = confusion_matrix(y, y_pred)
        print(f"Confusion matrix: {confusion_matrix_10_clusters}")

print(silhouette_list)

with open("kmeans_sil.pkl", "wb") as output_file:
    pickle.dump(silhouette_list, output_file)


# In[3]:


# ex 3.5
max_indices = np.argmax(confusion_matrix_10_clusters, axis=1)
max_indices_sorted = sorted(set(max_indices))

print(max_indices_sorted)

with open("kmeans_argmax.pkl", "wb") as output_file:
    pickle.dump(max_indices_sorted, output_file)


# In[4]:


# ex 3.6
distances_with_heuristic = np.array([np.linalg.norm(X[i] - X[j]) for i in range(300) for j in range(len(X))])
distances_with_heuristic = list(filter(lambda x: x != 0, distances_with_heuristic))

distances_with_heuristic = list(np.sort(distances_with_heuristic)[:10])

print(distances_with_heuristic)

with open("dist.pkl", "wb") as output_file:
    pickle.dump(distances_with_heuristic, output_file)


# In[5]:


# ex 3.7 / 3.8
from sklearn.cluster import DBSCAN

s = np.mean(distances_with_heuristic[:3])

eps_min = s
eps_max = s + 0.1 * s
eps_step = 0.04 * s
eps_values = np.arange(eps_min, eps_max, eps_step)

unique_labels_list = []

for eps in eps_values:
    dbscan = DBSCAN(eps=eps)
    dbscan.fit(X)
    number_of_unique_labels = len(np.unique(dbscan.labels_))
    unique_labels_list.append(number_of_unique_labels)

print(unique_labels_list)

with open("dbscan_len.pkl", "wb") as output_file:
    pickle.dump(unique_labels_list, output_file)

