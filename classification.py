import numpy as np
import os
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin
import matplotlib.pyplot as plt


def get_features(f_name):
    vector = np.load(f_name + '.npy')
    return vector


def get_labels(string):
    lab = 0
    if string == 'PRE':
        lab = 0
    elif string == 'POST':
        lab = 1
    return lab


feat_type = 'predictions'
classifier = 'VGG19'
features_path = os.path.join('dataset', feat_type, classifier)
metadata_file = 'metadata.csv'

data = pd.read_csv(metadata_file, header=0, sep=',', names=['filename', 'singer', 'class', 'date_rec'])

y = []

for file in data.iterrows():
    filename, _ = os.path.splitext(file[1]['filename'])
    label = get_labels(file[1]['class'])
    features_tmp = get_features(os.path.join(features_path, filename))
    y.append(label)

    if 'X' in locals():
        X = np.concatenate((X, features_tmp))
    else:
        X = features_tmp

n_clusters = 2
pca = PCA(n_components=16).fit_transform(X)
X = pca
kmeans = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10).fit(X)
# plt.figure()
# plt.subplot(211)
# plt.plot(pca)
# plt.subplot(212)
# plt.plot(y)
# plt.savefig('pca.png')

# plt.figure()
# plt.subplot(211)
# plt.plot(kmeans)
# plt.subplot(212)
# plt.plot(y)
# plt.savefig('kmeans.png')

# https://scikit-learn.org/stable/auto_examples/cluster/plot_mini_batch_kmeans.html#sphx-glr-auto-examples-cluster-plot-mini-batch-kmeans-py
# fig = plt.figure(figsize=(8, 3))
fig = plt.figure()
# fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.9)
colors = ['#4EACC5', '#FF9C34', '#4E9A06']

# We want to have the same colors for the same cluster from the
# MiniBatchKMeans and the KMeans algorithm. Let's pair the cluster centers per
# closest one.
k_means_cluster_centers = kmeans.cluster_centers_
k_means_labels = pairwise_distances_argmin(X, k_means_cluster_centers)
# KMeans
# ax = fig.add_subplot(111)
# for k, col in zip(range(n_clusters), colors):
#     my_members = k_means_labels == k
#     cluster_center = k_means_cluster_centers[k]
#     # ax.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=col, marker='.')
    # ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=6)
# ax.set_title('KMeans')
# ax.set_xticks(())
# ax.set_yticks(())
# plt.show()
# plt.savefig('kmeans_clusters.png')

plt.figure(figsize=(10, 10))
for i in range(0, len(y)):
    if y[i] == 0:
        plt.scatter(X[i, 0], X[i, 1], s=200,  c='black', marker='.')
    else:
        plt.scatter(X[i, 0], X[i, 1], s=200, c='red', marker='.')

for k, col in zip(range(n_clusters), colors):
    my_members = k_means_labels == k
    cluster_center = k_means_cluster_centers[k]
    plt.scatter(cluster_center[0], cluster_center[1], s=500,  c=col, marker='*')

plt.show()
# plt.savefig('kmeans_clusters.png')

print('Done')


