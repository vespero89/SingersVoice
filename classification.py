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
classifier = 'Xception'
features_path = os.path.join('dataset', feat_type, classifier)
metadata_file = 'metadata_test_FA.csv'

data = pd.read_csv(metadata_file, header=0, sep=',', names=['filename', 'singer', 'class', 'date_rec'])

y = []
date = []
for file in data.iterrows():
    filename, _ = os.path.splitext(file[1]['filename'])
    label = get_labels(file[1]['class'])
    features_tmp = get_features(os.path.join(features_path, filename))
    y.append(label)
    date.append(file[1]['date_rec'])

    if 'X' in locals():
        X = np.concatenate((X, features_tmp))
    else:
        X = features_tmp

n_clusters = 2
reduced_data = PCA(n_components=4).fit_transform(X)

kmeans = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10, random_state=21).fit(reduced_data)

# https://scikit-learn.org/stable/auto_examples/cluster/plot_mini_batch_kmeans.html#sphx-glr-auto-examples-cluster-plot-mini-batch-kmeans-py
# fig = plt.figure(figsize=(8, 3))
# fig = plt.figure()
# fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.9)
colors = ['red', 'blue']

# We want to have the same colors for the same cluster from the
# MiniBatchKMeans and the KMeans algorithm. Let's pair the cluster centers per
# closest one.
k_means_cluster_centers = kmeans.cluster_centers_
k_means_labels = pairwise_distances_argmin(reduced_data, k_means_cluster_centers)
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
# Step size of the mesh. Decrease to increase the quality of the VQ.
h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
# Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel(), xx.ravel(), yy.ravel()])
#
# # Put the result into a color plot
# Z = Z.reshape(xx.shape)
# plt.clf()
# plt.imshow(Z, interpolation='nearest',
#            extent=(xx.min(), yy.min(), xx.max(), yy.max()),
#            cmap=plt.cm.Paired,
#            aspect='auto', origin='lower')


for i in range(0, len(y)):
    if y[i] == 0:
        plt.scatter(reduced_data[i, 0], reduced_data[i, 1], s=500,  c='green', marker='.')
    else:
        plt.scatter(reduced_data[i, 0], reduced_data[i, 1], s=500, c='black', marker='.')

    plt.annotate(date[i], (reduced_data[i, 0], reduced_data[i, 1]))

for k, col in zip(range(n_clusters), colors):
    my_members = k_means_labels == k
    cluster_center = k_means_cluster_centers[k]
    plt.scatter(cluster_center[0], cluster_center[1], s=200,  c=col, marker='x')

plt.show()

print('Done')


