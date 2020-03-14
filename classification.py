import numpy as np
import os
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin
import matplotlib.pyplot as plt
import collections


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


def experiment(sel_criteria, metadata, out_folder, reduce=2):
    for i in range(0, len(filter_criteria)):
        metadata = metadata.loc[metadata[filter_criteria[i]] == sel_criteria[i]]

    if len(metadata):
        print('Starting Experiment for {} - {}'.format(sel_criteria[0], sel_criteria[1]))
        y = []
        date_rec = []
        plot_label = []
        for file in metadata.iterrows():
            filename, _ = os.path.splitext(file[1]['filename'])
            label = get_labels(file[1]['class'])
            features_tmp = get_features(os.path.join(features_path, filename))
            y.append(label)
            date_rec.append(file[1]['date_rec'])
            plot_label.append(file[1]['date_rec'] + '_' + file[1]['class'])

            if 'X' in locals():
                X = np.concatenate((X, features_tmp))
            else:
                X = features_tmp

        counter = collections.Counter(date_rec)
        n_clusters = max(len(counter), 2)
        print('Found N={} dates of rec, using as cluster number.'.format(n_clusters))

        if reduce:
            pca_comp = min(reduce, 2)
            reduced_data = PCA(n_components=pca_comp).fit_transform(X)
            out_filename = '{}_{}_{}_PCA-{}c_KM-{}c.png'.format(sel_criteria[0], sel_criteria[1], c, pca_comp, n_clusters)
            kmeans = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10, random_state=21).fit(reduced_data)
            k_means_cluster_centers = kmeans.cluster_centers_
            k_means_labels = pairwise_distances_argmin(reduced_data, k_means_cluster_centers)
            data_clustered = kmeans.predict(reduced_data)
        else:
            kmeans = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10, random_state=21).fit(X)
            out_filename = '{}_{}_{}_{}_KM-{}c.png'.format(sel_criteria[0], sel_criteria[1], c, X.shape[1], n_clusters)
            k_means_cluster_centers = kmeans.cluster_centers_
            data_clustered = kmeans.predict(X)
            reduced_data = PCA(n_components=2).fit_transform(X)

        # https://scikit-learn.org/stable/auto_examples/cluster/plot_mini_batch_kmeans.html#sphx-glr-auto-examples-cluster-plot-mini-batch-kmeans-py
        plt.figure(figsize=(10, 10))
        colors = ['red', 'blue', 'black', 'green', 'yellow', 'gray', 'pink', 'cyan']

        # Step size of the mesh. Decrease to increase the quality of the VQ.
        h = .02  # point in the mesh [x_min, x_max]x[y_min, y_max].
        x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
        y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        for i in range(0, len(y)):
            if y[i] == 0:
                plt.scatter(reduced_data[i, 0], reduced_data[i, 1], s=500, c=colors[data_clustered[i]], marker='.')
            else:
                plt.scatter(reduced_data[i, 0], reduced_data[i, 1], s=500, c=colors[data_clustered[i]], marker='.')
            plt.annotate(plot_label[i], (reduced_data[i, 0], reduced_data[i, 1]))

        if reduce:
            for k in range(n_clusters):
                my_members = k_means_labels == k
                cluster_center = k_means_cluster_centers[k]
                plt.scatter(cluster_center[0], cluster_center[1], s=200, c='red', marker='x')

        plt.title('Singer : {} - {} - {}'.format(singer, sel_criteria[1], c))
        plt.grid(True)
        plt.savefig(os.path.join(out_folder, out_filename))
        plt.show()
        return True
    else:
        return False


feat_type = 'predictions'
classifiers = ['Xception', 'VGG19', 'ResNet50']
metadata_file = 'metadata.csv'
filter_criteria = ['singer', 'test']
selection_criteria = ['Regina_Carbone', 'Drink']
data = pd.read_csv(metadata_file, header=0, sep=',', names=['filename', 'singer', 'class', 'date_rec', 'test'])
singers_list = collections.Counter(data['singer'].tolist()).keys()
reduction = 4
for c in classifiers:
    images_dir = 'Classified_wPCA-{}c_{}'.format(reduction, selection_criteria[1])
    os.makedirs(images_dir, exist_ok=True)
    features_path = os.path.join('dataset', feat_type, c)
    for singer in singers_list:
        res = experiment(sel_criteria=[singer, selection_criteria[1]], metadata=data,
                         out_folder=images_dir, reduce=reduction)
        if res:
            print('Done')
        else:
            print('ERROR! - No file found')

    print('Everything done')

