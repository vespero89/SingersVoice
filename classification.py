import numpy as np
import os
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


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

pca = PCA(n_components=2).fit_transform(X)



print('Done')


