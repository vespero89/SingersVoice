import os
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.applications.vgg19 import VGG19
from tensorflow.python.keras.applications.resnet50 import ResNet50
from tensorflow.python.keras.applications.xception import Xception
from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.applications.xception import preprocess_input
import numpy as np


def get_features(img_path, classifier):
    if classifier == 'VGG19':
        model = VGG19(weights='imagenet', include_top=False, pooling='avg')
        img = image.load_img(img_path, target_size=(224, 224))
    elif classifier == 'ResNet50':
        model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
        img = image.load_img(img_path, target_size=(224, 224))
    elif classifier == 'Xception':
        model = Xception(weights='imagenet', include_top=False, pooling='avg')
        img = image.load_img(img_path, target_size=(299, 299))
    else:
        return False

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    flatten = model.predict(x)
    if chunked:
        pred_dir = 'predictions_chunked'
        if normalized:
            pred_dir = pred_dir + '_normalized'
    else:
        pred_dir = 'predictions'
    feat_folder = os.path.join('dataset', pred_dir, classifier)
    os.makedirs(feat_folder, exist_ok=True)
    features_name = os.path.join(feat_folder, os.path.basename(img_path))
    features_name = features_name.split('.')[0]
    features_name = features_name.replace('_16bit', '')
    np.save(features_name, flatten)
    return list(flatten[0])


feat_type = 'LOGMEL_SPECTROGRAMS_CHUNKED'
features_path = os.path.join('dataset', feat_type)
chunked = True
normalized = False
classifiers = ['VGG19', 'ResNet50', 'Xception']
logmels = []
X = []
y = []
for (_, dirs, filenames) in os.walk(features_path):
    if chunked:
        for d in dirs:
            files = os.listdir(os.path.join(features_path, d))
            for f in files:
                logmels.append(os.path.join(d, f))
    else:
        logmels.extend(filenames)
    break

for c in classifiers:
    for lm in logmels:
        X.append(get_features(os.path.join(features_path, lm), classifier=c))
        y.append(0)

print('Done')


