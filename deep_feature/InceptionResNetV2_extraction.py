'''
this code is to extract feature of InceptionResNetV2 network
code is from: https://keras.io/api/applications/
'''

from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
import numpy as np
import os
import csv
import scipy.io as sio

# create the pre-trained model
model = InceptionResNetV2(weights='imagenet', include_top=False)
nn_name = 'inceptionresnetv2'

ro = '/home/na/1_MAD/2_Data/1_good_FRGC2.0/'
dst = '/home/na/1_MAD/2_Data/2_good_FRGC2.0_features/'
if os.path.exists(dst) is False:
    os.makedirs(dst)

dbs = os.listdir(ro)
for db in dbs:
    db_path = ro + db + '/'
    train_file = db_path + 'train.csv'
    test_file = db_path + 'test.csv'
    print(db)
    img_path = db_path

    dst_path = dst + db + '/'
    if os.path.exists(dst_path) is False:
        os.makedirs(dst_path)

    # read train feature
    train_feature = []
    train_label = []
    f = csv.reader(open(train_file, 'r', newline=''))
    for row in f:
        if row[0] == 'image_name': continue
        fi = img_path + 'train/' + row[0]
        img = image.load_img(fi, target_size=(299, 299))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        feature = model.predict(x)
        fea = feature.flatten()
        train_feature.append(fea)
        train_label.append(int(row[1]))

    # read test feature
    test_feature = []
    test_label = []
    f = csv.reader(open(test_file,'r', newline=''))
    for row in f:
        if row[0] == 'image_name': continue
        fi = img_path + 'test/' + row[0]
        img = image.load_img(fi, target_size=(299, 299))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        feature = model.predict(x)
        fea = feature.flatten()
        test_feature.append(fea)
        test_label.append(int(row[1]))

    # save data
    dict = {}
    dict['train_feature'] = np.asarray(train_feature)
    dict['train_label'] = np.asarray(train_label).flatten()
    dict['test_feature'] = np.asarray(test_feature)
    dict['test_label'] = np.asarray(test_label).flatten()
    dst_file = dst_path + nn_name + '_train_test.mat'
    sio.savemat(dst_file, mdict=dict)

    print(np.shape(train_feature))
    print(np.shape(train_label))
    print(np.shape(test_feature))
    print(np.shape(test_label))

