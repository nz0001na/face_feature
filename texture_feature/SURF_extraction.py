'''
In this example, we will use bag of visual words approach
to extract SURF feature
'''


import cv2
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from scipy.cluster.vq import kmeans,vq
import scipy.io as sio
import csv

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
    des_list = []
    f = csv.reader(open(train_file, 'r'))
    for row in f:
        if row[0] == 'image_name': continue
        fi = img_path + 'train/' + row[0]
        img = cv2.imread(fi)
        img = cv2.resize(img, (270, 270))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        surf = cv2.xfeatures2d.SURF_create()
        (keypoint, descriptor) = surf.detectAndCompute(gray, None)
        des_list.append((fi, descriptor))
        train_label.append(int(row[1]))

    descriptors = des_list[0][1]
    for image_path, descriptor in des_list[1:]:
        descriptors = np.vstack((descriptors, descriptor))

    print(descriptors.shape)
    descriptors_float = descriptors.astype(float)

    # bag of words: 200 code words
    k = 200
    voc, variance = kmeans(descriptors_float, k, 1)
    train_feature = np.zeros((len(train_label), k), "float32")
    for i in range(len(train_label)):
        words, distance = vq(des_list[i][1], voc)
        for w in words:
            train_feature[i][w] += 1

    stdslr = StandardScaler().fit(train_feature)
    train_feature = stdslr.transform(train_feature)


    # read test feature
    test_feature = []
    test_label = []
    des_list_test = []
    f = csv.reader(open(test_file, 'r'))
    for row in f:
        if row[0] == 'image_name': continue
        fi = img_path + 'test/' + row[0]
        img = cv2.imread(fi)
        img = cv2.resize(img, (270, 270))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        surf = cv2.xfeatures2d.SURF_create()
        (keypoint, descriptor) = surf.detectAndCompute(gray, None)
        des_list_test.append((fi, descriptor))
        test_label.append(int(row[1]))

    test_feature = np.zeros((len(test_label), k), "float32")
    for i in range(len(test_label)):
        words, distance = vq(des_list_test[i][1], voc)
        for w in words:
            test_feature[i][w] += 1
    test_feature = stdslr.transform(test_feature)

    dict = {}
    dict['train_feature'] = np.asarray(train_feature)
    dict['train_label'] = np.asarray(train_label).flatten()
    dict['test_feature'] = np.asarray(test_feature)
    dict['test_label'] = np.asarray(test_label).flatten()
    dst_file = dst_path + 'SURF_train_test.mat'
    sio.savemat(dst_file, mdict=dict)

    print(np.shape(train_feature))
    print(np.shape(train_label))
    print(np.shape(test_feature))
    print(np.shape(test_label))

