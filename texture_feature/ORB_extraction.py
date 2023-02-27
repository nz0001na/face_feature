'''
In this example, we will use bag of visual words approach
to extract ORB feature
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

    dst_path = dst + db + '/'
    if os.path.exists(dst_path) is False:
        os.makedirs(dst_path)
    if os.path.exists(dst_path + 'ORB_train_test.mat') is False:
        print(db)

    if os.path.exists(dst_path + 'ORB_train_test.mat') is True:
        continue

    # if db == '4_FERET_stylegan' or db == '2_FERET_facemorpher' : continue
    # or db == '3_FERET_opencv'
    db_path = ro + db + '/'
    train_file = db_path + 'train.csv'
    test_file = db_path + 'test.csv'

    img_path = db_path

    # read train feature
    train_feature = []
    train_label = []
    des_list = []
    f = csv.reader(open(train_file, 'rb'))
    for row in f:
        if row[0] == 'image_name': continue
        fi = img_path + 'train/' + row[0]
        # if row[0] == 'morph/00402_940422_fa_00403_940422_fa.jpg' \
        #         or row[0] == 'morph/00449_940422_fa.png_00450_940422_fa.png.jpg' \
        #         or row[0] == 'morph/00425_940422_fa.png_00426_940422_fa.png.jpg'\
        #         or row[0] == 'morph/00450_940422_fa.png_00452_940422_fa.png.jpg' :
        #     continue

        img = cv2.imread(fi)
        img = cv2.resize(img, (270, 270))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        orb = cv2.ORB_create()
        kp = orb.detect(img, None)
        keypoint, descriptor = orb.compute(gray, kp)
        des_list.append((fi, descriptor))
        train_label.append(int(row[1]))

    descriptors = des_list[0][1]
    for image_path, descriptor in des_list[1:]:
        # print(image_path)
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
    f = csv.reader(open(test_file, 'rb'))
    for row in f:
        if row[0] == 'image_name': continue
        fi = img_path + 'test/' + row[0]
        # if row[0] == 'morph/00327_940422_fa.png_00328_940422_fa.png.jpg' \
        #         or row[0] == 'morph/00358_940422_fa.png_00359_940422_fa.png.jpg' \
        #         or row[0] == 'morph/00332_940422_fa.png_00335_940422_fa.png.jpg'\
        #         or row[0] == 'morph/00453_940422_fa.png_00460_940422_fa.png.jpg' :
        #     continue

        img = cv2.imread(fi)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (270, 270))
        orb = cv2.ORB_create()
        kp = orb.detect(img, None)
        keypoint, descriptor = orb.compute(gray, kp)
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
    dst_file = dst_path + 'ORB_train_test.mat'
    sio.savemat(dst_file, mdict=dict)

    print(np.shape(train_feature))
    print(np.shape(train_label))
    print(np.shape(test_feature))
    print(np.shape(test_label))

