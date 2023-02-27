'''
This code is to extract IQM feature from grayscale cropped faces (270*270)
'''

import csv
from bob.ip.qualitymeasure import compute_quality_features, compute_msu_iqa_features
import scipy.io as sio
import matplotlib._png as png
from PIL import Image
import numpy as np
import os
import cv2

src_path = '/media/zn/BE2C40612C4016B5/2_zn_research/1_MAD/Data/3_AMSL_FFHQ_cropped_grayscale_resize/size_270_270/'

list_path = '/media/zn/BE2C40612C4016B5/2_zn_research/1_MAD/Data/100_detection_experiment/1_train_test_list/'
train_file = list_path + 'new_train_list.csv'
test_file = list_path + 'new_test_list.csv'

# read train feature
train_feature_18 = []
train_label_18 = []
train_feature_121 = []
train_label_121 = []
f = csv.reader(open(train_file, 'rb'))
for row in f:
    fi = src_path + row[0] + '/' + row[1]
    im = cv2.imread(fi)
    r = im.transpose(2, 0, 1)

    f_18 = compute_quality_features(r)
    train_label_18.append(int(row[2]))
    train_feature_18.append(f_18)

    f_121 = compute_msu_iqa_features(r)
    train_feature_121.append(f_121)
    train_label_121.append(int(row[2]))

# read test feature
test_feature_18 = []
test_label_18 = []
test_feature_121 = []
test_label_121 = []
f = csv.reader(open(test_file, 'rb'))
for row in f:
    fi = src_path + row[0] + '/' + row[1]
    im = cv2.imread(fi)
    r = im.transpose(2, 0, 1)

    f_18 = compute_quality_features(r)
    test_label_18.append(int(row[2]))
    test_feature_18.append(f_18)

    f_121 = compute_msu_iqa_features(r)
    test_feature_121.append(f_121)
    test_label_121.append(int(row[2]))


# save data
dict18 = {}
dict18['train_feature'] = np.asarray(train_feature_18)
dict18['train_label'] = np.asarray(train_label_18).flatten()
dict18['test_feature'] = np.asarray(test_feature_18)
dict18['test_label'] = np.asarray(test_label_18).flatten()
dst_file = '/media/zn/BE2C40612C4016B5/2_zn_research/1_MAD/Data/8_IQM_feature/IQM18_train_test.mat'
sio.savemat(dst_file, mdict=dict18)

dict121 = {}
dict121['train_feature'] = np.asarray(train_feature_121)
dict121['train_label'] = np.asarray(train_label_121).flatten()
dict121['test_feature'] = np.asarray(test_feature_121)
dict121['test_label'] = np.asarray(test_label_121).flatten()
dst_file = '/media/zn/BE2C40612C4016B5/2_zn_research/1_MAD/Data/8_IQM_feature/IQM121_train_test.mat'
sio.savemat(dst_file, mdict=dict121)
