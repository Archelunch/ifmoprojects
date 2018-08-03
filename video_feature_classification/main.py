# -*- coding: utf-8 -*-
import os, sys
import cv2
import random
import numpy as np
from tqdm import tqdm
import pickle

sys.path.append('../')
from pytorch.common.datasets_parsers.av_parser import AVDBParser

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.decomposition import PCA
from accuracy import Accuracy


def get_data(dataset_root, file_list, max_num_clips=0, max_num_samples=50):
    dataset_parser = AVDBParser(dataset_root, os.path.join(dataset_root, file_list),
                                max_num_clips=max_num_clips, max_num_samples=max_num_samples,
                                ungroup=False, load_image=True)
    data = dataset_parser.get_data()
    print('clips count:', len(data))
    print('frames count:', dataset_parser.get_dataset_size())
    return data





def calc_features1(data, draw=False):
    feat, targets = [], []
    for clip in data:
        if not clip.data_samples[0].labels in [7, 8]:
            continue

        # TODO: придумайте способы вычисления признаков на основе ключевых точек
        # distance between landmarks
        for i, sample in enumerate(clip.data_samples):
            if i % 8 != 0:
                continue
            dist = []
            lm_ref = sample.landmarks[30] # point on the nose
            for j in range(len(sample.landmarks)):
                lm = sample.landmarks[j]
                dist.append(np.sqrt((lm_ref[0] - lm[0]) ** 2 + (lm_ref[1] - lm[1]) ** 2))
            feat.append(dist)
            targets.append(sample.labels)

            if draw:
                img = cv2.imread(sample.img_rel_path)
                for lm in sample.landmarks:
                    cv2.circle(img, (int(lm[0]), int(lm[1])), 3, (0, 0, 255), -1)
                cv2.imshow(sample.text_labels, img)
                cv2.waitKey(100)

    print('feat count:', len(feat))
    return np.asarray(feat, dtype=np.float32), np.asarray(targets, dtype=np.float32)






def calc_features(data):
    orb = cv2.ORB_create()

    progresser = tqdm(iterable=range(0, len(data)),
                      desc='calc video features',
                      total=len(data),
                      unit='files')

    feat, targets = [], []
    for i in progresser:
        clip = data[i]

        rm_list = []
        for sample in clip.data_samples:

            dist = []
            lm_ref = sample.landmarks[30] # point on the nose
            for j in range(len(sample.landmarks)):
                lm = sample.landmarks[j]
                dist.append(np.sqrt((lm_ref[0] - lm[0]) ** 2 + (lm_ref[1] - lm[1]) ** 2))
            feat.append(dist)
            targets.append(sample.labels)

            # TODO: придумайте способы вычисления признаков по изображению с использованием ключевых точек
            # используйте библиотеку OpenCV

        for sample in rm_list:
            clip.data_samples.remove(sample)

    print('feat count:', len(feat))
    return np.asarray(feat, dtype=np.float32), np.asarray(targets, dtype=np.float32)

def classification(X_train, X_test, y_train, y_test, accuracy_fn, pca_dim):
    if pca_dim > 0:
        pass
        # TODO: выполните сокращение размерности признаков с использованием PCA
        pca = PCA(n_components=pca_dim)
        pca.fit(X_train)
        X_train = pca.transform(X_train)
        X_test = pca.transform(X_test)
    # shuffle
    combined = list(zip(X_train, y_train))
    random.shuffle(combined)
    X_train[:], y_train[:] = zip(*combined)

    # TODO: используйте классификаторы из sklearn
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy_fn.by_frames(y_pred)
    accuracy_fn.by_clips(y_pred)


if __name__ == "__main__":
    experiment_name = 'exp_1'
    max_num_clips = 0 # загружайте только часть данных для отладки кода
    use_dump = False # используйте dump для быстрой загрузки рассчитанных фич из файла

    # dataset dir
    base_dir = 'D:/AVER'
    if 1:
        train_dataset_root = r'C:\Users\ipmstud\Desktop\mlschool\ryerson\Video'
        train_file_list = r'C:\Users\ipmstud\Desktop\mlschool\ryerson\train_data_with_landmarks.txt'
        test_dataset_root = r'C:\Users\ipmstud\Desktop\mlschool\ryerson\Video'
        test_file_list = r'C:\Users\ipmstud\Desktop\mlschool\ryerson\test_data_with_landmarks.txt'
    elif 1:
        train_dataset_root = base_dir + '/OMGEmotionChallenge-master/omg_TrainVideos/preproc/frames'
        train_file_list = base_dir + '/OMGEmotionChallenge-master/omg_TrainVideos/preproc/train_data_with_landmarks.txt'
        test_dataset_root =base_dir + '/OMGEmotionChallenge-master/omg_ValidVideos/preproc/frames'
        test_file_list = base_dir + '/OMGEmotionChallenge-master/omg_ValidVideos/preproc/valid_data_with_landmarks.txt'

    if not use_dump:
        # load dataset
        train_data = get_data(train_dataset_root, train_file_list, max_num_clips=0)
        test_data = get_data(test_dataset_root, test_file_list, max_num_clips=0)

        # get features
        train_feat, train_targets = calc_features(train_data)
        test_feat, test_targets = calc_features(test_data)

        accuracy_fn = Accuracy(test_data, experiment_name=experiment_name)

        #with open(experiment_name + '.pickle', 'wb') as f:
        #    pickle.dump([train_feat, train_targets, test_feat, test_targets, accuracy_fn], f, protocol=2)
    else:
        with open(experiment_name + '.pickle', 'rb') as f:
            train_feat, train_targets, test_feat, test_targets, accuracy_fn = pickle.load(f)

    # run classifiers
    classification(train_feat, test_feat, train_targets, test_targets, accuracy_fn=accuracy_fn, pca_dim=0)