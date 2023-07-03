"""
Make train, val, test datasets based on train_test_split.txt, and by sampling val_ratio of the official train data to make a validation set 
Each dataset is a list of metadata, each includes official image id, full image path, class label, attribute labels, attribute certainty scores, and attribute labels calibrated for uncertainty
"""
import os
import random
import pickle
import argparse
from os import listdir
from os.path import isfile, isdir, join
from collections import defaultdict as ddict

import pandas as pd
from config import FITZPATRICK_CSV, FITZ_CONCEPTS_CSV

import numpy as np
from sklearn.model_selection import train_test_split

def extract_data(data_dir):
    cwd = os.getcwd()
    data_path = join(cwd,data_dir + 'fitz')
    val_ratio = 0.15
    test_ratio = 0.15

    fitz_concepts = pd.read_csv(FITZ_CONCEPTS_CSV)
    fitz_concepts = fitz_concepts[fitz_concepts["Do not consider this image"] != 1]

    path_to_id_map = dict() #map from full image path to image id
    id_to_path_map = dict()
    id_to_hash_map = dict()
    attribute_labels_all = ddict(list) #map from image id to a list of attribute labels
    id_list = []
    with open(FITZ_CONCEPTS_CSV, 'r') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            items = line.strip().split(',')
            file_idx = int(items[0])
            img_path = join(data_path, items[1])
            path_to_id_map[img_path] = file_idx
            id_to_path_map[file_idx] = img_path
            id_to_hash_map[file_idx] = items[1][:-4]
            id_list.append(file_idx)
            for j in range(2, len(items)-1):
                attribute_label = int(items[j])
                attribute_labels_all[int(file_idx)].append(attribute_label)

    is_train_test = dict() #map from image id to 0 / 1 (1 = train)
    
    n_samples = len(attribute_labels_all)

    all_data = []

    fitz_data = pd.read_csv(FITZPATRICK_CSV)

    fitz_data['label_int'] = pd.Categorical(fitz_data['label']).codes
    fitz_data['nine_partition_int'] = pd.Categorical(fitz_data['nine_partition_label']).codes
    fitz_data['three_partition_int'] = pd.Categorical(fitz_data['three_partition_label']).codes
    
    num_classes_whole = fitz_data['label_int'].max() + 1
    num_classes_nine = fitz_data['nine_partition_int'].max() + 1
    num_classes_three = fitz_data['three_partition_int'].max() + 1
    print('Number of classes: ', num_classes_whole, num_classes_nine, num_classes_three)

    labels_all = []
    nine_partitions_all = []
    three_partitions_all = []
    binary_all = []

    for idx in id_list:
        img_id = idx
        img_path = id_to_path_map[img_id]
        hash_id = id_to_hash_map[img_id]
        item = fitz_data.loc[fitz_data['md5hash'] == hash_id].values[0]
        label = item[-3]
        nine_partition_label = item[-2]
        three_partition_label = item[-1]
        
        if item[5] == 'malignant':
            benign_malignant = 1
        else:
            benign_malignant = 0
        metadata = {'id': img_id, 'img_path': img_path, 'attribute_label': attribute_labels_all[img_id],
                    'label': label, 'nine_partition_label': nine_partition_label, 
                    'three_partition_label': three_partition_label, 'benign_malignant': benign_malignant}
        all_data.append(metadata)

        labels_all.append(label)
        nine_partitions_all.append(nine_partition_label)
        three_partitions_all.append(three_partition_label)
        binary_all.append(benign_malignant)

    if args.class_label == 'whole':
        labels_list = np.array(labels_all)
        class_size = num_classes_whole
    elif args.class_label == 'nine':
        labels_list = np.array(nine_partitions_all)
        class_size = num_classes_nine
    elif args.class_label == 'three':
        labels_list = np.array(three_partitions_all)
        class_size = num_classes_three
    elif args.class_label == 'binary':
        labels_list = np.array(binary_all)
        class_size = 2

    print(labels_list)
    

    id_list = np.arange(n_samples)
    trainval_id, test_id, trainval_label, test_label = train_test_split(id_list, labels_list, stratify=labels_list, test_size=test_ratio, random_state=42)

    val_ratio = val_ratio/(1 - test_ratio)
    train_id, val_id, train_label, val_label = train_test_split(trainval_id, trainval_label, stratify=trainval_label, test_size=val_ratio, random_state=42)
    
    all_data = np.array(all_data)
    trainval_data = all_data[trainval_id]
    test_data = all_data[test_id]
    val_data = all_data[val_id]
    train_data = all_data[train_id]
    train_data = train_data.tolist()
    val_data = val_data.tolist()
    test_data = test_data.tolist()

    class_cnt = np.zeros(class_size)
    for i in range(len(trainval_data)):
        label = trainval_label[i]
        class_cnt[label] += 1
        
        print('Ratio of each class: ', args.class_label, (class_cnt/len(trainval_data)).tolist())

    print('Size of train/val/test set: ', len(train_data), len(val_data), len(test_data))
    
    return train_data, val_data, test_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dataset preparation')
    parser.add_argument('-save_dir', '-d', help='Where to save the new datasets')
    parser.add_argument('-data_dir', help='Where to load the datasets')
    parser.add_argument('-class_label', help='class label for stratified split')
    args = parser.parse_args()
    train_data, val_data, test_data = extract_data(args.data_dir)

    directory = join(os.getcwd(), args.save_dir, args.class_label)

    for dataset in ['train','val','test']:
        print("Processing %s set" % dataset)
        if not os.path.exists(directory):
            os.makedirs(directory)
        f = open(join(os.getcwd(), args.save_dir, args.class_label, dataset + '.pkl'), 'wb+')
        if 'train' in dataset:
            pickle.dump(train_data, f)
        elif 'val' in dataset:
            pickle.dump(val_data, f)
        else:
            pickle.dump(test_data, f)
        f.close()

