import os
import sys
import pickle
import random
import argparse

import numpy as np

import copy
import torch

from sklearn.model_selection import train_test_split

import sys
sys.path.insert(0, '../')

def gen_data(args):
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    # sample \alpha
    alpha = np.random.normal(loc=args.alpha_mean, scale=args.alpha_var, size=args.n_attributes)

    alpha = np.clip(alpha, 0, 1)

    beta = np.random.uniform(size=(args.n_groups, args.n_attributes))

    group_concepts = (beta >= alpha).astype(int)

    n_classes_per_group = args.n_classes // args.n_groups

    class_concepts = np.empty((args.n_classes, args.n_attributes))

    for group in range(args.n_groups):
        group_concept_value = group_concepts[group]
        change_indexes = np.random.choice(args.n_attributes, n_classes_per_group, replace=False)
        for i, index in enumerate(change_indexes):
            class_concept = group_concept_value.copy()
            class_concept[index] = 1 - class_concept[index]
            class_index = group * n_classes_per_group + i
            class_concepts[class_index, :] = class_concept
    
    class_concepts = class_concepts.astype(int)
    print(class_concepts)

    w_x = np.random.normal(scale=args.w_var, size=(args.input_dim, args.n_attributes))

    np.save(os.path.join(args.out_dir, 'wx_save'), w_x)

    new_data = []

    labels_list = []

    for y in range(args.n_classes):
        for i in range(args.n_samples_per_class):
            z = np.random.normal(scale=args.z_var, size=args.input_dim)
            c = class_concepts[y, :]
            x = w_x @ c + z
            x = np.float32(x)
            id = y*args.n_samples_per_class + i
            data = {'id': id, 'input': x, 'z': z, 'attribute_label': c, 'label': y}
            new_data.append(data)
            labels_list.append(y)
    new_data = np.array(new_data)
    n_total_samples = args.n_samples_per_class * args.n_classes
    trainval_id, test_id, trainval_label, test_label = train_test_split(np.arange(n_total_samples), labels_list, stratify=labels_list, test_size=args.test_ratio, random_state=42)
    
    val_ratio = args.val_ratio/(1 - args.test_ratio)
    train_id, val_id, train_label, val_label = train_test_split(trainval_id, trainval_label, stratify=trainval_label, test_size=val_ratio, random_state=42)
    
    test_data = new_data[test_id]
    val_data = new_data[val_id]
    train_data = new_data[train_id]
    
    f_test = open(os.path.join(args.out_dir, 'test.pkl'), 'wb')
    f_val = open(os.path.join(args.out_dir, 'val.pkl'), 'wb')
    f_train = open(os.path.join(args.out_dir, 'train.pkl'), 'wb')

    pickle.dump(test_data, f_test)
    pickle.dump(val_data, f_val)
    pickle.dump(train_data, f_train)

    f_test.close()
    f_val.close()
    f_train.close()

def inference(d, model, use_relu, use_sigmoid, is_train, layer_idx=None):
    """
    For a single image stored in img_path, run inference using model and return A\hat (if layer_idx is None) or values extracted from layer layer_idx 
    """
    model.eval()

    inputs = torch.from_numpy(d['input']).unsqueeze(0)
    input_var = torch.autograd.Variable(inputs).cuda()

    outputs = model(input_var)
    new_outputs = []

    for c in range(args.n_attributes):
        new_outputs.append(outputs[:, c].unsqueeze(1))

    outputs = new_outputs
    if use_relu:
        attr_outputs = [torch.nn.ReLU()(o) for o in outputs]
    elif use_sigmoid:
        attr_outputs = [torch.nn.Sigmoid()(o) for o in outputs]
    else:
        attr_outputs = outputs

    attr_outputs = torch.cat([o.unsqueeze(1) for o in attr_outputs], dim=1).squeeze()
    return list(attr_outputs.data.cpu().numpy())

def create_new_dataset(out_dir, field_change, compute_fn, datasets=['train', 'val', 'test'], data_dir=''):
    """
    Generic function that given datasets stored in data_dir, modify/ add one field of the metadata in each dataset based on compute_fn
                          and save the new datasets to out_dir
    compute_fn should take in a metadata object (that includes 'img_path', 'class_label', 'attribute_label', etc.)
                          and return the updated value for field_change
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for dataset in datasets:
        path = os.path.join(data_dir, dataset + '.pkl')
        if not os.path.exists(path):
            continue
        data = pickle.load(open(path, 'rb'))
        new_data = []
        for d in data:
            new_d = copy.deepcopy(d)
            new_value = compute_fn(d)
            if field_change in d:
                old_value = d[field_change]
                new_value = np.array(new_value)
                assert (type(old_value) == type(new_value))
            new_d[field_change] = new_value
            new_data.append(new_d)
        f = open(os.path.join(out_dir, dataset + '.pkl'), 'wb')
        pickle.dump(new_data, f)
        f.close()

def create_logits_data(model_path, out_dir, data_dir='', use_relu=False, use_sigmoid=False):
    """
    Replace attribute labels in data_dir with the logits output by the model from model_path and save the new data to out_dir
    """
    model = torch.load(model_path)
    get_logits_train = lambda d: inference(d, model, use_relu, use_sigmoid, is_train=True)
    get_logits_test = lambda d: inference(d, model, use_relu, use_sigmoid, is_train=False)
    create_new_dataset(out_dir, 'attribute_label', get_logits_train, datasets=['train'], data_dir=data_dir)
    create_new_dataset(out_dir, 'attribute_label', get_logits_train, datasets=['val', 'test'], data_dir=data_dir)

def create_hidden_data(out_dir, data_dir, hidden_ratio, n_attributes, datasets=['train', 'val', 'test']):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    revealed_concept_num = int((1 - hidden_ratio) * n_attributes)
    revealed_concepts_indexes = np.random.choice(args.n_attributes, revealed_concept_num, replace=False)
    for dataset in datasets:
        path = os.path.join(data_dir, dataset + '.pkl')
        if not os.path.exists(path):
            continue
        data = pickle.load(open(path, 'rb'))
        new_data = []
        for d in data:
            new_d = copy.deepcopy(d)
            attribute_label = d['attribute_label']
            new_d['attribute_label'] = attribute_label[revealed_concepts_indexes]
            assert (type(d['attribute_label']) == type(new_d['attribute_label']))
            new_data.append(new_d)
        f = open(os.path.join(out_dir, dataset + '.pkl'), 'wb')
        pickle.dump(new_data, f)
        f.close()

def create_diversity_data(out_dir, data_dir, diversity_ratio, n_attributes, datasets=['train', 'val', 'test']):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    w_x = np.load(os.path.join(data_dir, 'wx_save.npy'))
    for dataset in datasets:
        path = os.path.join(data_dir, dataset + '.pkl')
        if not os.path.exists(path):
            continue
        data = pickle.load(open(path, 'rb'))
        new_data = []
        for d in data:
            new_d = copy.deepcopy(d)
            attribute_label = d['attribute_label']
            reverted_label = 1 - attribute_label
            reverted_mask = np.random.uniform(size=n_attributes) < np.ones(n_attributes) * diversity_ratio
            new_d['attribute_label'] = np.where(reverted_mask == 1, reverted_label, attribute_label)
            assert (type(d['attribute_label']) == type(new_d['attribute_label']))
            new_x = w_x @ new_d['attribute_label'] + d['z']
            new_d['input'] = np.float32(new_x)
            assert (type(d['input']) == type(new_d['input']))
            new_data.append(new_d)
        f = open(os.path.join(out_dir, dataset + '.pkl'), 'wb')
        pickle.dump(new_data, f)
        f.close()

def create_sparsity_data(args, datasets=['train', 'val', 'test']):
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    alpha = np.random.normal(loc=args.alpha_mean, scale=args.alpha_var, size=args.n_attributes)

    alpha = np.clip(alpha, 0, 1)

    beta = np.random.uniform(size=(args.n_groups, args.n_attributes))

    group_concepts = (beta >= alpha).astype(int)

    n_classes_per_group = args.n_classes // args.n_groups

    class_concepts = np.empty((args.n_classes, args.n_attributes))

    for group in range(args.n_groups):
        group_concept_value = group_concepts[group]
        class_concepts[group * n_classes_per_group, :] = group_concept_value
        change_indexes = np.random.choice(args.n_attributes, n_classes_per_group, replace=False)
        prev_concept = group_concept_value.copy()
        for i, index in enumerate(change_indexes):
            class_concept = group_concept_value.copy()
            class_concept[index] = 1 - class_concept[index]
            class_index = group * n_classes_per_group + i
            class_concepts[class_index, :] = class_concept

    w_x = np.load(os.path.join(args.data_dir, 'wx_save.npy'))
    for dataset in datasets:
        path = os.path.join(args.data_dir, dataset + '.pkl')
        if not os.path.exists(path):
            continue
        data = pickle.load(open(path, 'rb'))
        new_data = []
        for d in data:
            new_d = copy.deepcopy(d)
            y = d['label']
            c = class_concepts[y, :]
            z = d['z']
            x = w_x@c + z
            x = np.float32(x)
            new_d['attribute_label'] = c 
            new_d['input'] = x
            assert (type(d['attribute_label']) == type(new_d['attribute_label']))
            assert (type(d['input']) == type(new_d['input']))
            new_data.append(new_d)
        f = open(os.path.join(args.out_dir, dataset + '.pkl'), 'wb')
        pickle.dump(new_data, f)
        f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-exp', type=str,
                        choices=['GenData', 'Hidden', 'Diversity', 'ExtractConcepts', 'Sparsity', 'Similarity'],
                        help='Name of experiment to run.', default='GenData')
    parser.add_argument('-out_dir', type=str, help='Output directory')
    parser.add_argument('-n_samples_per_class', type=int, help='Number of samples to generate', default=100)
    parser.add_argument('-data_dir', type=str, help="Data dir to be modified")
    parser.add_argument('-input_dim', type=int, help='dimension of x', default=100)
    parser.add_argument('-n_attributes', type=int, help='dimension of c', default=100)
    parser.add_argument('-n_classes', type=int, help='dimension of y', default=100)
    parser.add_argument('-z_var', type=float, help='variance of z', default=2.0)
    parser.add_argument('-alpha_var', type=float, help='variance of alpha', default=0.1)
    parser.add_argument('-alpha_mean', type=float, help='mean of alpha', default=0.8)
    parser.add_argument('-w_var', type=float, help='variance of matrix W', default=0.1)
    parser.add_argument('-diversity_ratio', type=float, help='concept diversity factor')
    parser.add_argument('-hidden_ratio', type=float, help='number of hidden concepts')
    parser.add_argument('-test_ratio', type=float, help='ratio of test samples', default=0.15)
    parser.add_argument('-val_ratio', type=float, help='ratio of validation samples', default=0.15)
    parser.add_argument('-datasets', default=['train', 'test'], help='datasets to generate')
    parser.add_argument('-model_path', type=str, help='Path of model')
    parser.add_argument('--use_relu', action='store_true', help='Use Relu')
    parser.add_argument('--use_sigmoid', action='store_true', help='Use Sigmoid')
    parser.add_argument('-n_groups', type=int, help='number of similar groups', default=50)
    args = parser.parse_args()

    if args.exp == 'GenData':
        gen_data(args)
    elif args.exp == 'ExtractConcepts':
        create_logits_data(args.model_path, args.out_dir, args.data_dir, args.use_relu, args.use_sigmoid)
    elif args.exp == 'Hidden':
        create_hidden_data(args.out_dir, args.data_dir, args.hidden_ratio, args.n_attributes)
    elif args.exp == 'Diversity':
        create_diversity_data(args.out_dir, args.data_dir, args.diversity_ratio, args.n_attributes)
    elif args.exp == 'Sparsity':
        create_sparsity_data(args)
    elif args.exp == 'Similarity':
        create_sparsity_data(args)
    