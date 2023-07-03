"""
Create variants of the initial CUB dataset
"""
import os
import sys
import copy
import torch
import random
import pickle
import argparse
import numpy as np
from PIL import Image
from shutil import copyfile
import torchvision.transforms as transforms
from collections import defaultdict as ddict

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from SKINCON.config import N_ATTRIBUTES, N_CLASSES

from tqdm import tqdm
from sklearn.manifold import TSNE

def remove_sparse_concepts(min_count, out_dir, modify_data_dir='', keep_instance_data=False):
    """
    Use train.pkl to only keep those present at more than min_count images
    Transform data in modify_data_dir file and save the new dataset to out_dir
    """
    data = []
    for dataset in ['train', 'val', 'test']:
        full_path = os.path.join(os.getcwd(), modify_data_dir, dataset + '.pkl' )
        data.extend(pickle.load(open(full_path, 'rb')))
    
    attr_count = np.zeros(len(data[0]['attribute_label']))
    for d in data:
        for attr_idx, a in enumerate(d['attribute_label']):
            attr_count[attr_idx] += a

    mask = np.where(attr_count >= min_count) #select attributes that are present in at least [min_count] images

    collapse_fn = lambda d: list(np.array(d['attribute_label'])[mask])
    create_new_dataset(out_dir, 'attribute_label', collapse_fn, data_dir=modify_data_dir)

def create_logits_data(model_path, out_dir, data_dir='', use_relu=False, use_sigmoid=False):
    """
    Replace attribute labels in data_dir with the logits output by the model from model_path and save the new data to out_dir
    """
    model = torch.load(model_path)
    get_logits_train = lambda d: inference(d['img_path'], model, use_relu, use_sigmoid, is_train=True)
    get_logits_test = lambda d: inference(d['img_path'], model, use_relu, use_sigmoid, is_train=False)
    create_new_dataset(out_dir, 'attribute_label', get_logits_train, datasets=['train'], data_dir=data_dir)
    create_new_dataset(out_dir, 'attribute_label', get_logits_train, datasets=['val', 'test'], data_dir=data_dir)

def inference(img_path, model, use_relu, use_sigmoid, is_train, resol=299, layer_idx=None):
    """
    For a single image stored in img_path, run inference using model and return A\hat (if layer_idx is None) or values extracted from layer layer_idx 
    """
    model.eval()
    # see utils.py
    if is_train:
        transform = transforms.Compose([
            transforms.ColorJitter(brightness=32 / 255, saturation=(0.5, 1.5)),
            transforms.RandomResizedCrop(resol),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),  # implicitly divides by 255
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[2, 2, 2])
        ])
    else:
        transform = transforms.Compose([
            transforms.CenterCrop(resol),
            transforms.ToTensor(),  # implicitly divides by 255
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[2, 2, 2])
        ])

    try:
        idx = img_path.split('/').index('data')
        img_path = '/'.join(img_path.split('/')[idx:])
        img = Image.open(img_path).convert('RGB')
    except:
        img_path_split = img_path.split('/')
        split = 'train' if self.is_train else 'test'
        img_path = '/'.join(img_path_split[:2] + [split] + img_path_split[2:])
        img = Image.open(img_path).convert('RGB')
    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)
    input_var = torch.autograd.Variable(img_tensor).cuda()
    if layer_idx is not None:
        all_mods = list(model.modules())
        cropped_model = torch.nn.Sequential(*list(model.children())[:layer_idx])  # nn.ModuleList(all_mods[:layer_idx])
        print(type(input_var), input_var.shape, input_var)
        return cropped_model(input_var)

    outputs = model(input_var)
    if use_relu:
        attr_outputs = [torch.nn.ReLU()(o) for o in outputs]
    elif use_sigmoid:
        attr_outputs = [torch.nn.Sigmoid()(o) for o in outputs]
    else:
        attr_outputs = outputs

    attr_outputs = torch.cat([o.unsqueeze(1) for o in attr_outputs], dim=1).squeeze()
    return list(attr_outputs.data.cpu().numpy())


def inference_no_grad(img_path, model, use_relu, use_sigmoid, is_train, resol=299, layer_idx=None):
    """
    Extract activation from layer_idx of model for input from img_path (for linear probe)
    """
    with torch.no_grad():
        attr_outputs = inference(img_path, model, use_relu, use_sigmoid, is_train, resol, layer_idx)
    #return [list(o.cpu().numpy().squeeze())[0] for o in attr_outputs]
    return [o.cpu().numpy().squeeze()[()] for o in attr_outputs]


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
                assert (type(old_value) == type(new_value))
            new_d[field_change] = new_value
            new_data.append(new_d)
        f = open(os.path.join(out_dir, dataset + '.pkl'), 'wb')
        pickle.dump(new_data, f)
        f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str,
                        choices=['ExtractConcepts', 'ExtractProbeRepresentations', 'DataEfficiencySplits', 'ChangeAdversarialDataDir', 'Concept', 'MV'],
                        help='Name of experiment to run.')
    parser.add_argument('--model_path', type=str, help='Path of model')
    parser.add_argument('--out_dir', type=str, help='Output directory')
    parser.add_argument('--data_dir', type=str, help='Data directory')
    parser.add_argument('--adv_data_dir', type=str, help='Adversarial data directory')
    parser.add_argument('--train_splits', type=str, nargs='+', help='Train splits to use')
    parser.add_argument('--use_relu', action='store_true', help='Use Relu')
    parser.add_argument('--use_sigmoid', action='store_true', help='Use Sigmoid')
    parser.add_argument('--layer_idx', type=int, default=None, help='Layer id to extract probe representations')
    parser.add_argument('--n_samples', type=int, help='Number of samples for data efficiency split')
    parser.add_argument('--splits_dir', type=str, help='Data dir of splits')
    parser.add_argument('--modify_data_dir', type=str, help="Data dir to be modified")
    parser.add_argument('-class_label', type=str, default='binary', help='which class label to use')
    args = parser.parse_args()

    if args.exp == 'Concept':
        out_dir = os.path.join(os.getcwd(), args.data_dir)
        modify_data_dir = os.path.join(os.getcwd(), args.modify_data_dir)
        remove_sparse_concepts(50, out_dir, modify_data_dir=modify_data_dir)
    elif args.exp == 'ExtractConcepts':
        create_logits_data(args.model_path, args.out_dir, args.data_dir, args.use_relu, args.use_sigmoid)