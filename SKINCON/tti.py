

#!/usr/bin/env python
# coding: utf-8
import os
from pyexpat import model
import sys
import torch
import pickle
import random
from scipy.stats import entropy
from scipy.special import softmax, kl_div
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from SKINCON.inference import *
from SKINCON.config import N_CLASSES, N_ATTRIBUTES

import matplotlib.pyplot as plt

from tqdm import tqdm

from itertools import permutations

from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score

# calculate score for 'ectp' and 'eudtp' criteria
def calculate_etp_score(criterion, b_class_logits, intervened_idx, ptl_5, ptl_95, model2, b_attr_outputs, b_attr_outputs_sigmoid, b_class_labels, 
                                use_relu, use_sigmoid, connect_CY=False):

    b_attr_0 = np.array(b_attr_outputs[:])
    b_attr_1 = np.array(b_attr_outputs[:])
    b_attr_point5 = np.array(b_attr_outputs[:])
    
    prob = []
    attr_replace_idx = []
    all_attr_ids = []
    
    for img_id in range(len(b_class_labels)):
        concept_idx = intervened_idx[img_id]
        replace_idx = [concept_idx]
        all_attr_ids.extend(replace_idx)
        attr_replace_idx.extend(np.array(replace_idx) + img_id * args.n_attributes)
        prob.append(b_attr_outputs_sigmoid[img_id * args.n_attributes + intervened_idx[img_id]])
    
    prob = np.array(prob)

    if use_relu or not use_sigmoid:
        b_attr_0[attr_replace_idx] = ptl_5[intervened_idx[0]]
        b_attr_1[attr_replace_idx] = ptl_95[intervened_idx[0]]
    else:
        b_attr_0[attr_replace_idx] = 0
        b_attr_1[attr_replace_idx] = 1
        b_attr_point5[attr_replace_idx] = 0.5

    # stage 2: calcaulte the label prediction error
    model2.eval()

    b_attr_0 = b_attr_0.reshape(-1, args.n_attributes)
    b_attr_1 = b_attr_1.reshape(-1, args.n_attributes)

    stage2_inputs_0 = torch.from_numpy(np.array(b_attr_0)).cuda()
    stage2_inputs_1 = torch.from_numpy(np.array(b_attr_1)).cuda()
    
    class_outputs_0 = model2(stage2_inputs_0).cpu().detach().numpy()
    class_outputs_1 = model2(stage2_inputs_1).cpu().detach().numpy()

    b_attr_outputs_sigmoid = np.array(b_attr_outputs_sigmoid)

    softmax_outputs_0 = softmax(class_outputs_0, axis=1)
    softmax_outputs_1 = softmax(class_outputs_1, axis=1)

    original_softmax = softmax(np.array(b_class_logits), axis=1)
    original_score = entropy(original_softmax, axis=1)

    if criterion == 'eudtp':
        score_0 = entropy(softmax_outputs_0, axis=1)
        score_1 = entropy(softmax_outputs_1, axis=1)
    elif criterion == 'ectp':
        score_0 = entropy(softmax_outputs_0 ,original_softmax, axis=1)
        score_1 = entropy(softmax_outputs_1 ,original_softmax, axis=1)
 
    score = score_0 * (1 - prob) + score_1 * prob

    if criterion == 'eudtp':
        score = (original_score - score)

    score *= -1

    return score

# calcaulte score for other criteria
def calculate_c_score(criterion, attr_preds_sigmoid, attr_labels, model2):
    if criterion == 'lcp':
        score = np.abs(np.array(attr_preds_sigmoid) - np.array(attr_labels))**2
    elif criterion == 'ucp':
        score = 1/np.abs(np.array(attr_preds_sigmoid) - 0.5)**2
    elif criterion == 'cctp':
        if 'weight' in model2.state_dict():
            layer_name = model2.weight
        else:
            layer_name = model2.linear.weight
        score = torch.sum(torch.abs(layer_name), axis=0)
        score = score.cpu().detach().numpy()
        if args.batch_intervention: # tile up weights in batch intervention
            n_samples = int(len(attr_labels) / args.n_attributes)
            score = np.tile(score, n_samples)
        score = score * attr_preds_sigmoid
    return score  

def calculate_intervention_order(criterion, replace_val, preds_by_attr, ptl_5, ptl_95, model2, b_attr_binary_outputs, b_class_labels, b_class_logits,
                                b_attr_outputs, b_attr_outputs_sigmoid, b_attr_outputs2, b_attr_labels,
                                n_replace, use_relu, use_sigmoid, batch_intervention, n_trials=1, connect_CY=False, class_label='binary'):

    all_intervention_order_list = [] # n_trials * N_TEST * n_groups (order of intervention)
    
    for i in range(n_trials):
        if criterion in ['ectp', 'eudtp']:
            n_orders = args.n_attributes
            score_list = np.zeros((len(b_class_labels), n_orders)) # intervene on increasing order of score
            for j in range(n_orders):
                score = calculate_etp_score(criterion, b_class_logits, np.full(len(b_class_labels), j), ptl_5, ptl_95, model2, b_attr_outputs, b_attr_outputs_sigmoid, b_class_labels, use_relu, use_sigmoid, connect_CY=False)
                score_list[:, j] = score
            if batch_intervention:
                score_list = score_list.flatten()
                intervention_order_list = np.argsort(score_list)
            else:
                intervention_order_list = np.argsort(score_list, axis=1)
            all_intervention_order_list.append(intervention_order_list)
        else: # 'rand', 'ucp', 'lcp', 'cctp'
            if batch_intervention:
                if criterion == 'rand':
                    whole_size = args.n_attributes * len(b_class_labels)
                    intervention_order_list = list(random.sample(list(range(whole_size)), whole_size))
                else:
                    score_list = calculate_c_score(criterion, b_attr_outputs_sigmoid, b_attr_labels, model2)
                    intervention_order_list = np.argsort(score_list)[::-1]
            else: # single intervention
                intervention_order_list = [] # N_TEST * n_groups (order of intervention)
                for img_id in range(len(b_class_labels)):
                    attr_preds = b_attr_outputs[img_id * args.n_attributes: (img_id + 1) * args.n_attributes]
                    attr_preds_sigmoid = b_attr_outputs_sigmoid[img_id * args.n_attributes: (img_id + 1) * args.n_attributes]
                    attr_preds2 = b_attr_outputs2[img_id * args.n_attributes: (img_id + 1) * args.n_attributes]
                    attr_labels = b_attr_labels[img_id * args.n_attributes: (img_id + 1) * args.n_attributes]
                    n_orders = args.n_attributes
                    if criterion == 'rand':
                        intervention_order = list(random.sample(list(range(n_orders)), n_orders))
                    else:
                        score = calculate_c_score(criterion, attr_preds_sigmoid, attr_labels, model2)
                        intervention_order = np.argsort(score)[::-1]
                    intervention_order_list.append(intervention_order)
            all_intervention_order_list.append(intervention_order_list)
    return all_intervention_order_list



def simulate_intervention(intervention_order, criterion, replace_val, preds_by_attr, ptl_5, ptl_95, model2, b_attr_binary_outputs, b_class_labels, b_class_logits,
                                b_attr_outputs, b_attr_outputs_sigmoid, b_attr_outputs2, b_attr_labels,
                                b_attr_intervened, class_label,
                                n_replace, use_relu, use_sigmoid, batch_intervention, trial, incorrect_idx_list,
                                n_trials=1, inference_mode='soft', mc_samples=5, connect_CY=False):

    if n_replace == 1 or n_replace == 0:
        b_attr_new = np.array(b_attr_outputs[:])
    else:
        b_attr_new = b_attr_intervened

    attr_replace_idx = []
    all_attr_ids = []

    if n_replace > 0:
        if batch_intervention:
            start_idx = (n_replace - 1) * len(b_class_labels)
            end_idx = n_replace * len(b_class_labels)
            concept_idx = intervention_order[trial][start_idx:end_idx]
            replace_idx = concept_idx
            all_attr_ids.extend(np.array(replace_idx) % args.n_attributes)
            attr_replace_idx.extend(np.array(replace_idx))
        else:
            for img_id in range(len(b_class_labels)):
                concept_idx = intervention_order[trial][img_id][n_replace-1]
                replace_idx = [concept_idx]
                all_attr_ids.extend(replace_idx)
                attr_replace_idx.extend(np.array(replace_idx) + img_id * args.n_attributes)

        b_attr_new[attr_replace_idx] = np.array(b_attr_labels)[attr_replace_idx]

        if use_relu or not use_sigmoid:  # replace with percentile values
            binary_vals = b_attr_new[attr_replace_idx]
            for j, replace_idx in enumerate(attr_replace_idx):
                attr_idx = replace_idx % args.n_attributes
                b_attr_new[replace_idx] = (1 - binary_vals[j]) * ptl_5[attr_idx] + binary_vals[j] * ptl_95[attr_idx]
    
    # stage 2: calcaulte the label prediction error
    K = [1]
    model2.eval()

    b_attr_new = b_attr_new.reshape(-1, args.n_attributes)
    stage2_inputs = torch.from_numpy(np.array(b_attr_new)).cuda()
    if inference_mode == 'soft' or inference_mode == 'hard':
        if inference_mode == 'hard':
            stage2_inputs = stage2_inputs >= (torch.ones_like(stage2_inputs) * 0.5)
            stage2_inputs = stage2_inputs.float()
        if connect_CY:  # class_outputs is currently contributed by C --> Y
            new_cy_outputs = model2(stage2_inputs)
            old_stage2_inputs = torch.from_numpy(np.array(b_attr_outputs).reshape(-1, args.n_attributes)).cuda()
            old_cy_outputs = model2(old_stage2_inputs)
            class_outputs = torch.from_numpy(b_class_logits).cuda() + (new_cy_outputs - old_cy_outputs)
        else:
            class_outputs = model2(stage2_inputs)
    elif inference_mode == 'samp':
        class_outputs_all = []
        for _ in range(mc_samples):
            rand_num = torch.rand(*stage2_inputs.size())
            sampled_stage2_inputs = rand_num.cuda() < stage2_inputs
            sampled_stage2_inputs = sampled_stage2_inputs.float()
            if connect_CY:  # class_outputs is currently contributed by C --> Y
                new_cy_outputs = model2(sampled_stage2_inputs)
                old_stage2_inputs = torch.from_numpy(np.array(b_attr_outputs).reshape(-1, args.n_attributes)).cuda()
                old_cy_outputs = model2(old_stage2_inputs)
                _class_outputs = torch.from_numpy(b_class_logits).cuda() + (new_cy_outputs - old_cy_outputs)
            else:
                _class_outputs = model2(sampled_stage2_inputs)
            class_outputs_all.append(_class_outputs)
        class_outputs = torch.mean(torch.stack(class_outputs_all, axis=0), axis=0)
    

    _, preds = class_outputs.topk(1, 1, True, True)
    b_class_outputs_new = preds.data.cpu().numpy().squeeze()

    class_balanced_acc = balanced_accuracy_score(b_class_labels, b_class_outputs_new)

    return class_balanced_acc, b_attr_new.flatten()


def parse_arguments(parser=None):
    if parser is None: parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-log_dir', default='.', help='where results are stored')
    parser.add_argument('-model_dirs', nargs='+', help='where the trained model is saved')
    parser.add_argument('-model_dirs2', nargs='+', default=None, help='where another trained model is saved (for bottleneck only)')
    parser.add_argument('-eval_data', default='test', help='Type of data (val/ test) to be used')
    parser.add_argument('-batch_size', type=int, default=16)
    parser.add_argument('-use_attr', help='whether to use attributes (FOR COTRAINING ARCHITECTURE ONLY)', action='store_true')
    parser.add_argument('-no_img', help='if included, only use attributes (and not raw imgs) for class prediction', action='store_true')
    parser.add_argument('-bottleneck', help='whether to predict attributes before class labels', action='store_true')
    parser.add_argument('-no_background', help='whether to test on images with background removed', action='store_true')
    parser.add_argument('-n_class_attr', type=int, default=2, help='whether attr prediction is a binary or triary classification')
    parser.add_argument('-data_dir', default='', help='directory to the data used for evaluation')
    parser.add_argument('-n_attributes', type=int, default=22, help='whether to apply bottlenecks to only a few attributes')
    parser.add_argument('-image_dir', default='images', help='test image folder to run inference on')
    parser.add_argument('-attribute_group', default=None, help='file listing the (trained) model directory for each attribute group')
    parser.add_argument('-feature_group_results', help='whether to print out performance of individual atttributes', action='store_true')
    parser.add_argument('-use_relu', help='Whether to include relu activation before using attributes to predict Y. For end2end & bottleneck model', action='store_true')
    parser.add_argument('-use_sigmoid', help='Whether to include sigmoid activation before using attributes to predict Y. For end2end & bottleneck model', action='store_true')
    parser.add_argument('-class_level', help='Whether to correct with class- (if set) or instance- (if not set) level values', action='store_true')
    parser.add_argument('-use_invisible', help='Whether to include attribute visibility information', action='store_true')
    parser.add_argument('-criterion', help='Intervention criterion', default='rand')
    parser.add_argument('-level', default='i+s', help='intervention level')
    parser.add_argument('-inference_mode', default='soft', help='mode of inference')
    parser.add_argument('-mc_samples', default=5, help='MC samples for prediction')
    parser.add_argument('-n_trials', help='Number of trials to run, when mode is random', type=int, default=1)
    parser.add_argument('-connect_CY', help='Whether to use concepts as auxiliary features (in multitasking) to predict Y', action='store_true')
    parser.add_argument('-no_intervention_when_invisible', action='store_true', help='No intervention when the related part is not visible')
    parser.add_argument('-class_label', type=str, default='binary', help='which class label to use')
    args = parser.parse_args()
    return args

def run(args):
    print(args)
    
    # stage 1: calculate \hat{c} & the percentiles
    _, _, b_class_labels, b_topk_class_outputs, b_class_logits, b_attr_labels, b_attr_outputs, b_attr_outputs_sigmoid, \
        b_wrong_idx, b_attr_outputs2, incorrect_idx_list, img_path_list, yerr_by_cerr,\
        b_class_f1, b_class_balanced_acc = eval(args)

    b_attr_binary_outputs = np.rint(b_attr_outputs_sigmoid).astype(int)

    preds_by_attr, ptl_5, ptl_95 = dict(), dict(), dict()
    for i, val in enumerate(b_attr_outputs):
        attr_idx = i % args.n_attributes
        if attr_idx in preds_by_attr:
            preds_by_attr[attr_idx].append(val)
        else:
            preds_by_attr[attr_idx] = [val]

    for attr_idx in range(args.n_attributes):
        preds = preds_by_attr[attr_idx]
        ptl_5[attr_idx] = np.percentile(preds, 5)
        ptl_95[attr_idx] = np.percentile(preds, 95)


    print(args.model_dir)

    N_TRIALS = args.n_trials
    assert args.criterion in ['rand', 'ucp', 'lcp', 'cctp', 'ectp', 'eudtp']
    assert args.level in ['i+s', 'i+b']
                        
    REPLACE_VAL = 'class_level'

    batch_intervention = args.level[-1] == 'b' # Batch / single intervention

    # stage 2: intervene on the concepts
    model = torch.load(args.model_dir)
    if args.model_dir2:
        if 'rf' in args.model_dir2:
            model2 = load(args.model_dir2)
        else:
            model2 = torch.load(args.model_dir2)
    else:  # end2end, split model into 2
        all_mods = list(model.modules())
        model2 = all_mods[-1]  # last fully connected layer



    outputs_sigmoid = b_attr_outputs_sigmoid

    intervention_order = calculate_intervention_order(args.criterion, REPLACE_VAL,
                                          preds_by_attr, ptl_5, ptl_95,
                                          model2,
                                          b_attr_binary_outputs,
                                          b_class_labels,
                                          b_class_logits,
                                          b_attr_outputs,
                                          outputs_sigmoid,
                                          b_attr_outputs2,
                                          b_attr_labels,
                                          1, args.use_relu,
                                          args.use_sigmoid,
                                          batch_intervention,
                                          n_trials=N_TRIALS,
                                          connect_CY=args.connect_CY,
                                          class_label=args.class_label)


    results = []
    all_balanced_err_list = []
    b_attr_intervened = b_attr_outputs

    intervention_num = args.n_attributes + 1

    for trial in range(N_TRIALS):
        balanced_err_list = []
        for n_replace in list(range(intervention_num)):
            class_balanced_acc, b_attr_intervened = simulate_intervention(intervention_order,
                                            args.criterion, REPLACE_VAL,
                                            preds_by_attr, ptl_5, ptl_95,
                                            model2,
                                            b_attr_binary_outputs,
                                            b_class_labels,
                                            b_class_logits,
                                            b_attr_outputs,
                                            b_attr_outputs_sigmoid,
                                            b_attr_outputs2,
                                            b_attr_labels,
                                            b_attr_intervened,
                                            args.class_label,
                                            n_replace, args.use_relu,
                                            args.use_sigmoid,
                                            batch_intervention,
                                            trial,
                                            incorrect_idx_list,
                                            n_trials=N_TRIALS,
                                            inference_mode=args.inference_mode,
                                            connect_CY=args.connect_CY)            
        
            results.append([n_replace, class_balanced_acc])
            balanced_err_list.append(1 - class_balanced_acc)
            
        all_balanced_err_list.append(balanced_err_list)

    balanced_err_mean_list = np.array(all_balanced_err_list).mean(axis=0)

    return np.array([float(err) for err in balanced_err_mean_list])

if __name__ == '__main__':
    torch.backends.cudnn.benchmark=True
    args = parse_arguments()
    print(args)

    args.group_intervention = args.level[0] == 'g' # Group-wise / Individual intervention
    args.batch_intervention = args.level[-1] == 'b' # Batch / single intervention

    assert(not args.group_intervention)

    err_list = []

    for i, model_dir in enumerate(args.model_dirs):
        print('----------')
        args.model_dir = model_dir
        args.model_dir2 = args.model_dirs2[i] if args.model_dirs2 else None
        err_values = run(args)
        err_list.append(err_values)

    err_mean = np.array(err_list).mean(axis=0)
    err_std = np.array(err_list).std(axis=0)

    print("------ Result ------")
    print("Criterion: ",args.criterion)
    print("Level: ", args.level)
    print("Conceptualization: ", args.inference_mode)

    print('err mean')
    print(err_mean.tolist())
    print("err_std")
    print(err_std.tolist())