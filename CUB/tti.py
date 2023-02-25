

#!/usr/bin/env python
# coding: utf-8
import os
from pyexpat import model
import sys
import torch
import pickle
import random
from scipy.stats import entropy
from scipy.special import softmax
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from CUB.inference import *
from CUB.config import N_CLASSES, N_ATTRIBUTES, GROUP_DICT
from CUB.utils import get_class_attribute_names

import matplotlib.pyplot as plt

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

def calculate_intervention_order(criterion, replace_val, preds_by_attr, ptl_5, ptl_95, model2, attr_group_dict, b_attr_binary_outputs, b_class_labels, b_class_logits,
                                b_attr_outputs, b_attr_outputs_sigmoid, b_attr_outputs2, b_attr_labels,
                                instance_attr_labels, uncertainty_attr_labels, use_not_visible, min_uncertainty,
                                n_replace, use_relu, use_sigmoid, group_intervention, batch_intervention, n_trials=1, connect_CY=False):

    all_intervention_order_list = [] # n_trials * N_TEST * n_groups (order of intervention)
    
    for i in range(n_trials):
        if criterion in ['ectp', 'eudtp']:
            if group_intervention:
                n_orders = args.n_groups
            else: 
                n_orders = args.n_attributes
            score_list = np.zeros((len(b_class_labels), n_orders)) # intervene on increasing order of score
            for j in range(n_orders):
                if group_intervention:
                    score = np.zeros(len(b_class_labels))
                    for attr_idx in GROUP_DICT[j]:
                        attr_score = calculate_etp_score(criterion, b_class_logits, np.full(len(b_class_labels), attr_idx), ptl_5, ptl_95, model2, b_attr_outputs, b_attr_outputs_sigmoid, b_class_labels, use_relu, use_sigmoid, connect_CY=False)
                        score += attr_score
                    score = score/len(GROUP_DICT[j])
                else:
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
                    if group_intervention:
                        whole_size = args.n_groups * len(b_class_labels)
                    else:
                        whole_size = args.n_attributes * len(b_class_labels)
                    intervention_order_list = list(random.sample(list(range(whole_size)), whole_size))
                else:
                    score_list = calculate_c_score(criterion, b_attr_outputs_sigmoid, b_attr_labels, model2)
                    if group_intervention:
                        score = []
                        for img_id in range(len(b_class_labels)):
                            for group_idx in range(args.n_groups):
                                group_score = 0
                                for attr_idx in GROUP_DICT[group_idx]:
                                    group_score += score_list[img_id * args.n_attributes + attr_idx]
                                score.append(group_score/len(GROUP_DICT[group_idx]))
                    else:
                        score = score_list
                    intervention_order_list = np.argsort(score)[::-1]
            else: # single intervention
                intervention_order_list = [] # N_TEST * n_groups (order of intervention)
                for img_id in range(len(b_class_labels)):
                    attr_preds = b_attr_outputs[img_id * args.n_attributes: (img_id + 1) * args.n_attributes]
                    attr_preds_sigmoid = b_attr_outputs_sigmoid[img_id * args.n_attributes: (img_id + 1) * args.n_attributes]
                    attr_preds2 = b_attr_outputs2[img_id * args.n_attributes: (img_id + 1) * args.n_attributes]
                    attr_labels = b_attr_labels[img_id * args.n_attributes: (img_id + 1) * args.n_attributes]
                    if group_intervention:
                        n_orders = args.n_groups
                    else:
                        n_orders = args.n_attributes
                    if criterion == 'rand':
                        intervention_order = list(random.sample(list(range(n_orders)), n_orders))
                    else:
                        score_list = calculate_c_score(criterion, attr_preds_sigmoid, attr_labels, model2)
                        if group_intervention:
                            score = []
                            for group_idx in range(args.n_groups):
                                group_score = 0
                                for attr_idx in GROUP_DICT[group_idx]:
                                    group_score += score_list[attr_idx]
                                score.append(group_score/len(GROUP_DICT[group_idx]))
                        else:
                            score = score_list
                        intervention_order = np.argsort(score)[::-1]
                    intervention_order_list.append(intervention_order)
            all_intervention_order_list.append(intervention_order_list)
    return all_intervention_order_list

def simulate_intervention(intervention_order, criterion, replace_val, preds_by_attr, ptl_5, ptl_95, model2, attr_group_dict, b_attr_binary_outputs, b_class_labels, b_class_logits,
                                b_attr_outputs, b_attr_outputs_sigmoid, b_attr_outputs2, b_attr_labels,
                                instance_attr_labels, uncertainty_attr_labels, b_attr_intervened,
                                use_not_visible, min_uncertainty,
                                n_replace, use_relu, use_sigmoid, group_intervention, batch_intervention, trial, incorrect_idx_list, no_int_invisible,
                                n_trials=1, inference_mode='soft', mc_samples=5, connect_CY=False, intervention_ckpt=None):
    assert len(instance_attr_labels) == len(b_attr_labels), 'len(instance_attr_labels): %d, len(b_attr_labels): %d' % (
    len(instance_attr_labels), len(b_attr_labels))
    assert len(uncertainty_attr_labels) == len(
        b_attr_labels), 'len(uncertainty_attr_labels): %d, len(b_attr_labels): %d' % (
    len(uncertainty_attr_labels), len(b_attr_labels))


    if n_replace == 1 or n_replace == 0:
        b_attr_new = np.array(b_attr_outputs[:])
    else:
        b_attr_new = b_attr_intervened

    attr_replace_idx = []
    all_attr_ids = []

    if n_replace > 0:
        if batch_intervention:
            if intervention_ckpt is not None:
                start_idx = intervention_ckpt[n_replace - 1]
                end_idx = intervention_ckpt[n_replace]
                concept_idx = intervention_order[trial][start_idx:end_idx]
            else:
                start_idx = (n_replace - 1) * len(b_class_labels)
                end_idx = n_replace * len(b_class_labels)
                concept_idx = intervention_order[trial][start_idx:end_idx]
            if group_intervention:
                for concept_id in concept_idx:
                    group_idx = concept_id % args.n_groups
                    img_id = concept_id // args.n_groups
                    replace_idx = attr_group_dict[group_idx]
                    all_attr_ids.extend(replace_idx)
                    attr_replace_idx.extend(np.array(replace_idx) + img_id * args.n_attributes)
            else:
                replace_idx = concept_idx
                all_attr_ids.extend(np.array(replace_idx) % args.n_attributes)
                attr_replace_idx.extend(np.array(replace_idx))
        else: # single intervention
            for img_id in range(len(b_class_labels)):
                if group_intervention:
                    group_idx = intervention_order[trial][img_id][n_replace-1]
                    replace_idx = attr_group_dict[group_idx]
                else:
                    concept_idx = intervention_order[trial][img_id][n_replace-1]
                    replace_idx = [concept_idx]
                all_attr_ids.extend(replace_idx)
                attr_replace_idx.extend(np.array(replace_idx) + img_id * args.n_attributes)
        if replace_val == 'class_level':
            b_attr_new[attr_replace_idx] = np.array(b_attr_labels)[attr_replace_idx]
        else:
            b_attr_new[attr_replace_idx] = np.array(instance_attr_labels)[attr_replace_idx]
        
        if use_not_visible:
            not_visible_idx = np.where(np.array(uncertainty_attr_labels) == 1)[0]
            for idx in attr_replace_idx:
                if idx in not_visible_idx:
                    if no_int_invisible:
                        if n_replace == 1:
                            b_attr_new[idx] = b_attr_outputs_sigmoid[idx]
                        else:
                            b_attr_new[idx] = b_attr_intervened[idx]
                    else:
                        b_attr_new[idx] = 0

        if use_relu or not use_sigmoid:  # replace with percentile values
            binary_vals = b_attr_new[attr_replace_idx]
            for j, replace_idx in enumerate(attr_replace_idx):
                attr_idx = replace_idx % args.n_attributes
                b_attr_new[replace_idx] = (1 - binary_vals[j]) * ptl_5[attr_idx] + binary_vals[j] * ptl_95[attr_idx]
    
    # stage 2: calcaulte the label prediction error
    K = [1, 3, 5]
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
    class_acc = np.mean(np.array(b_class_outputs_new) == np.array(b_class_labels))

    incorrect_err_list = []
    b_class_outputs_new_numpy = np.array(b_class_outputs_new)
    b_class_labels_numpy = np.array(b_class_labels)
    for i in range(16):
        idx = incorrect_idx_list[i]
        pred = b_class_outputs_new_numpy[idx]
        label = b_class_labels_numpy[idx]
        incorrect_err_list.append(1 - np.mean(pred == label))
    
    eq_arr = np.array(b_class_outputs_new) == np.array(b_class_labels)
    return class_acc, incorrect_err_list, eq_arr, b_attr_new.flatten()


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
    parser.add_argument('-data_dir2', default='class_attr_data_10', help='directory to the raw data')
    parser.add_argument('-n_attributes', type=int, default=112, help='whether to apply bottlenecks to only a few attributes')
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
    parser.add_argument('-mc_samples', default=5, type=int, help='Number of MC samples for samp inference mode')
    parser.add_argument('-n_trials', help='Number of trials to run, when mode is random', type=int, default=1)
    parser.add_argument('-n_groups', help='Number of groups', type=int, default=28)
    parser.add_argument('-connect_CY', help='Whether to use concepts as auxiliary features (in multitasking) to predict Y', action='store_true')
    parser.add_argument('-no_intervention_when_invisible', action='store_true', help='No intervention when the related part is not visible')
    args = parser.parse_args()
    return args

def run(args):
    print(args)

    class_to_folder, attr_id_to_name = get_class_attribute_names()

    data = pickle.load(open(os.path.join(args.data_dir2, 'train.pkl'), 'rb'))
    class_attr_count = np.zeros((N_CLASSES, N_ATTRIBUTES, 2))
    for d in data:
        class_label = d['class_label']
        certainties = d['attribute_certainty']
        for attr_idx, a in enumerate(d['attribute_label']):
            if a == 0 and certainties[attr_idx] == 1:  # not visible
                continue
            class_attr_count[class_label][attr_idx][a] += 1

    class_attr_min_label = np.argmin(class_attr_count, axis=2)
    class_attr_max_label = np.argmax(class_attr_count, axis=2)
    equal_count = np.where(class_attr_min_label == class_attr_max_label)  # check where 0 count = 1 count, set the corresponding class attribute label to be 1
    class_attr_max_label[equal_count] = 1

    attr_class_count = np.sum(class_attr_max_label, axis=0)
    mask = np.where(attr_class_count >= 10)[0]  # select attributes that are present (on a class level) in at least [min_class_count] classes

    instance_attr_labels, uncertainty_attr_labels = [], []
    test_data = pickle.load(open(os.path.join(args.data_dir2, args.eval_data + '.pkl'), 'rb'))
    for d in test_data:
        instance_attr_labels.extend(list(np.array(d['attribute_label'])[mask]))
        uncertainty_attr_labels.extend(list(np.array(d['attribute_certainty'])[mask]))

    class_attr_id_to_name = dict() # map from 112 concepts to attribute name
    for k, v in attr_id_to_name.items():
        if k in mask:
            class_attr_id_to_name[list(mask).index(k)] = v

    attr_group_dict = dict() # dictionary of group id: {concept ids}
    curr_group_idx = 0
    with open('CUB_200_2011/attributes/attributes.txt', 'r') as f:
        all_lines = f.readlines()
        line0 = all_lines[0]
        prefix = line0.split()[1][:10]
        attr_group_dict[curr_group_idx] = [0]
        for i, line in enumerate(all_lines[1:]):
            curr = line.split()[1][:10]
            if curr != prefix:
                curr_group_idx += 1
                prefix = curr
                attr_group_dict[curr_group_idx] = [i + 1]
            else:
                attr_group_dict[curr_group_idx].append(i + 1)

    for group_id, attr_ids in attr_group_dict.items(): # remove sparse concepts from attr_group_dict
        new_attr_ids = []
        for attr_id in attr_ids:
            if attr_id in mask:
                new_attr_ids.append(attr_id)
        attr_group_dict[group_id] = new_attr_ids

    total_so_far = 0
    for group_id, attr_ids in attr_group_dict.items(): # change to 112 concept ids
        class_attr_ids = list(range(total_so_far, total_so_far + len(attr_ids)))
        total_so_far += len(attr_ids)
        attr_group_dict[group_id] = class_attr_ids

    class_attr_id = 0
    for i in range(len(mask)):
        class_attr_id_to_name[i] = attr_id_to_name[mask[i]]

    # stage 1: calculate \hat{c} & the percentiles
    _, _, b_class_labels, b_topk_class_outputs, b_class_logits, b_attr_labels, b_attr_outputs, b_attr_outputs_sigmoid, \
        b_wrong_idx, b_attr_outputs2, incorrect_idx_list, yerr_by_cerr = eval(args)

    
    b_class_outputs = b_topk_class_outputs[:, 0]
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

    N_TRIALS = args.n_trials
    MIN_UNCERTAINTY_GAP = 0
    assert args.criterion in ['rand', 'ucp', 'lcp', 'cctp', 'ectp', 'eudtp']
    assert args.level in ['i+s', 'i+b', 'g+s', 'g+b']
    if args.class_level:
        REPLACE_VAL = 'class_level'
    else:
        REPLACE_VAL = 'instance_level'

    group_intervention = args.level[0] == 'g' # Group-wise / Individual intervention
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
                                          attr_group_dict,
                                          b_attr_binary_outputs,
                                          b_class_labels,
                                          b_class_logits,
                                          b_attr_outputs,
                                          outputs_sigmoid,
                                          b_attr_outputs2,
                                          b_attr_labels,
                                          instance_attr_labels,
                                          uncertainty_attr_labels,
                                          args.use_invisible,
                                          MIN_UNCERTAINTY_GAP,
                                          1, args.use_relu,
                                          args.use_sigmoid,
                                          group_intervention,
                                          batch_intervention,
                                          n_trials=N_TRIALS,
                                          connect_CY=args.connect_CY)

    results = []
    all_err_list = []
    b_attr_intervened = b_attr_outputs

    if group_intervention:
        intervention_num = args.n_groups + 1
    else:
        intervention_num = args.n_attributes + 1


    intervened_concept_num_arr = np.zeros(args.n_attributes + 1) # i: number of images with i concepts intervened
    intervened_correct_num_arr = np.zeros(args.n_attributes + 1) # i: number of images with i concepts intervened & label correct

    incorrect_0_all_err_list = []


    if group_intervention and batch_intervention:
        intervened_concepts_cnt = 0
        intervened_avg_cnt_arr = np.zeros(args.n_groups + 1)
        intervened_avg_cnt_arr[0] = 0

    for trial in range(N_TRIALS):
        err_list = []
        incorrect_0_err_list = []
        intervened_concept_cnts_arr = np.zeros(len(b_class_labels)) # i: number of intervened concepts so far for i-th image
        for n_replace in list(range(intervention_num)):
            acc, incorrect_err_list, eq_arr, b_attr_intervened = simulate_intervention(intervention_order,
                                            args.criterion, REPLACE_VAL,
                                            preds_by_attr, ptl_5, ptl_95,
                                            model2,
                                            attr_group_dict,
                                            b_attr_binary_outputs,
                                            b_class_labels,
                                            b_class_logits,
                                            b_attr_outputs,
                                            b_attr_outputs_sigmoid,
                                            b_attr_outputs2,
                                            b_attr_labels,
                                            instance_attr_labels,
                                            uncertainty_attr_labels,
                                            b_attr_intervened,
                                            args.use_invisible,
                                            MIN_UNCERTAINTY_GAP,
                                            n_replace, args.use_relu,
                                            args.use_sigmoid,
                                            group_intervention,
                                            batch_intervention,
                                            trial,
                                            incorrect_idx_list,
                                            args.no_intervention_when_invisible,
                                            n_trials=N_TRIALS,
                                            inference_mode=args.inference_mode,
                                            mc_samples=args.mc_samples,
                                            connect_CY=args.connect_CY)

            if group_intervention:
                if n_replace == 0:
                    intervened_concept_num_arr[0] += len(b_class_labels)
                    intervened_correct_num_arr[0] += np.sum(eq_arr)
                else:
                    intervened_concepts_cnt = 0
                    for j in range(len(b_class_labels)):
                        if batch_intervention:
                            start_idx = (n_replace - 1) * len(b_class_labels)
                            end_idx = n_replace * len(b_class_labels)
                            intervened_group_ids = intervention_order[trial][start_idx : end_idx]
                        else:
                            intervened_group_id = intervention_order[trial][j][n_replace - 1]
                            intervened_concept_cnts_arr[j] += len(GROUP_DICT[intervened_group_id])
                            intervened_num = int(intervened_concept_cnts_arr[j])
                            intervened_concept_num_arr[intervened_num] += 1
                            intervened_correct_num_arr[intervened_num] += eq_arr[j]
                            
        
            results.append([n_replace, acc])
            err_list.append(1 - acc)
            
            incorrect_0_err_list.append(incorrect_err_list[0])


        all_err_list.append(err_list)
        incorrect_0_all_err_list.append(incorrect_0_err_list)

    if group_intervention:
        if not batch_intervention:
            intervened_err_list = 1 - intervened_correct_num_arr/(intervened_concept_num_arr+1e-6)
        else:
            groupbatch_all_avg_err_list = []
            for trial in range(N_TRIALS):
                groupbatch_avg_err_list = []
                intervention_cnts = 0
                check_point = np.zeros(args.n_attributes + 1)
                target_avg_intervened_cnt = 1
                check_point[0] = int(0)
                for j in range(len(intervention_order[trial])):
                    intervened_group_id = intervention_order[trial][j]
                    img_id = intervened_group_id // args.n_groups
                    group_id = intervened_group_id % args.n_groups
                    intervention_cnts += len(GROUP_DICT[group_id])
                    if intervention_cnts/len(b_class_labels) >= target_avg_intervened_cnt:
                        check_point[target_avg_intervened_cnt] = j
                        target_avg_intervened_cnt += 1
                
                check_point = check_point.astype(int)

                for n_replace in list(range(args.n_attributes + 1)):
                    acc, incorrect_err_list, eq_arr, b_attr_intervened = simulate_intervention(intervention_order,
                                                args.criterion, REPLACE_VAL,
                                                preds_by_attr, ptl_5, ptl_95,
                                                model2,
                                                attr_group_dict,
                                                b_attr_binary_outputs,
                                                b_class_labels,
                                                b_class_logits,
                                                b_attr_outputs,
                                                b_attr_outputs_sigmoid,
                                                b_attr_outputs2,
                                                b_attr_labels,
                                                instance_attr_labels,
                                                uncertainty_attr_labels,
                                                b_attr_intervened,
                                                args.use_invisible,
                                                MIN_UNCERTAINTY_GAP,
                                                n_replace, args.use_relu,
                                                args.use_sigmoid,
                                                group_intervention,
                                                batch_intervention,
                                                trial,
                                                incorrect_idx_list,
                                                args.no_intervention_when_invisible,
                                                n_trials=N_TRIALS,
                                                inference_mode=args.inference_mode,
                                                connect_CY=args.connect_CY,
                                                mc_samples=args.mc_samples,
                                                intervention_ckpt=check_point
                                                )
                    groupbatch_avg_err_list.append(1 - acc)
                groupbatch_all_avg_err_list.append(groupbatch_avg_err_list)

    err_mean_list = np.array(all_err_list).mean(axis=0)
    incorrect_0_err_mean_list = np.array(incorrect_0_all_err_list).mean(axis=0)

    if group_intervention:
        if batch_intervention:
            groupbatch_avg_mean_err_list = np.array(groupbatch_all_avg_err_list).mean(axis=0)
            return np.array([float(err) for err in err_mean_list]), np.array([float(err) for err in incorrect_0_err_mean_list]), np.array([float(err) for err in groupbatch_avg_mean_err_list])
        else:
            return np.array([float(err) for err in err_mean_list]), np.array([float(err) for err in incorrect_0_err_mean_list]), np.array([float(err) for err in intervened_err_list])
    else:    
        return np.array([float(err) for err in err_mean_list]), np.array([float(err) for err in incorrect_0_err_mean_list])

if __name__ == '__main__':
    torch.backends.cudnn.benchmark=True
    args = parse_arguments()
    print(args)

    args.group_intervention = args.level[0] == 'g' # Group-wise / Individual intervention
    args.batch_intervention = args.level[-1] == 'b' # Batch / single intervention

    if args.group_intervention:
        err_list = []
        incorrect_0_err_list = []
        intervened_err_list = []
    else:
        err_list = []
        incorrect_0_err_list = []
    for i, model_dir in enumerate(args.model_dirs):
        print('----------')
        args.model_dir = model_dir
        args.model_dir2 = args.model_dirs2[i] if args.model_dirs2 else None
        if args.group_intervention:
            err_values, incorrect_0_values, intervened_err_values = run(args)
            intervened_err_list.append(intervened_err_values)
        else:
            err_values, incorrect_0_values = run(args)
        err_list.append(err_values)
        incorrect_0_err_list.append(incorrect_0_values)

    output_string = ''

    err_mean = np.array(err_list).mean(axis=0)
    err_std = np.array(err_list).std(axis=0)

    incorrect_0_err_mean = np.array(incorrect_0_err_list).mean(axis=0)
    incorrect_0_err_std = np.array(incorrect_0_err_list).std(axis=0)

    if args.group_intervention:
        intervened_err_mean = np.array(intervened_err_list).mean(axis=0)
        intervened_err_std = np.array(intervened_err_list).std(axis=0)

    print("------ Result ------")
    print("Criterion: ",args.criterion)
    print("Level: ", args.level)
    print("Conceptualization: ", args.inference_mode)

    print('err mean')
    print(err_mean.tolist())
    print("err_std")
    print(err_std.tolist())

    print("incorrect_0_mean")
    print(incorrect_0_err_mean.tolist())
    print("incorrect_0_std") 
    print(incorrect_0_err_std.tolist())

    # Print comparison value to individual intervention
    if args.group_intervention:
        print("Comparison to individual intervention")
        print("group_err_mean", intervened_err_mean.tolist())
        print("group_err_std", intervened_err_std.tolist())