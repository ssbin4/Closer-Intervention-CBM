"""
Evaluate trained models on the official CUB test set
"""
import os
import sys
import torch
import joblib
import argparse
import numpy as np
from sklearn.metrics import f1_score
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from CUB.dataset import load_data
from CUB.config import BASE_DIR, N_CLASSES, N_ATTRIBUTES, GROUP_DICT
from analysis import AverageMeter, multiclass_metric, accuracy, binary_accuracy

K = [1, 3, 5] #top k class accuracies to compute

def eval(args):
    """
    Run inference using model (and model2 if bottleneck)
    Returns: (for notebook analysis)
    all_class_labels: flattened list of class labels for each image
    topk_class_outputs: array of top k class ids predicted for each image. Shape = size of test set * max(K)
    all_class_outputs: array of all logit outputs for class prediction, shape = N_TEST * N_CLASS
    all_attr_labels: flattened list of labels for each attribute for each image (length = N_ATTRIBUTES * N_TEST)
    all_attr_outputs: flatted list of attribute logits (after ReLU/ Sigmoid respectively) predicted for each attribute for each image (length = N_ATTRIBUTES * N_TEST)
    all_attr_outputs_sigmoid: flatted list of attribute logits predicted (after Sigmoid) for each attribute for each image (length = N_ATTRIBUTES * N_TEST)
    wrong_idx: image ids where the model got the wrong class prediction (to compare with other models)
    incorrect_idx_list: list of lists where the i-th list contains the images which predicted i concepts incorrectly
    """
    if args.model_dir:
        model = torch.load(args.model_dir)
    else:
        model = None

    if not hasattr(model, 'use_relu'):
        if args.use_relu:
            model.use_relu = True
        else:
            model.use_relu = False
    if not hasattr(model, 'use_sigmoid'):
        if args.use_sigmoid:
            model.use_sigmoid = True
        else:
            model.use_sigmoid = False
    if not hasattr(model, 'cy_fc'):
        model.cy_fc = None
    model.eval()

    incorrect_idx_list = [[] for _ in range(16) ]

    if args.model_dir2:
        if 'rf' in args.model_dir2:
            model2 = joblib.load(args.model_dir2)
        else:
            model2 = torch.load(args.model_dir2)
        if not hasattr(model2, 'use_relu'):
            if args.use_relu:
                model2.use_relu = True
            else:
                model2.use_relu = False
        if not hasattr(model2, 'use_sigmoid'):
            if args.use_sigmoid:
                model2.use_sigmoid = True
            else:
                model2.use_sigmoid = False
        model2.eval()
    else:
        model2 = None

    if args.use_attr:
        attr_acc_meter = [AverageMeter()]
        if args.feature_group_results:  # compute acc for each feature individually in addition to the overall accuracy
            for _ in range(args.n_attributes):
                attr_acc_meter.append(AverageMeter())
    else:
        attr_acc_meter = None

    class_acc_meter = []
    for j in range(len(K)):
        class_acc_meter.append(AverageMeter())

    data_dir = os.path.join(BASE_DIR, args.data_dir, args.eval_data + '.pkl')
    loader = load_data([data_dir], args.use_attr, args.no_img, args.batch_size, image_dir=args.image_dir,
                       n_class_attr=args.n_class_attr)
    
    all_outputs, all_targets = [], []
    all_attr_labels, all_attr_outputs, all_attr_outputs_sigmoid, all_attr_outputs2 = [], [], [], []
    all_class_labels, all_class_outputs, all_class_logits = [], [], []
    topk_class_labels, topk_class_outputs = [], []

    data_size = 0

    incorrect_cnt = np.zeros(args.n_attributes)            # i-th element: number of images where 'i' concepts are mispredicted
    y_correct_by_incorrect_c = np.zeros(args.n_attributes) # i-th element: number of correctly classified images when i number of concepts are mispredicted

    for data_idx, data in enumerate(loader):
        if args.use_attr:
            if args.no_img:  # A -> Y
                inputs, labels = data
                if isinstance(inputs, list):
                    inputs = torch.stack(inputs).t().float()
                inputs = inputs.float()
            else:
                inputs, labels, attr_labels = data
                attr_labels = torch.stack(attr_labels).t()  # N x 312
        else:  # simple finetune
            inputs, labels = data

        inputs_var = torch.autograd.Variable(inputs).cuda()
        labels_var = torch.autograd.Variable(labels).cuda()

        if args.attribute_group:
            outputs = []
            f = open(args.attribute_group, 'r')
            for line in f:
                attr_model = torch.load(line.strip())
                outputs.extend(attr_model(inputs_var))
        else:
            outputs = model(inputs_var)
        if args.use_attr:
            if args.no_img:  # A -> Y
                class_outputs = outputs
            else:
                if args.bottleneck:
                    if args.use_relu:
                        attr_outputs = [torch.nn.ReLU()(o) for o in outputs]
                        attr_outputs_sigmoid = [torch.nn.Sigmoid()(o) for o in outputs]
                    elif args.use_sigmoid:
                        attr_outputs = [torch.nn.Sigmoid()(o) for o in outputs]
                        attr_outputs_sigmoid = attr_outputs
                    else:
                        attr_outputs = outputs
                        attr_outputs_sigmoid = [torch.nn.Sigmoid()(o) for o in outputs]
                    if model2:
                        stage2_inputs = torch.cat(attr_outputs, dim=1)
                        if args.inference_mode == 'soft' or args.inference_mode == 'hard':
                            if args.inference_mode == 'hard':
                                stage2_inputs = stage2_inputs >= (torch.ones_like(stage2_inputs) * 0.5)
                                stage2_inputs = stage2_inputs.float()
                            class_outputs = model2(stage2_inputs)
                        elif args.inference_mode == 'samp':
                            class_outputs_all = []
                            for _ in range(args.mc_samples):
                                rand_num = torch.rand(*stage2_inputs.size())
                                sampled_stage2_inputs = rand_num.cuda() < stage2_inputs
                                sampled_stage2_inputs = sampled_stage2_inputs.float()
                                _class_outputs = model2(sampled_stage2_inputs)
                                class_outputs_all.append(class_outputs_all)
                            class_outputs = torch.mean(torch.stack(class_outputs_all, axis=0), axis=0)
    
                    else:  # for debugging bottleneck performance without running stage 2
                        class_outputs = torch.zeros([inputs.size(0), N_CLASSES],
                                                    dtype=torch.float64).cuda()  # ignore this
                else:  # cotraining, end2end
                    if args.use_relu:
                        attr_outputs = [torch.nn.ReLU()(o) for o in outputs[1:]]
                        attr_outputs_sigmoid = [torch.nn.Sigmoid()(o) for o in outputs[1:]]
                    elif args.use_sigmoid:
                        attr_outputs = [torch.nn.Sigmoid()(o) for o in outputs[1:]]
                        attr_outputs_sigmoid = attr_outputs
                    else:
                        attr_outputs = outputs[1:]
                        attr_outputs_sigmoid = [torch.nn.Sigmoid()(o) for o in outputs[1:]]

                    class_outputs = outputs[0]
                
                batch_incorrect_num = np.zeros(inputs.size(0))

                for i in range(args.n_attributes):

                    pred = attr_outputs_sigmoid[i].squeeze().cpu() >= 0.5
                    eq = (pred.int()).eq(attr_labels[:,i].int())

                    not_eq = [int(not ele) for ele in eq]

                    batch_incorrect_num += not_eq

                    acc = binary_accuracy(attr_outputs_sigmoid[i].squeeze(), attr_labels[:, i])
                    acc = acc.data.cpu().numpy()
                    attr_acc_meter[0].update(acc, inputs.size(0))
                    if args.feature_group_results:  # keep track of accuracy of individual attributes
                        attr_acc_meter[i + 1].update(acc, inputs.size(0))

     
                attr_outputs = torch.cat([o.unsqueeze(1) for o in attr_outputs], dim=1)
                attr_outputs_sigmoid = torch.cat([o for o in attr_outputs_sigmoid], dim=1)
                all_attr_outputs.extend(list(attr_outputs.flatten().data.cpu().numpy()))
                all_attr_outputs_sigmoid.extend(list(attr_outputs_sigmoid.flatten().data.cpu().numpy()))
                all_attr_labels.extend(list(attr_labels.flatten().data.cpu().numpy()))

                for i in range(inputs.size(0)):
                    incorrect_cnt[int(batch_incorrect_num[i])] += 1

                
                for i in range(inputs.size(0)):
                    incorrect = int(batch_incorrect_num[i])
                    if incorrect >= 15:
                        incorrect_idx_list[15].append(data_size + i)
                    else:
                        incorrect_idx_list[incorrect].append(data_size + i)
                data_size += inputs.size(0)

        else:
            class_outputs = outputs[0]

        _, topk_preds = class_outputs.topk(max(K), 1, True, True)
        _, preds = class_outputs.topk(1, 1, True, True)

        all_class_outputs.extend(list(preds.detach().cpu().numpy().flatten()))
        all_class_labels.extend(list(labels.data.cpu().numpy()))
        all_class_logits.extend(class_outputs.detach().cpu().numpy())
        topk_class_outputs.extend(topk_preds.detach().cpu().numpy())
        topk_class_labels.extend(labels.view(-1, 1).expand_as(preds))


        np.set_printoptions(threshold=sys.maxsize)
        class_acc = accuracy(class_outputs, labels, topk=K)  # only class prediction accuracy

        _, pred = class_outputs.topk(1, 1, True, True)
        pred = pred.t()
        temp = labels.view(1, -1).expand_as(pred)
        temp = temp.cuda()
        class_eq = pred.eq(temp)[0]

        for i in range(inputs.size(0)):
            incorrect_num = int(batch_incorrect_num[i])
            y_correct_by_incorrect_c [incorrect_num] += int(class_eq[i])

        for m in range(len(class_acc_meter)):
            class_acc_meter[m].update(class_acc[m], inputs.size(0))

    all_class_logits = np.vstack(all_class_logits)
    topk_class_outputs = np.vstack(topk_class_outputs)
    topk_class_labels = np.vstack(topk_class_labels)
    wrong_idx = np.where(np.sum(topk_class_outputs == topk_class_labels, axis=1) == 0)[0]

    yerr_by_cerr = 1 - np.array(y_correct_by_incorrect_c)/(np.array(incorrect_cnt)+1e-6)

    for j in range(len(K)):
        print('Average top %d class accuracy: %.5f' % (K[j], class_acc_meter[j].avg))

    if args.use_attr and not args.no_img:  # print some metrics for attribute prediction performance
        print('Average attribute accuracy: %.5f' % attr_acc_meter[0].avg)
        all_attr_outputs_int = np.array(all_attr_outputs_sigmoid) >= 0.5
        if args.feature_group_results:
            n = len(all_attr_labels)
            all_attr_acc, all_attr_f1 = [], []
            for i in range(args.n_attributes):
                acc_meter = attr_acc_meter[1 + i]
                attr_acc = float(acc_meter.avg)
                attr_preds = [all_attr_outputs_int[j] for j in range(n) if j % args.n_attributes == i]
                attr_labels = [all_attr_labels[j] for j in range(n) if j % args.n_attributes == i]
                attr_f1 = f1_score(attr_labels, attr_preds)
                all_attr_acc.append(attr_acc)
                all_attr_f1.append(attr_f1)

            bins = np.arange(0, 1.01, 0.1)
            acc_bin_ids = np.digitize(np.array(all_attr_acc) / 100.0, bins)
            acc_counts_per_bin = [np.sum(acc_bin_ids == (i + 1)) for i in range(len(bins))]
            f1_bin_ids = np.digitize(np.array(all_attr_f1), bins)
            f1_counts_per_bin = [np.sum(f1_bin_ids == (i + 1)) for i in range(len(bins))]
            print("Accuracy bins:")
            print(acc_counts_per_bin)
            print("F1 bins:")
            print(f1_counts_per_bin)
            np.savetxt(os.path.join(args.log_dir, 'concepts.txt'), f1_counts_per_bin)

        balanced_acc, report = multiclass_metric(all_attr_outputs_int, all_attr_labels)
        f1 = f1_score(all_attr_labels, all_attr_outputs_int)
        print("Total 1's predicted:", sum(np.array(all_attr_outputs_sigmoid) >= 0.5) / len(all_attr_outputs_sigmoid))
        print('Avg attribute balanced acc: %.5f' % (balanced_acc))
        print("Avg attribute F1 score: %.5f" % f1)
        print(report + '\n')
    return class_acc_meter, attr_acc_meter, all_class_labels, topk_class_outputs, all_class_logits, all_attr_labels, all_attr_outputs, all_attr_outputs_sigmoid, wrong_idx, all_attr_outputs2, incorrect_idx_list, yerr_by_cerr

if __name__ == '__main__':
    torch.backends.cudnn.benchmark=True
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-log_dir', default='.', help='where results are stored')
    parser.add_argument('-model_dirs', default=None, nargs='+', help='where the trained models are saved')
    parser.add_argument('-model_dirs2', default=None, nargs='+', help='where another trained model are saved (for bottleneck only)')
    parser.add_argument('-eval_data', default='test', help='Type of data (train/ val/ test) to be used')
    parser.add_argument('-use_attr', help='whether to use attributes (FOR COTRAINING ARCHITECTURE ONLY)', action='store_true')
    parser.add_argument('-no_img', help='if included, only use attributes (and not raw imgs) for class prediction', action='store_true')
    parser.add_argument('-bottleneck', help='whether to predict attributes before class labels', action='store_true')
    parser.add_argument('-image_dir', default='images', help='test image folder to run inference on')
    parser.add_argument('-n_class_attr', type=int, default=2, help='whether attr prediction is a binary or triary classification')
    parser.add_argument('-data_dir', default='', help='directory to the data used for evaluation')
    parser.add_argument('-n_attributes', type=int, default=N_ATTRIBUTES, help='whether to apply bottlenecks to only a few attributes')    
    parser.add_argument('-attribute_group', default=None, help='file listing the (trained) model directory for each attribute group')
    parser.add_argument('-feature_group_results', help='whether to print out performance of individual atttributes', action='store_true')
    parser.add_argument('-use_relu', help='Whether to include relu activation before using attributes to predict Y. For end2end & bottleneck model', action='store_true')
    parser.add_argument('-use_sigmoid', help='Whether to include sigmoid activation before using attributes to predict Y. For end2end & bottleneck model', action='store_true')
    parser.add_argument('-inference_mode', default='soft', help='mode of inference')
    parser.add_argument('-mc_samples', default=5, type=int, help='Number of MC samples for samp inference mode')
    args = parser.parse_args()
    args.batch_size = 16

    print(args)
    y_results, c_results = [], []
    yerr_by_cerr_results = []
    for i, model_dir in enumerate(args.model_dirs):
        args.model_dir = model_dir
        args.model_dir2 = args.model_dirs2[i] if args.model_dirs2 else None
        result = eval(args)
        class_acc_meter, attr_acc_meter = result[0], result[1]
        yerr_by_cerr = result[-1]
        y_results.append(1 - class_acc_meter[0].avg[0].item() / 100.)
        if attr_acc_meter is not None:
            c_results.append(1 - attr_acc_meter[0].avg.item() / 100.)
        else:
            c_results.append(-1)
        yerr_by_cerr_results.append(yerr_by_cerr)
    values = (np.mean(y_results), np.std(y_results), np.mean(c_results), np.std(c_results))
    output_string = '%.4f %.4f %.4f %.4f' % values
    print_string = 'Error of y: %.4f +- %.4f, Error of C: %.4f +- %.4f' % values
    print(print_string)

    yerr_by_cerr_mean = np.array(yerr_by_cerr_results).mean(axis=0)
    yerr_by_cerr_std = np.array(yerr_by_cerr_results).std(axis=0)

    print("yerr_by_cerr_mean", yerr_by_cerr_mean.tolist())
    print("yerr_by_cerr_std", yerr_by_cerr_std.tolist())
    output = open(os.path.join(args.log_dir, 'results.txt'), 'w')
    output.write(output_string)