"""
InceptionV3 Network modified from https://github.com/pytorch/vision/blob/master/torchvision/models/inception.py
New changes: add softmax layer + option for freezing lower layers except fc
"""
from cmath import exp
import os
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

import torchvision.models as models
from torchvision import transforms

class End2EndModel(torch.nn.Module):
    def __init__(self, model1, model2, n_attributes, use_relu=False, use_sigmoid=False, n_class_attr=2):
        super(End2EndModel, self).__init__()
        self.first_model = model1
        self.sec_model = model2
        self.use_relu = use_relu
        self.use_sigmoid = use_sigmoid
        self.n_attributes = n_attributes

    def forward_stage2(self, stage1_out):
        if self.use_relu:
            attr_outputs = [nn.ReLU()(o) for o in stage1_out]
        elif self.use_sigmoid:
            attr_outputs = [torch.nn.Sigmoid()(o) for o in stage1_out]
        else:
            attr_outputs = stage1_out

        stage2_inputs = attr_outputs
        stage2_inputs = torch.cat(stage2_inputs, dim=1)
        all_out = [self.sec_model(stage2_inputs)]
        all_out.extend(stage1_out)
        return all_out

    def forward(self, x):
        outputs = self.first_model(x)
        new_outputs = []

        for c in range(self.n_attributes):
            new_outputs.append(outputs[:, c].unsqueeze(1))

        outputs = new_outputs
        return self.forward_stage2(outputs)

class MLP(nn.Module):
    def __init__(self, input_dim, num_classes, expand_dim):
        super(MLP, self).__init__()
        self.expand_dim = expand_dim
        self.num_classes = num_classes
        if self.expand_dim:
            self.linear = nn.Linear(input_dim, expand_dim)
            self.activation1 = torch.nn.ReLU()
            self.linear2 = nn.Linear(expand_dim, expand_dim) #softmax is automatically handled by loss function
            self.activation2 = torch.nn.ReLU()
            self.linear3 = nn.Linear(expand_dim, num_classes) #softmax is automatically handled by loss function
        else:
            self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = self.linear(x)
        if hasattr(self, 'expand_dim') and self.expand_dim:
            x = self.activation1(x)
            x = self.linear2(x)
            x = self.activation2(x)
            x = self.linear3(x)
        return x


class FC(nn.Module):

    def __init__(self, input_dim, output_dim, expand_dim, stddev=None):
        """
        Extend standard Torch Linear layer to include the option of expanding into 2 Linear layers
        """
        super(FC, self).__init__()
        self.expand_dim = expand_dim
        if self.expand_dim > 0:
            self.relu = nn.ReLU()
            self.fc_new = nn.Linear(input_dim, expand_dim)
            self.fc = nn.Linear(expand_dim, output_dim)
        else:
            self.fc = nn.Linear(input_dim, output_dim)
        if stddev:
            self.fc.stddev = stddev
            if expand_dim > 0:
                self.fc_new.stddev = stddev

    def forward(self, x):
        if self.expand_dim > 0:
            x = self.fc_new(x)
            x = self.relu(x)
        x = self.fc(x)
        return x