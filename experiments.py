
import pdb
import sys


import torch.backends.cudnn as cudnn
import random


def run_experiments(dataset, args):

    if dataset == 'CUB':
        from CUB.train import (
            train_X_to_C,
            train_oracle_C_to_y_and_test_on_Chat,
            train_Chat_to_y_and_test_on_Chat,
            train_X_to_C_to_y,
            train_X_to_y,
            train_X_to_Cy,
            test_time_intervention
        )

    elif dataset == 'SKINCON':
        from SKINCON.train import (
            train_X_to_C,
            train_oracle_C_to_y_and_test_on_Chat,
            train_Chat_to_y_and_test_on_Chat,
            train_X_to_C_to_y,
            train_X_to_y,
            train_X_to_Cy,
            test_time_intervention
        )
    
    elif dataset == 'SYNTHETIC':
        from SYNTHETIC.train import (
            train_X_to_C,
            train_oracle_C_to_y_and_test_on_Chat,
            train_Chat_to_y_and_test_on_Chat,
            train_X_to_C_to_y
        )

    experiment = args[0].exp
    if experiment == 'Concept_XtoC':
        train_X_to_C(*args)

    elif experiment == 'Independent_CtoY':
        train_oracle_C_to_y_and_test_on_Chat(*args)

    elif experiment == 'Sequential_CtoY':
        train_Chat_to_y_and_test_on_Chat(*args)

    elif experiment == 'Joint':
        train_X_to_C_to_y(*args)
        
    elif experiment == 'Standard':
        train_X_to_y(*args)

    elif experiment == 'Multitask':
        train_X_to_Cy(*args)

    elif experiment == 'TTI':
        test_time_intervention(*args)

def parse_arguments():
    # First arg must be dataset, and based on which dataset it is, we will parse arguments accordingly
    assert len(sys.argv) > 2, 'You need to specify dataset and experiment'
    assert sys.argv[1].upper() in ['CUB', 'SKINCON', 'SYNTHETIC'], 'Please specify the dataset'
    assert sys.argv[2] in ['Concept_XtoC', 'Independent_CtoY', 'Sequential_CtoY',
                           'Standard', 'StandardWithAuxC', 'Multitask', 'Joint','TTI'], \
        'Please specify valid experiment. Current: %s' % sys.argv[2]
    dataset = sys.argv[1].upper()
    experiment = sys.argv[2].upper()

    # Handle accordingly to dataset
    if dataset == 'CUB':
        from CUB.train import parse_arguments
    elif dataset == 'SKINCON':
        from SKINCON.train import parse_arguments
    elif dataset == 'SYNTHETIC':
        from SYNTHETIC.train import parse_arguments

    args = parse_arguments(experiment=experiment)
    return dataset, args

if __name__ == '__main__':

    import torch
    import numpy as np

    dataset, args = parse_arguments()

    if args[0].fix_seed:
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        np.random.seed(0)
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(0)
    else:
    # Seeds
        np.random.seed(args[0].seed)
        torch.manual_seed(args[0].seed)

    run_experiments(dataset, args)
