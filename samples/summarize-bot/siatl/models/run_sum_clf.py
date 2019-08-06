import argparse

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(
    os.path.realpath(__file__)), "./"))

from models.sum_clf import sum_clf, sum_clf_test
# from models.sent_clf_no_aux import sent_clf_no_aux
from utils.data_parsing import load_dataset
from utils.opts import train_options


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", required=False,
                    default='SCV2_aux_ft_gu.yaml',
                    help="config file of input data")
parser.add_argument("-dv", "--device_to_use", required=False,
                    default='auto',
                    help="device for training")
parser.add_argument("-t", "--transfer", action='store_true',
                    help="transfer from pretrained language model or train"
                         "a randomly initialized model")
parser.add_argument("-a", "--aux_loss", action='store_true',
                    help="add an auxiliary LM loss to the transferred model"
                         "or simply transfer a LM to a classifier"
                         " and fine-tune")
parser.add_argument("-o", "--output_dir", default='./output/forum/',
                    help="The output Directory")
parser.add_argument("-j", "--job", default='Train',
                    help="Train or Test")
parser.add_argument("-tc", "--test_checkpoint_name", default='scv2_aux_ft_gu_last',
                    help="checkpoint to use for testing")

args = parser.parse_args()
input_config = args.input
transfer = args.transfer
aux_loss = args.aux_loss
device_to_use = args.device_to_use
opts, config = train_options(input_config, parser, device_to_use)

if args.job == 'Train':
    test = False
else:
    test = True

output_dir = args.output_dir#'./output/forum/'

test_checkpoint_name = args.test_checkpoint_name

dataset = load_dataset(config, test=test)

# if aux_loss:
if test is True:
    print('Testing..............')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    sum_clf_test(dataset=dataset, config=config, opts=opts, transfer=transfer, output_dir=output_dir, checkpoint_name=test_checkpoint_name)
else:
    print('Training..............')
    loss, accuracy, f1 = sum_clf(dataset=dataset, config=config, opts=opts, transfer=transfer)
# else:
#     loss, accuracy, f1 = sent_clf_no_aux(dataset=dataset, config=config, opts=opts, transfer=transfer)
