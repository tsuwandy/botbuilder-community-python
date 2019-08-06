from itertools import chain

from torch import nn, optim
from torch.optim import Adam
from torch.utils.data import DataLoader
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(
    os.path.realpath(__file__)), "../"))

from models.sum_clf_trainer import SumClfTrainer
from modules.modules import SummarizationClassifier
from sys_config import EXP_DIR
from utils.datasets import BucketBatchSampler, SortedSampler, SequentialSampler, SUMDataset, SumCollate
from utils.early_stopping import EarlyStopping
from utils.nlp import twitter_preprocessor
from utils.training import load_checkpoint, f1_macro, acc
from utils.transfer import dict_pattern_rename, load_state_dict_subset

####################################################################
# SETTINGS
####################################################################


def sum_clf_test(dataset, config, opts, transfer=False, output_dir=None, checkpoint_name='scv2_aux_ft_gu_last'):
    opts.name = config["name"]
    X_test, y_test, posts_test, pids, human_summaries = dataset
    vocab = None
    if transfer:
        opts.transfer = config["pretrained_lm"]
        checkpoint = load_checkpoint(opts.transfer)
        config["vocab"].update(checkpoint["config"]["vocab"])
        dict_pattern_rename(checkpoint["config"]["model"],
                            {"rnn_": "bottom_rnn_"})
        config["model"].update(checkpoint["config"]["model"])
        vocab = checkpoint["vocab"]

    ####################################################################
    # Load Preprocessed Datasets
    ####################################################################
    if config["preprocessor"] == "twitter":
        preprocessor = twitter_preprocessor()
    else:
        preprocessor = None

    ####################################################################
    # Model
    ####################################################################
    ntokens = 70004
    model = SummarizationClassifier(ntokens, len(set([0, 1])), **config["model"])
    model.to(opts.device)

    clf_criterion = nn.CrossEntropyLoss()
    lm_criterion = nn.CrossEntropyLoss(ignore_index=0)

    embed_parameters = filter(lambda p: p.requires_grad,
                              model.embed.parameters())
    bottom_parameters = filter(lambda p: p.requires_grad,
                               chain(model.bottom_rnn.parameters(),
                                     model.vocab.parameters()))
    if config["model"]["has_att"]:
        top_parameters = filter(lambda p: p.requires_grad,
                                chain(model.top_rnn.parameters(),
                                      model.attention.parameters(),
                                      model.classes.parameters()))
    else:
        top_parameters = filter(lambda p: p.requires_grad,
                                chain(model.top_rnn.parameters(),
                                      model.classes.parameters()))

    embed_optimizer = optim.ASGD(embed_parameters, lr=0.0001)
    rnn_optimizer = optim.ASGD(bottom_parameters)
    top_optimizer = Adam(top_parameters, lr=config["top_lr"])
    ####################################################################
    # Training Pipeline
    ####################################################################

    # Trainer: responsible for managing the training process
    trainer = SumClfTrainer(model, None, None,
                             (lm_criterion, clf_criterion),
                             [embed_optimizer,
                              rnn_optimizer,
                              top_optimizer],
                             config, opts.device,
                             valid_loader_train_set=None,
                             unfreeze_embed=config["unfreeze_embed"],
                             unfreeze_rnn=config["unfreeze_rnn"], test_loader=None)

    ####################################################################
    # Resume Training from a previous checkpoint
    ####################################################################
    if transfer:
        print("Transferring Encoder weights ...")
        dict_pattern_rename(checkpoint["model"],
                            {"encoder": "bottom_rnn", "decoder": "vocab"})
        load_state_dict_subset(model, checkpoint["model"])
    print(model)

    _vocab = trainer.load_checkpoint(name=checkpoint_name, path=None)
    test_set = SUMDataset(X_test, posts_test, y_test,
                          seq_len=config['data']['seq_len'], post_len=config['data']['post_len'], preprocess=preprocessor,
                          vocab=_vocab)
    test_lengths = [len(x) for x in test_set.data]
    test_sampler = SortedSampler(test_lengths)

    # test_loader = DataLoader(test_set, sampler=test_sampler,
    #                          batch_size=config["batch_size"],
    #                          num_workers=opts.cores, collate_fn=SumCollate())

    test_loader = DataLoader(test_set, sampler=test_sampler,
                             batch_size=config["batch_size"],
                             num_workers=0, collate_fn=SumCollate())

    trainer.test_loader = test_loader

    _, labels_array, predicted = trainer.test_epoch()

    pids_dic = {}
    if human_summaries is None:
        for x, y, sent, z in zip(y_test, predicted, X_test, pids):
            if z in pids_dic:
                pids_dic[z].append([x, y, sent])
            else:
                pids_dic[z] = [[x, y, sent]]
    else:
        for x, y, sent, z, h_summary in zip(y_test, predicted, X_test, pids, human_summaries):
            if z in pids_dic:
                pids_dic[z].append([x, y, sent, h_summary])
            else:
                pids_dic[z] = [[x, y, sent, h_summary]]

    # import os
    # if not os.path.exists('{}/ref_abs'.format(output_dir)):
    #     os.mkdir('{}/ref_abs'.format(output_dir))
    # if not os.path.exists('{}/dec'.format(output_dir)):
    #     os.mkdir('{}/dec'.format(output_dir))

    file_index = 0
    all_summaries = []
    for elem_key in pids_dic:
        current_summary = ''
        for pair in pids_dic[elem_key]:
            if pair[1] == 1:
                current_summary += pair[2] + '\n'

        all_summaries.append(current_summary)

    return all_summaries
    ########################## Write Results #########################


