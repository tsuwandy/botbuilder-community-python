from itertools import chain

from torch import nn, optim
from torch.optim import Adam
from torch.utils.data import DataLoader
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(
    os.path.realpath(__file__)), "../"))

from models.sent_clf_trainer import SentClfTrainer
from modules.modules import Classifier
from sys_config import EXP_DIR
from utils.datasets import BucketBatchSampler, SortedSampler, SequentialSampler, ClfDataset, \
    ClfCollate
from utils.early_stopping import EarlyStopping
from utils.nlp import twitter_preprocessor
from utils.training import load_checkpoint, f1_macro, acc
from utils.transfer import dict_pattern_rename, load_state_dict_subset

####################################################################
# SETTINGS
####################################################################


def sent_clf(dataset, config, opts, transfer=False):
    from logger.experiment import Experiment

    opts.name = config["name"]
    X_train, y_train, _, X_val, y_val, _ = dataset
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
    else: preprocessor = None

    print("Building training dataset...")
    train_set = ClfDataset(X_train, y_train,
                           vocab=vocab, preprocess=preprocessor,
                           vocab_size=config["vocab"]["size"],
                           seq_len=config["data"]["seq_len"])

    print("Building validation dataset...")
    val_set = ClfDataset(X_val, y_val,
                         seq_len=train_set.seq_len, preprocess=preprocessor,
                         vocab=train_set.vocab)

    src_lengths = [len(x) for x in train_set.data]
    val_lengths = [len(x) for x in val_set.data]

    # select sampler & dataloader
    train_sampler = BucketBatchSampler(src_lengths, config["batch_size"], True)
    val_sampler = SortedSampler(val_lengths)
    val_sampler_train = SortedSampler(src_lengths)

    train_loader = DataLoader(train_set, batch_sampler=train_sampler,
                              num_workers=opts.cores, collate_fn=ClfCollate())
    val_loader = DataLoader(val_set, sampler=val_sampler,
                            batch_size=config["batch_size"],
                            num_workers=opts.cores, collate_fn=ClfCollate())
    val_loader_train_dataset = DataLoader(train_set,
                                          sampler=val_sampler_train,
                                          batch_size=config["batch_size"],
                                          num_workers=opts.cores,
                                          collate_fn=ClfCollate())
    ####################################################################
    # Model
    ####################################################################
    ntokens = len(train_set.vocab)
    model = Classifier(ntokens, len(set(train_set.labels)), **config["model"])
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
    trainer = SentClfTrainer(model, train_loader, val_loader,
                         (lm_criterion, clf_criterion),
                         [embed_optimizer,
                          rnn_optimizer,
                          top_optimizer],
                         config, opts.device,
                         valid_loader_train_set=val_loader_train_dataset,
                         unfreeze_embed=config["unfreeze_embed"],
                         unfreeze_rnn=config["unfreeze_rnn"])

    ####################################################################
    # Experiment: logging and visualizing the training process
    ####################################################################

    # exp = Experiment(opts.name, config, src_dirs=opts.source,
    #                  output_dir=EXP_DIR)
    # exp.add_metric("ep_loss_lm", "line", "epoch loss lm",
    #                ["TRAIN", "VAL"])
    # exp.add_metric("ep_loss_cls", "line", "epoch loss class",
    #                ["TRAIN", "VAL"])
    # exp.add_metric("ep_f1", "line", "epoch f1", ["TRAIN", "VAL"])
    # exp.add_metric("ep_acc", "line", "epoch accuracy", ["TRAIN", "VAL"])
    #
    # exp.add_value("epoch", title="epoch summary")
    # exp.add_value("progress", title="training progress")

    ep_loss_lm = [10000, 10000]
    ep_loss_cls = [10000, 10000]
    ep_f1 = [0, 0]
    ep_acc = [0, 0]
    e_log = 0
    progress = 0
    ####################################################################
    # Resume Training from a previous checkpoint
    ####################################################################
    if transfer:
        print("Transferring Encoder weights ...")
        dict_pattern_rename(checkpoint["model"],
                            {"encoder": "bottom_rnn", "decoder": "vocab"})
        load_state_dict_subset(model, checkpoint["model"])
    print(model)

    ####################################################################
    # Training Loop
    ####################################################################
    best_loss = None
    early_stopping = EarlyStopping("min", config["patience"])

    for epoch in range(0, config["epochs"]):

        train_loss = trainer.train_epoch()
        val_loss, y, y_pred = trainer.eval_epoch(val_set=True)
        _, y_train, y_pred_train = trainer.eval_epoch(train_set=True)
        # exp.update_metric("ep_loss_lm", train_loss[0], "TRAIN")
        ep_loss_lm[0] = train_loss[0]
        # exp.update_metric("ep_loss_lm", val_loss[0], "VAL")
        ep_loss_lm[1] = val_loss[0]
        # exp.update_metric("ep_loss_cls", train_loss[1], "TRAIN")
        # exp.update_metric("ep_loss_cls", val_loss[1], "VAL")
        ep_loss_cls[0] = train_loss[1]
        ep_loss_cls[1] = val_loss[1]

        # exp.update_metric("ep_f1", f1_macro(y_train, y_pred_train),
        #                   "TRAIN")
        ep_f1[0] = f1_macro(y_train, y_pred_train)
        # exp.update_metric("ep_f1", f1_macro(y, y_pred), "VAL")
        ep_f1[1] = f1_macro(y, y_pred)

        # exp.update_metric("ep_acc", acc(y_train, y_pred_train), "TRAIN")
        # exp.update_metric("ep_acc", acc(y, y_pred), "VAL")

        ep_acc[0] = acc(y_train, y_pred_train)
        ep_acc[1] = acc(y, y_pred)

        # print('Train lm Loss : {}\nVal lm Loss : {}\nTrain cls Loss : {}\nVal cls Loss : {}\n Train f1 : {}\nVal f1 : {}\nTrain acc : {}\n Val acc : {}'.format(
        #     ep_loss_lm[0], ep_loss_lm[1], ep_loss_cls[0], ep_loss_cls[1], ep_f1[0], ep_f1[1], ep_acc[0], ep_acc[1]
        # ))
        # epoch_log = exp.log_metrics(["ep_loss_lm", "ep_loss_cls","ep_f1", "ep_acc"])
        epoch_log = 'Train lm Loss : {}\nVal lm Loss : {}\nTrain cls Loss : {}\nVal cls Loss : {}\n Train f1 : {}\nVal f1 : {}\nTrain acc : {}\n Val acc : {}'.format(
            ep_loss_lm[0], ep_loss_lm[1], ep_loss_cls[0], ep_loss_cls[1], ep_f1[0], ep_f1[1], ep_acc[0], ep_acc[1])
        print(epoch_log)
        # exp.update_value("epoch", epoch_log)
        e_log = epoch_log
        # print('')
        # Save the model if the val loss is the best we've seen so far.
        # if not best_loss or val_loss[1] < best_loss:
        #     best_loss = val_loss[1]
        #     trainer.best_acc = acc(y, y_pred)
        #     trainer.best_f1 = f1_macro(y, y_pred)
        #     trainer.checkpoint(name=opts.name, timestamp=True)
        best_loss = val_loss[1]
        trainer.best_acc = acc(y, y_pred)
        trainer.best_f1 = f1_macro(y, y_pred)
        trainer.checkpoint(name=opts.name, tags=str(epoch))

        # if early_stopping.stop(val_loss[1]):
        #     print("Early Stopping (according to classification loss)....")
        #     break

        print("\n" * 2)
    
    return best_loss, trainer.best_acc, trainer.best_f1


def sent_clf_test(dataset, config, opts, transfer=False, output_dir=None, checkpoint_name='scv2_aux_ft_gu_last'):
    opts.name = config["name"]
    X_test, y_test, _, pids, human_summaries = dataset
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
    model = Classifier(ntokens, len(set([0, 1])), **config["model"])
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
    trainer = SentClfTrainer(model, None, None,
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
    test_set = ClfDataset(X_test, y_test,
                         seq_len=config['data']['seq_len'], preprocess=preprocessor,
                         vocab=_vocab)
    test_lengths = [len(x) for x in test_set.data]
    # test_sampler = SequentialSampler(test_lengths)
    test_sampler = SortedSampler(test_lengths)

    test_loader = DataLoader(test_set, sampler=test_sampler,
                            batch_size=config["batch_size"],
                            num_workers=opts.cores, collate_fn=ClfCollate())

    trainer.test_loader = test_loader

    _, labels_array, predicted = trainer.test_epoch()

    # str_to_print = ''
    # for x, y in zip(labels_array, predicted):
    #     str_to_print += '{}\t{}\n'.format(x, y)
    # print(str_to_print)

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

    import os
    if not os.path.exists('{}/ref_abs'.format(output_dir)):
        os.mkdir('{}/ref_abs'.format(output_dir))
    if not os.path.exists('{}/dec'.format(output_dir)):
        os.mkdir('{}/dec'.format(output_dir))

    file_index = 0
    for elem_key in pids_dic:
        write_ref = open('{}/ref_abs/{}.ref'.format(output_dir, file_index), 'w')
        write_pred = open('{}/dec/{}.dec'.format(output_dir, file_index), 'w')
        if human_summaries is None:
            for pair in pids_dic[elem_key]:
                if pair[0] == 1:
                    write_ref.write(pair[2] + '\n')
                if pair[1] == 1:
                    write_pred.write(pair[2] + '\n')
        else:
            write_ref.write(pids_dic[elem_key][0][3] + '\n')
            for pair in pids_dic[elem_key]:
                if pair[1] == 1:
                    write_pred.write(pair[2] + '\n')
        write_ref.close()
        write_pred.close()

        file_index += 1

    # write_ref.close()
    # write_pred.close()
    ########################## Write Results #########################


