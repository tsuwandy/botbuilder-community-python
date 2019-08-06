# import HelpingFunctions
# from model import EncoderBiRNN
# import constants
# import torch.nn as nn
# import torch
# import data_loader as dL
# import model_loader as mL
# import random
# import math
# import trainer as trainer
# from time import sleep
import pickle
import os
import SummaRuNNer.HelpingFunctions as hF
from tqdm import tqdm

# ######################################################
# params = {}
# ############ Data params
# params['DATA_Path'] = './cnn_data/finished_files/'  # './github_data/issues_v2_combined.xml'#'./cnn_data/finished_files/' #'./forum_data/data_V2/Parsed_Data.xml'
# params['data_set_name'] = 'cnn'
# ############ Model params
# params['use_BERT'] = False
# params['BERT_Model_Path'] = '../pytorch-pretrained-BERT/bert_models/uncased_L-12_H-768_A-12/'
# params['BERT_embedding_size'] = 768
# params['BERT_layers'] = [-1]
#
# params['vocab_size'] = 70000
# params['use_back_translation'] = False
# params['back_translation_file'] = None
# params['Global_max_sequence_length'] = 25
# params['Global_max_num_sentences'] = 20
# params['use_external_vocab'] = False
# params['external_vocab_file'] = './checkpoint/forum_vocab.pickle'
# params['encoding_batch_size'] = 64
# params['data_split_size'] = 15000
# ############ device
# params['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ThreadObject(object):
    def __init__(self, issue_id, issue_title, initial_post):
        self.issue_id = issue_id
        self.issue_title = issue_title

        self.initial_post = initial_post
        self.initial_post_translated = []

        self.reply_sentences = []
        self.reply_sentences_translated = []

        self.selected_sentences = []
        self.selected_sentences_translated = []

        self.owner = ''
        self.repository = ''
        self.human_summary_1 = ''
        self.human_summary_2 = ''
        self.generated_full_summary = ''

    def add_comments(self, comments):
        self.reply_sentences += comments

    def set_selected_sentences(self, selected_sentences):
        self.selected_sentences = selected_sentences

class preprocessor(object):
    def __init__(self, params):
        self.params = params

    ######################################################
    def tokenize(self, data, use_back_translation=False):  # , max_num_sentences=None, max_sentence_length=None):
        all_comments = []
        all_posts = []
        all_answers = []
        all_human_summaries = []

        all_comments_translated = []
        all_posts_translated = []

        print('Tokenizing Data...')
        for i in tqdm(range(0, len(data))):
            post = [x.replace('\n', '').replace('\r', '').strip() for x in data[i].initial_post]
            comments = [x.replace('\n', '').replace('\r', '').strip() for x in data[i].reply_sentences]
            selected_sentences = [x.replace('\n', '').replace('\r', '').strip() for x in data[i].reply_sentences]

            answers = [1 if x in selected_sentences else 0 for x in comments]
            post = [hF.tokenize_text(x) for x in post]

            comments = [hF.tokenize_text(x) for x in comments]
            human_summary = data[i].human_summary_1

            if use_back_translation is True:
                post_translated = [x.replace('\n', '').replace('\r', '').strip().split(' ') for x in data[i].initial_post_translated]
                comments_translated = [x.replace('\n', '').replace('\r', '').strip() for x in data[i].reply_sentences_translated]

                post_translated = [x.split(' ') for x in post_translated]
                all_comments_translated.append(comments_translated)
                all_posts_translated.append(post_translated)

            all_answers.append(answers)
            all_comments.append(comments)
            all_posts.append(post)
            all_human_summaries.append(human_summary)

        if use_back_translation is True:
            return all_posts, all_comments, all_answers, all_human_summaries, all_posts_translated, all_comments_translated
        else:
            return all_posts, all_comments, all_answers, all_human_summaries


    def batchify(self, all_posts, all_comments, all_answers, all_human_summaries, all_sentence_str, batch_size,
                 use_back_translation=False, all_posts_translated=None, all_comments_translated=None):
        comments_batches = []
        posts_batches = []
        answer_batches = []
        human_summary_batches = []
        sentences_str_batches = []

        posts_translated_batches = []
        comments_translated_batches = []

        print('Batchifying Data...')
        for i in tqdm(range(0, len(all_posts), batch_size)):
            answer_batch = []
            comments_batch = []
            post_batch = []
            human_summary_batch = []
            sentences_str_batch = []

            comments_translated_batch = []
            posts_translated_batch = []

            for j in range(i, i + batch_size):
                if j < len(all_posts):
                    post = all_posts[j]  # [x.replace('\n','').replace('\r','').strip() for x in data[j].initial_post]
                    comments = all_comments[j]  # [x.replace('\n','').replace('\r','').strip() for x in data[j].reply_sentences]
                    #                 selected_sentences = [x.replace('\n','').replace('\r','').strip() for x in data[j].selected_sentences]
                    answers = all_answers[j]  # [1 if x in selected_sentences else 0 for x in comments]
                    human_summary = all_human_summaries[j]
                    #                 post = [hF.tokenize_text(x) for x in post]
                    #                 comments = [hF.tokenize_text(x) for x in comments]

                    answer_batch.append(answers)
                    comments_batch.append(comments)
                    post_batch.append(post)
                    human_summary_batch.append(human_summary)
                    sentences_str_batch.append(all_sentence_str[j])

                    if use_back_translation is True and all_posts_translated is not None and all_comments_translated is not None:
                        comments_translated_batch.append(all_comments_translated[j])
                        posts_translated_batch.append(all_posts_translated[j])

            comments_batches.append(comments_batch)
            posts_batches.append(post_batch)
            answer_batches.append(answer_batch)
            human_summary_batches.append(human_summary_batch)
            sentences_str_batches.append(sentences_str_batch)

            if use_back_translation is True and all_posts_translated is not None and all_comments_translated is not None:
                comments_translated_batches.append(comments_translated_batch)
                posts_translated_batches.append(posts_translated_batch)

        if use_back_translation is True and all_posts_translated is not None and all_comments_translated is not None:
            return posts_batches, comments_batches, answer_batches, human_summary_batches, sentences_str_batches, posts_translated_batches, comments_translated_batches
        else:
            return posts_batches, comments_batches, answer_batches, human_summary_batches, sentences_str_batches


    def encode(self, data, word2id_dictionary):
        for index, doc in tqdm(enumerate(data)):
            data[index] = hF.encode_document(doc, word2id_dictionary)
        return data


    def pad(self, data_batches):
        print('padding Data...')
        max_sentences = []
        max_length = []
        no_padding_sentences = []
        no_padding_lengths = []
        for index, batch in tqdm(enumerate(data_batches)):
            num_sentences = [len(x) for x in batch]
            sentence_lengthes = [[len(x) for x in y] for y in batch]
            max_num_sentences = max(num_sentences)
            max_sentences_length = max([max(x) for x in sentence_lengthes])

            batch, no_padding_num_sentences = hF.pad_batch_with_sentences(batch, max_num_sentences)
            batch, no_padding_sentence_lengths = hF.pad_batch_sequences(batch, max_sentences_length)

            max_sentences.append(max_num_sentences)
            max_length.append(max_sentences_length)
            no_padding_sentences.append(no_padding_num_sentences)
            no_padding_lengths.append(no_padding_sentence_lengths)
            data_batches[index] = batch
        ##########################################
        return data_batches, max_sentences, max_length, no_padding_sentences, no_padding_lengths


    def pad_batch(self, data_batch):
        num_sentences = [len(x) for x in data_batch]
        sentence_lengthes = [[len(x) for x in y] for y in data_batch]
        max_num_sentences = max(num_sentences)
        max_sentences_length = max([max(x) for x in sentence_lengthes])

        data_batch, no_padding_num_sentences = hF.pad_batch_with_sentences(data_batch, max_num_sentences)
        data_batch, no_padding_sentence_lengths = hF.pad_batch_sequences(data_batch, max_sentences_length)

        ##########################################
        return data_batch, max_num_sentences, max_sentences_length, no_padding_num_sentences, no_padding_sentence_lengths


    def encode_and_pad(self, data_batches, word2id_dictionary):
        #################### Prepare Training data################
        print('Encoding Data...')
        max_sentences = []
        max_length = []
        no_padding_sentences = []
        no_padding_lengths = []
        for index, batch in tqdm(enumerate(data_batches)):
            batch = hF.encode_batch(batch, word2id_dictionary)

            num_sentences = [len(x) for x in batch]
            sentence_lengthes = [[len(x) for x in y] for y in batch]
            max_num_sentences = max(num_sentences)
            max_sentences_length = max([max(x) for x in sentence_lengthes])

            batch, no_padding_num_sentences = hF.pad_batch_with_sentences(batch, max_num_sentences)
            batch, no_padding_sentence_lengths = hF.pad_batch_sequences(batch, max_sentences_length)

            max_sentences.append(max_num_sentences)
            max_length.append(max_sentences_length)
            no_padding_sentences.append(no_padding_num_sentences)
            no_padding_lengths.append(no_padding_sentence_lengths)
            data_batches[index] = batch
        ##########################################
        return data_batches, max_sentences, max_length, no_padding_sentences, no_padding_lengths


    def encode_BERT(self, data, Bert_model_Path, device, bert_layers, batch_size):
        from pytorch_pretrained_bert import BertTokenizer, BertModel
        if not os.path.exists(Bert_model_Path):
            print('Bet Model not found.. make sure path is correct')
            return
        tokenizer = BertTokenizer.from_pretrained(Bert_model_Path)  # '../../pytorch-pretrained-BERT/bert_models/uncased_L-12_H-768_A-12/')
        model = BertModel.from_pretrained(Bert_model_Path)  # '../../pytorch-pretrained-BERT/bert_models/uncased_L-12_H-768_A-12/')
        model.eval()
        model.to(device)
        #################### Prepare Training data################
        print('Encoding Data using BERT...')
        max_sentences = []
        no_padding_sentences = []
        j = 0
        for j in tqdm(range(0, len(data), batch_size)):
            if j + batch_size < len(data):
                batch = data[j: j + batch_size]
            else:
                batch = data[j:]
            batch = hF.encode_batch_BERT(batch, model, tokenizer, device, bert_layers)

            for i, doc in enumerate(batch):
                data[j + i] = batch[i]

        ##########################################
        return data


    def pad_BERT(self, data_batches, bert_layers, bert_dims):
        print('Padding Data using BERT...')
        max_sentences = []
        no_padding_sentences = []
        for index, batch in tqdm(enumerate(data_batches)):
            num_sentences = [len(x) for x in batch]
            max_num_sentences = max(num_sentences)

            batch, no_padding_num_sentences = hF.pad_batch_with_sentences_BERT(batch, max_num_sentences, bert_layers, bert_dims)

            max_sentences.append(max_num_sentences)
            no_padding_sentences.append(no_padding_num_sentences)
            data_batches[index] = batch
        ##########################################
        return data_batches, max_sentences, None, no_padding_sentences, None


    def pad_batch_BERT(self, batch, bert_layers, bert_dims):
        num_sentences = [len(x) for x in batch]
        max_num_sentences = max(num_sentences)
        batch, no_padding_num_sentences = hF.pad_batch_with_sentences_BERT(batch, max_num_sentences, bert_layers, bert_dims)
        ##########################################
        return batch, max_num_sentences, None, no_padding_num_sentences, None


    def encode_and_pad_BERT(self, data_batches, Bert_model_Path, device, bert_layers, bert_dims):
        from pytorch_pretrained_bert import BertTokenizer, BertModel
        tokenizer = BertTokenizer.from_pretrained(Bert_model_Path)  # '../../pytorch-pretrained-BERT/bert_models/uncased_L-12_H-768_A-12/')
        model = BertModel.from_pretrained(Bert_model_Path)  # '../../pytorch-pretrained-BERT/bert_models/uncased_L-12_H-768_A-12/')
        model.eval()
        model.to(device)
        #################### Prepare Training data################
        print('Encoding Data using BERT...')
        max_sentences = []
        no_padding_sentences = []
        for index, batch in tqdm(enumerate(data_batches)):
            batch = hF.encode_batch_BERT(batch, model, tokenizer, device, bert_layers)
            # data_batches[index] = batch
            num_sentences = [len(x) for x in batch]
            max_num_sentences = max(num_sentences)

            batch, no_padding_num_sentences = hF.pad_batch_with_sentences_BERT(batch, max_num_sentences, bert_layers, bert_dims)

            max_sentences.append(max_num_sentences)
            no_padding_sentences.append(no_padding_num_sentences)
            data_batches[index] = batch
        ##########################################
        return data_batches, max_sentences, None, no_padding_sentences, None

    ######################################################


    def read_vocabulary(self, vocab_path):
        with open(vocab_path, "rb") as vocab_file:
            [word2id_dictionary, id2word_dictionary] = pickle.load(vocab_file)
        return word2id_dictionary, id2word_dictionary


    def tokenize_data(self, data):
        all_posts_translated = None
        all_comments_translated = None
        if self.params['use_back_translation'] is True:
            all_posts, all_comments, all_answers, all_human_summaries, all_posts_translated, all_comments_translated = self.tokenize(data, use_back_translation=True)
        else:
            all_posts, all_comments, all_answers, all_human_summaries = self.tokenize(data)

        all_sentence_str = []
        for index, comment in enumerate(all_comments):
            all_sentence_str.append(all_comments[index])

        return all_posts, all_comments, all_answers, all_human_summaries, all_sentence_str, all_posts_translated, all_comments_translated


    def encode_data(self, all_posts, all_comments, all_answers, all_human_summaries, all_sentence_str, all_posts_translated, all_comments_translated, word2id_dictionary):
        use_BERT = self.params['use_BERT']
        encoding_batch_size = self.params['encoding_batch_size']

        if use_BERT is True:
            all_comments = self.encode_BERT(all_comments, self.params['BERT_Model_Path'], self.params['device'], self.params['BERT_layers'], encoding_batch_size)
            all_posts = self.encode_BERT(all_posts, self.params['BERT_Model_Path'], self.params['device'], self.params['BERT_layers'], encoding_batch_size)
            if self.params['use_back_translation'] is True:
                all_comments_translated = self.encode_BERT(all_comments_translated, self.params['BERT_Model_Path'], self.params['device'], self.params['BERT_layers'], encoding_batch_size)
                all_posts_translated = self.encode_BERT(all_posts_translated, self.params['BERT_Model_Path'], self.params['device'], self.params['BERT_layers'], encoding_batch_size)
            else:
                all_comments_translated = None
                all_posts_translated = None

        else:
            all_comments = self.encode(all_comments, word2id_dictionary)
            all_posts = self.encode(all_posts, word2id_dictionary)
            if self.params['use_back_translation'] is True:
                all_comments_translated = self.encode(all_comments_translated, word2id_dictionary)
                all_posts_translated = self.encode(all_posts_translated, self.params['BERT_Model_Path'], word2id_dictionary)
            else:
                all_comments_translated = None
                all_posts_translated = None

        return all_posts, all_comments, all_answers, all_human_summaries, all_sentence_str, all_comments_translated, all_posts_translated


    def preprcoess(self, data):
        word2id_dictionary, id2word_dictionary = self.read_vocabulary('./SummaRuNNer/checkpoints/vocab.pickle')
        all_posts, all_comments, all_answers, all_human_summaries, all_sentence_str, all_posts_translated, all_comments_translated = self.tokenize_data(data)

        '''
        Reduce data size to max sentences and max sequence length
        '''

        max_num_sentences = self.params['Global_max_num_sentences']
        max_sentence_length = self.params['Global_max_sequence_length']

        if max_num_sentences is not None:
            for index, comment in enumerate(all_comments):
                all_comments[index] = all_comments[index][:max_num_sentences]
                all_sentence_str[index] = all_sentence_str[index][:max_num_sentences]

            for index, answer in enumerate(all_answers):
                all_answers[index] = all_answers[index][:max_num_sentences]

            if self.params['use_back_translation'] is True:
                for index, comment in enumerate(all_comments_translated):
                    all_comments_translated[index] = all_comments_translated[index][:max_num_sentences]

        if max_sentence_length is not None:
            for index, comment in enumerate(all_comments):
                for index_2, sent in enumerate(comment):
                    all_comments[index][index_2] = all_comments[index][index_2][:max_sentence_length]

            if self.params['use_back_translation'] is True:
                for index, comment in enumerate(all_comments_translated):
                    for index_2, sent in enumerate(comment):
                        all_comments_translated[index][index_2] = all_comments_translated[index][index_2][:max_sentence_length]

        all_posts, all_comments, all_answers, all_human_summaries, all_sentence_str, all_comments_translated, all_posts_translated = self.encode_data(all_posts, all_comments, all_answers, all_human_summaries, all_sentence_str, all_posts_translated,
                                                                                                                                               all_comments_translated, word2id_dictionary)

        return all_posts, all_comments, all_answers, all_human_summaries, all_sentence_str, all_comments_translated, all_posts_translated





