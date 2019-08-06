from SummaRuNNer.data_handler import preprocessor, ThreadObject
import helpers.email_cleaner as ecleaner

import torch.nn as nn
import torch
# import data_loader as dL
import SummaRuNNer.model_loader as mL
import SummaRuNNer.trainer as trainer
import pickle
import os
import glob
import torch
from torch.utils import data
from tqdm import tqdm as tqdm
import codecs

class ExtractiveSummarizer(object):
    '''
    Other encode options are LSA-own, w2vec
    '''

    def __init__(self):
        ################################
        ######### Tdidf based parameters
        self.max_sents = 2
        ################################
        # self.nlp = spacy.load('en_core_web_lg')

        self.remove_urls = True
        self.remove_signature = True
        self.use_stanford = False
        self.parser = None
        self.stemmer = None
        self.summarizer = None

        self.params = {}
        ############ Data params
        # params['DATA_Path'] = '/mnt/Summarization/SummRunner_V2/cnn_data/finished_files/'  # './forum_data/data_V2/Parsed_Data.xml'
        # params['data_set_name'] = 'forum'
        ############ Model params
        self.params['vocab_path'] = ''
        self.params['use_coattention'] = True
        self.params['use_BERT'] = True
        self.params['BERT_Model_Path'] = './SummaRuNNer/pytorch-pretrained-BERT/bert_models/uncased_L-12_H-768_A-12/'
        self.params['BERT_embedding_size'] = 768
        self.params['BERT_layers'] = [-1, -2]

        self.params['embedding_size'] = 64
        self.params['hidden_size'] = 128
        self.params['batch_size'] = 8
        self.params['max_num_sentences'] = 20
        self.params['lr'] = 0.001
        self. params['vocab_size'] = 70000
        self.params['use_back_translation'] = False
        self.params['back_translation_file'] = None
        self.params['Global_max_sequence_length'] = 75
        self.params['Global_max_num_sentences'] = 30
        self.params['encoding_batch_size'] = 32
        ############ logging params
        self.params['load_model'] = True
        self.params['load_model_path'] = './SummaRuNNer/checkpoints/model_forum_30_75_coatt_bert_2_10.pkl'
        ############ device
        self.params['device'] = torch.device('cpu')

        self.preprocessor = preprocessor(params=self.params)

    def init_model(self, model_type):
        print('init')

    def preprocess(self, docs):
        preprocessed_docs = []
        for doc in docs:
            if self.remove_signature:
                doc = ecleaner.clean_email(doc)
            if not doc.replace('\n', '').replace('\r', '').strip() == '':
                preprocessed_docs.append(doc)
        return preprocessed_docs

    def evaluate(self, posts, comments, answers, human_summaries, sentence_str, comments_translated, posts_translated, summRunnerModel):
        # output_dir = params['output_dir'] + '/test_{}/'.format(epoch)
        summRunnerModel.eval()
        sample_index = 0
        # print('Loading testing data part {}'.format(part))
        if self.params['use_back_translation'] is True:
            posts_batches, comments_batches, answer_batches, human_summary_batches, sentences_str_batches, posts_translated_batches, comments_translated_batches = self.preprocessor.batchify(posts, comments, answers,
                                                                                                                                                                                    human_summaries,
                                                                                                                                                                                    sentence_str, self.params['batch_size'],
                                                                                                                                                                                    use_back_translation=self.params['use_back_translation'],
                                                                                                                                                                                    all_posts_translated=comments_translated,
                                                                                                                                                                                    all_comments_translated=posts_translated)
            pbar = tqdm(zip(posts_batches, comments_batches, human_summary_batches, answer_batches, sentences_str_batches, posts_translated_batches, comments_translated_batches))
        else:
            posts_batches, comments_batches, answer_batches, human_summary_batches, sentences_str_batches = self.preprocessor.batchify(posts, comments, answers,
                                                                                                                             human_summaries,
                                                                                                                             sentence_str, self.params['batch_size'],
                                                                                                                             use_back_translation=self.params['use_back_translation'],
                                                                                                                             all_posts_translated=comments_translated,
                                                                                                                             all_comments_translated=posts_translated)
            pbar = tqdm(zip(posts_batches, comments_batches, human_summary_batches, answer_batches, sentences_str_batches, posts_batches, comments_batches))
        batch_index = 1
        all_predicted_sentences = []
        for post_batch, comment_batch, human_summary_batch, answer_batch, sentence_str_batch, post_translated_batch, comment_translated_batch in pbar:
            pbar.set_description("Evaluating using testing data {}/{}".format(batch_index, len(posts_batches)))
            batch_index += 1
            if self.params['use_BERT'] is True:
                comment_batch, max_sentences, max_length, no_padding_sentences, no_padding_lengths = self.preprocessor.pad_batch_BERT(comment_batch, self.params['BERT_layers'], self.params['BERT_embedding_size'])
                post_batch, posts_max_sentences, posts_max_length, posts_no_padding_sentences, posts_no_padding_lengths = self.preprocessor.pad_batch_BERT(post_batch, self.params['BERT_layers'], self.params['BERT_embedding_size'])

            else:
                comment_batch, max_sentences, max_length, no_padding_sentences, no_padding_lengths = self.preprocessor.pad_batch(comment_batch)
                post_batch, posts_max_sentences, posts_max_length, posts_no_padding_sentences, posts_no_padding_lengths = self.preprocessor.pad_batch(post_batch)

            predicted_sentences, target_sentences, human_summaries = trainer.test_batch(summRunnerModel, self.params['device'], post_batch, comment_batch, answer_batch, human_summary_batch, sentence_str_batch,
                                                                                        max_sentences, max_length, no_padding_sentences, no_padding_lengths,
                                                                                        posts_max_sentences, posts_max_length, posts_no_padding_sentences, posts_no_padding_lengths, self.params['use_BERT'])

            for predicted, target, human in zip(predicted_sentences, target_sentences, human_summaries):
                all_predicted_sentences.append(predicted)
        return all_predicted_sentences

    def convert_to_thread_obj(self, post, comments):
        post_sentences = post.replace('\r', '').split('\n')
        post_sentences = [x.strip() for x in post_sentences if x.strip() != '' and len(x.strip()) > 1]

        comments_sentences = comments.replace('\r', '').split('\n')
        comments_sentences = [x.strip() for x in comments_sentences if x.strip() != '' and len(x.strip()) > 1]

        thobj = ThreadObject('0', 'id_1', post_sentences)
        thobj.add_comments(comments_sentences)
        thobj.set_selected_sentences(comments[0])
        thobj.human_summary_1 = 'Human summary'
        thobj.human_summary_2 = 'Human summary'

        return thobj

    def summarize(self, documents, posts, max_sents):
        word2id_dictionary, id2word_dictionary = self.preprocessor.read_vocabulary('./SummaRuNNer/checkpoints/vocab.pickle')

        data = []
        for x, y in zip(posts, documents):
            thobj = self.convert_to_thread_obj(x, y)
            data.append(thobj)

        vocab_size = len(word2id_dictionary)
        max_number_sentences = self.params['Global_max_num_sentences'] + 1  # max([max(train_max_sentences), max(val_max_sentences), max(test_max_sentences)]) + 1

        self.params['max_num_sentences'] = max_number_sentences
        self.params['vocab_size'] = vocab_size

        summRunnerModel = mL.init_model(self.params, vocab_size)
        optimizer = torch.optim.Adam(summRunnerModel.parameters(), lr=self.params['lr'])  # 1e-3)

        if self.params['load_model'] is True:
            print('Loading Model from {}'.format(self.params['load_model_path']))
            summRunnerModel, optimizer = mL.load_model(optimizer=optimizer, path=self.params['load_model_path'], device=self.params['device'])

        if 'cuda' in self.params['device'].type:
            summRunnerModel.cuda()

        summRunnerModel.eval()
        posts, comments, answers, human_summaries, sentence_str, comments_translated, posts_translated = self.preprocessor.preprcoess(data)
        all_predicted_sentences = self.evaluate(posts, comments, answers, human_summaries, sentence_str, comments_translated, posts_translated, summRunnerModel)
        return all_predicted_sentences



# from nltk import sent_tokenize, word_tokenize
# eSummarizer = ExtractiveSummarizer()
# strDoc = 'Hi , Heading to NY shortly for a couple of weeks . We were thinking of renting cars and traveling out of the city for a few days . We really dont have much of a plan as to where we would like to go . We do have friends in Washington DC who we could visit , but then again a trip to Boston tempts . But more intriguing would be a trip to someplace a little more rural , and get out of the city for a couple of days . I do n\'t know if it foolish to think this would be possible within driving distance from NY , but if anyone on the forum could add any other recommendations to the mix , it would be greatly appreciated . Morgan , It is not foolish to think that there are places with 90 miles of NYC to put you in a country setting in NY State ... ..A nice day trip or overnight trip would be exploring the Hudson Valley Region on NY ... ... The Mid-Hudson Valley is one of those spots that would offer you a good base for an overnight or two . This region has many mansions to tour { Roosevelt/Vanderbuilt/Mills Mansion and Locust Grove } , please do look up when they are open since this is the off season . The Culinary Institute of American { Hyde Park NY } offers a great dining option ... this is a school that over 4 dining restaurants { again check days that this is open } . The area also offers great places to hike , cross country ski , snow shoe etc ... . Lots of great restaurants { Many run by Culinary Instit alumni } . The Would { Highland NY } . Beso Restaurant { New Paltz NY } . Culinary Instit of America { Hyde Park NY } . Neat towns to explore . New Paltz/Rhinebeck/Cold Spring/Millbrook etc ... ... Best of Luck with your travels , hope this helps ... . Boston is also a great city if you have not been before ... ..You can also get there rather cheaply with several bus services ... or by train if you choose not to rent a car ... Many thanks for the advise innladyHighlandNy . Sounds great and definitely interesting me . If I have my geography correct , I might take in a trip to the Woodbury Common outlet while I \'m up that part of the state . ( please apologise if I am miles out ! ! ! ) Going to try to take it in . Many thanks . Any other recommendations ? Depends on time of year , but there is lots to do close by to NYC come the spring . Mansions , gardens , resturants .'
# documents = [strDoc, strDoc]
# documents = ['\n'.join([' '.join(word_tokenize(y)) for y in sent_tokenize(x)]) for x in documents]
# summaries = eSummarizer.summarize(documents, 1)