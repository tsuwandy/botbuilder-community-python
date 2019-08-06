import helpers.email_cleaner as ecleaner
from tqdm import tqdm as tqdm


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

    def convert_data(self, documents, posts):
        post_index = 0
        sentences_text = []
        post_ids = []
        target = []
        initial_comments = []

        train_ids = []
        val_ids = []
        test_ids = []

        global_sentence_idx = 0

        for document, post in tqdm(zip(documents, posts)):
            # post_id = post_item.tag

            post_sentences = post.replace('\r', '').split('\n')
            post_sentences = [x.strip() for x in post_sentences if x.strip() != '' and len(x.strip()) > 1]

            comments_sentences = document.replace('\r', '').split('\n')
            comments_sentences = [x.strip() for x in comments_sentences if x.strip() != '' and len(x.strip()) > 1]

            selected_sentences = comments_sentences
            initial_comment_str = ' '.join(post_sentences)
            sentences = comments_sentences #post_sentences +

            num_found = 0
            for sentence in sentences:
                sentences_text.append(sentence)
                post_ids.append(post_index)
                initial_comments.append(initial_comment_str)
                if sentence in selected_sentences:
                    target.append({'label': 1})
                    num_found += 1
                else:
                    target.append({'label': 0})
                test_ids.append(global_sentence_idx)
                global_sentence_idx += 1
            if num_found == 0:
                print('Here')
            post_index += 1

        save_dic = {'info': target, 'texts': sentences_text, 'val_ind': val_ids, 'train_ind': train_ids, 'test_ind': test_ids, 'post_ids': post_ids, 'posts': initial_comments}
        return save_dic

    def summarize(self, documents, posts, max_sents):
        import argparse

        import sys
        import os

        sys.path.insert(0, os.path.join(os.path.dirname(
            os.path.realpath(__file__)), "./"))

        from siatl.models.sum_clf import sum_clf_test
        # from models.sent_clf_no_aux import sent_clf_no_aux
        from siatl.utils.data_parsing import load_dataset
        from siatl.utils.opts import train_options

        parser = argparse.ArgumentParser()
        parser.add_argument("-i", "--input", required=False,
                            default='forum_bi_coatt_att_combined_aux_ft_gu.yaml',
                            help="config file of input data")
        parser.add_argument("-dv", "--device_to_use", required=False, default='auto', help="device for training")
        parser.add_argument("-t", "--transfer", action='store_true', required=False, default=True, help="transfer from pretrained language model or train a randomly initialized model")
        parser.add_argument("-a", "--aux_loss", action='store_true', required=False, default=True,
                            help="add an auxiliary LM loss to the transferred model"
                                 "or simply transfer a LM to a classifier"
                                 " and fine-tune")
        parser.add_argument("-o", "--output_dir", default='./output/forum/',
                            help="The output Directory")
        parser.add_argument("-j", "--job", default='Test', help="Train or Test")
        parser.add_argument("-tc", "--test_checkpoint_name", default='forum_bi_coatt_att_combined_aux_ft_gu_67', help="checkpoint to use for testing")

        args = parser.parse_args()
        input_config = args.input
        transfer = args.transfer
        aux_loss = args.aux_loss
        device_to_use = args.device_to_use
        opts, config = train_options(input_config, parser, device_to_use)

        output_dir = args.output_dir  # './output/forum/'

        test_checkpoint_name = args.test_checkpoint_name

        data = self.convert_data(documents, posts)
        test = [(data['texts'][x], data['info'][x]["label"], data['post_ids'][x], data['posts'][x]) for x in
                 data['test_ind']]

        X_test = [x[0] for x in test]
        y_test = [x[1] for x in test]
        pids_test = [x[2] for x in test]
        posts = [x[3] for x in test]
        dataset = [X_test, y_test, posts, pids_test, None]

        print('Testing..............')
        summaries = sum_clf_test(dataset=dataset, config=config, opts=opts, transfer=True, output_dir=output_dir, checkpoint_name=test_checkpoint_name)

        return summaries