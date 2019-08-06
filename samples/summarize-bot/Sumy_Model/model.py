import spacy
import helpers.email_cleaner as ecleaner
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer as Summarizer
from sumy.summarizers.lex_rank import LexRankSummarizer as lxrSummarizer
from sumy.summarizers.edmundson import EdmundsonSummarizer as edmSummarizer
from sumy.summarizers.luhn import LuhnSummarizer as luhSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer as texrSummarizer
from sumy.summarizers.kl import KLSummarizer as klSummarizer

from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words


class ExtractiveSummarizer(object):
    '''
    Other encode options are LSA-own, w2vec
    '''

    def __init__(self):
        ################################
        ######### Tdidf based parameters
        self.max_sents = 2
        ################################
        self.nlp = spacy.load('en_core_web_lg')

        self.remove_urls = True
        self.remove_signature = True
        self.use_stanford = False
        self.parser = None
        self.stemmer = None
        self.summarizer = None

    def init_model(self, model_type):
        self.stemmer = Stemmer('english')
        if model_type == 'lsa':
            self.summarizer = Summarizer(self.stemmer)
        elif model_type == 'lexrank':
            self.summarizer = lxrSummarizer(self.stemmer)
        elif model_type == 'textrank':
            self.summarizer = texrSummarizer(self.stemmer)
        elif model_type == 'luhn':
            self.summarizer = luhSummarizer(self.stemmer)
        elif model_type == 'kl':
            self.summarizer = klSummarizer(self.stemmer)
        elif model_type == 'edmun':
            self.summarizer = edmSummarizer(self.stemmer)

    def preprocess(self, docs):
        preprocessed_docs = []
        for doc in docs:
            if self.remove_signature:
                doc = ecleaner.clean_email(doc)
            if not doc.replace('\n', '').replace('\r', '').strip() == '':
                preprocessed_docs.append(doc)
        return preprocessed_docs


    def summarize(self, documents, posts, max_sents):
        # print('Preprecesing...')
        documents = self.preprocess(documents)
        summaries = []
        for doc in documents:
            doc = doc.replace('\n', ' ').replace('  ', ' ').replace('\r', ' ').replace('  ', ' ')
            parser = PlaintextParser.from_string(doc, Tokenizer('english'))
            summary = ''
            summary_sents = self.summarizer(parser.document, max_sents)
            for sentence in summary_sents:
                # print(sentence)
                summary += sentence._text + ' '
            summaries.append(summary.strip())
        return summaries


if __name__ == '__main__':
    print('main')
