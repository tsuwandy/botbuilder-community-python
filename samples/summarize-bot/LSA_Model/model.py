import helpers.tokenizer as tkzr
import spacy
from helpers import url_extractor as url_extractor
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import helpers.email_cleaner as ecleaner

from nltk import word_tokenize


class ExtractiveSummarizer(object):
    '''
    Other encode options are LSA-own, w2vec
    '''
    def __init__(self):
        ## cluster based parameters ####
        self.use_phrases = False
        self.clustering = 'kmeans'
        self.distance_metric = 'cos'
        ################################
        ######### Tdidf based parameters
        self.max_sents = 2
        self.similarity_threshold = 0.9
        ################################
        self.nlp = spacy.load('en_core_web_lg')

        self.remove_urls = True
        self.remove_signature = True
        self.use_stanford = False
        # self.encode_method = encode_method

        self.lsa_model_type = 'wiki'
        self.lsa_model_path = './aux_data/'
        self.lsa_model = None
        self.lsa_vectorizer = None
        self.lsa_dims = 150
        # self.LSA_dictionary = None

    def init_model(self, model_type):
        self.lsa_model_type = model_type
        if model_type == 'wiki':
            lsa_model, _ = self.load_pretrained_model(self.lsa_model_path + self.lsa_model_type + '/')
        else:
            lsa_model, vectorizer = self.load_pretrained_model(self.lsa_model_path + self.lsa_model_type + '/')
            self.lsa_vectorizer = vectorizer
            self.lsa_model = lsa_model


    def load_pretrained_model(self, path):
        if self.lsa_model_type == 'wiki':
            import gensim
            import pickle
            import os

            if os.path.exists(path + '/lsa.model'):
                with open(path + '/wiki.model','rb') as f:
                    lsi = pickle.load(f)
            else:
                print('Loading wiki data')
                id2word = gensim.corpora.Dictionary.load_from_text(path + '/_wordids.txt')
                mm = gensim.corpora.MmCorpus(path + '/_bow.mm')#wiki_en_tfidf.mm')
                print('Training Wiki model....')
                lsi = gensim.models.lsimodel.LsiModel(corpus=mm, id2word=id2word, num_topics=300)
                self.lsa_model = lsi
                print('Saving Wiki model....')
                with open(path + '/wiki.model','wb') as f:
                    pickle.dump(lsi,f)
            self.lsa_model = lsi
            return lsi, None
        else:
            import pickle
            with open(path + '/lsa.model', "rb") as output_file:
                [lsa, vectorizer] = pickle.load(output_file)
            return lsa, vectorizer

    def preprocess(self, docs):
        p_docs = []
        for doc in docs:
            if self.remove_signature:
                doc = ecleaner.clean_email(doc)
            if not doc.replace('\n', '').replace('\r', '').strip() == '':
                p_docs.append(doc)
        return p_docs

    def encode_lsa(self, docs):
        if self.lsa_model_type == 'wiki':
            documents_sentences = []
            documents_vectors = []
            '''Loop over documents and for each document tokenize into sentences
            For each sentence get the average of words representation as sentence representation
            '''
            for doc in docs:
                sentences, _ = tkzr.tokenize_text(doc, self.use_stanford)
                vectors = []
                encoded_sentences = []
                dictionary = self.lsa_model.id2word
                for sentence in sentences:
                    vector = lsa_model[dictionary.doc2bow(word_tokenize(sentence))]  # apply model to BoW document
                    vector = [x[1] for x in vector]
                    encoded_sentences.append(sentence)
                    vectors.append(vector)
                documents_sentences.append(encoded_sentences)
                documents_vectors.append(vectors)
            return documents_sentences, documents_vectors
        else:
            documents_sentences = []
            documents_vectors = []
            '''Loop over documents and for each document tokenize into sentences
            For each sentence get the average of words representation as sentence representation
            '''
            for doc in docs:
                sentences, _ = tkzr.tokenize_text(doc, self.use_stanford)
                vectors = []

                encoded_sentences = []
                for sentence in sentences:
                    X_test_tfidf = self.lsa_vectorizer.transform([sentence])
                    X_test_lsa = self.lsa_model.transform(X_test_tfidf)
                    encoded_sentences.append(sentence)
                    vectors.append(X_test_lsa[0].tolist())
                documents_sentences.append(encoded_sentences)
                documents_vectors.append(vectors)
            return documents_sentences, documents_vectors


    def summarize(self, documents, posts, max_sents):
        # print('Preprecesing...')
        docs = self.preprocess(documents)

        n_docs = len(docs)
        summaries = [None] * n_docs
        # print('Starting to encode...')

        doc_sentences, doc_vectors = self.encode_lsa(docs)
        # print('Splitting into sentences and Encoding Finished')
        for i in range(n_docs):
            doc_vector = doc_vectors[i]
            doc_vector = [x for x in doc_vector if len(x) > 0]
            if len(doc_vector) == 0:
                summaries[i] = ' '.join(doc_sentences[i])
            else:
                n_clusters = int(np.ceil(len(doc_vector) ** 0.5))
                kmeans = KMeans(n_clusters=n_clusters, random_state=0)
                kmeans = kmeans.fit(doc_vector)
                avg = []
                closest = []
                for j in range(n_clusters):
                    idx = np.where(kmeans.labels_ == j)[0]
                    avg.append(np.mean(idx))
                closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, doc_vector)
                ordering = sorted(range(n_clusters), key=lambda k: avg[k])
                summaries[i] = ' '.join([doc_sentences[i][closest[idx]] for idx in ordering])
        # print('Clustering Finished')
        for index, summary in enumerate(summaries):
            sentences, _ = tkzr.tokenize_text(summary, self.use_stanford)
            summaries[index] = ' '.join(sentences[:max_sents])
        return summaries


if __name__ == '__main__':
    esum = ExtractiveSummarizer('LSA')
    lsa_model = esum.load_pretrained_model('../aux_data/wiki/')

