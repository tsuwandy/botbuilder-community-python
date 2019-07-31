import os
import corenlp
from nltk import sent_tokenize
from nltk import word_tokenize

os.environ['CORENLP_HOME'] = 'C:/Data/Stanford/stanford-corenlp-full-2018-10-05'


def tokenize_text(text, use_stanford):
    if use_stanford is True:
        sentences = []
        words = []
        with corenlp.CoreNLPClient(annotators="tokenize ssplit".split()) as client:
            ann = client.annotate(text)
            for sentence in ann.sentence:
                sentence_str = corenlp.to_text(sentence)
                sentences.append(sentence_str)

                for i in range(len(sentence.token)):
                    word = sentence.token[i].word
                    words.append(word)
        return sentences, words
    else:
        sentences = sent_tokenize(text)
        words = word_tokenize(text)

        return sentences, words

