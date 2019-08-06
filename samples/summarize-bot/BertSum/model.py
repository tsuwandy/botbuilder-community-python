import BertSum.data_handler as data_handler
import helpers.email_cleaner as ecleaner

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


    def summarize(self, documents, posts, max_sents):
        # print('Preprecesing...')
        import os
        if not os.path.exists('./BertSum/temp_dir'):
            os.mkdir('./BertSum/temp_dir')
        if not os.path.exists('./BertSum/temp_dir/raw/'):
            os.mkdir('./BertSum/temp_dir/raw/')
        if not os.path.exists('./BertSum/temp_dir/bert/'):
            os.mkdir('./BertSum/temp_dir/bert/')
        if not os.path.exists('./BertSum/temp_dir/json/'):
            os.mkdir('./BertSum/temp_dir/json/')
        if not os.path.exists('./BertSum/temp_dir/models/'):
            os.mkdir('./BertSum/temp_dir/models/')
        if not os.path.exists('./BertSum/temp_dir/results/'):
            os.mkdir('./BertSum/temp_dir/results/')
        if not os.path.exists('./BertSum/temp_dir/results/bert_classifier/'):
            os.mkdir('./BertSum/temp_dir/results/bert_classifier/')
        if not os.path.exists('./BertSum/temp_dir/tokenized/'):
            os.mkdir('./BertSum/temp_dir/tokenized/')
        if not os.path.exists('./BertSum/temp_dir/urls/'):
            os.mkdir('./BertSum/temp_dir/urls/')
		########################
        documents = self.preprocess(documents)
        data_handler.write_to_story_files(documents, './BertSum/temp_dir/raw/', 'test')
        data_handler.convert_data_summarize('/BertSum/temp_dir/')
        summaries = data_handler.read_output('/BertSum/temp_dir/')
        return summaries

# from nltk import sent_tokenize, word_tokenize
# eSummarizer = ExtractiveSummarizer()
# strDoc = 'Hi , Heading to NY shortly for a couple of weeks . We were thinking of renting cars and traveling out of the city for a few days . We really dont have much of a plan as to where we would like to go . We do have friends in Washington DC who we could visit , but then again a trip to Boston tempts . But more intriguing would be a trip to someplace a little more rural , and get out of the city for a couple of days . I do n\'t know if it foolish to think this would be possible within driving distance from NY , but if anyone on the forum could add any other recommendations to the mix , it would be greatly appreciated . Morgan , It is not foolish to think that there are places with 90 miles of NYC to put you in a country setting in NY State ... ..A nice day trip or overnight trip would be exploring the Hudson Valley Region on NY ... ... The Mid-Hudson Valley is one of those spots that would offer you a good base for an overnight or two . This region has many mansions to tour { Roosevelt/Vanderbuilt/Mills Mansion and Locust Grove } , please do look up when they are open since this is the off season . The Culinary Institute of American { Hyde Park NY } offers a great dining option ... this is a school that over 4 dining restaurants { again check days that this is open } . The area also offers great places to hike , cross country ski , snow shoe etc ... . Lots of great restaurants { Many run by Culinary Instit alumni } . The Would { Highland NY } . Beso Restaurant { New Paltz NY } . Culinary Instit of America { Hyde Park NY } . Neat towns to explore . New Paltz/Rhinebeck/Cold Spring/Millbrook etc ... ... Best of Luck with your travels , hope this helps ... . Boston is also a great city if you have not been before ... ..You can also get there rather cheaply with several bus services ... or by train if you choose not to rent a car ... Many thanks for the advise innladyHighlandNy . Sounds great and definitely interesting me . If I have my geography correct , I might take in a trip to the Woodbury Common outlet while I \'m up that part of the state . ( please apologise if I am miles out ! ! ! ) Going to try to take it in . Many thanks . Any other recommendations ? Depends on time of year , but there is lots to do close by to NYC come the spring . Mansions , gardens , resturants .'
# documents = [strDoc, strDoc]
# documents = ['\n'.join([' '.join(word_tokenize(y)) for y in sent_tokenize(x)]) for x in documents]
# summaries = eSummarizer.summarize(documents, 1)