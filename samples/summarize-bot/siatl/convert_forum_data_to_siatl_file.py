import pickle
from lxml import etree as ET
from tqdm import tqdm


def read_data(data_file_path):
    train_data = []
    val_data = []
    test_data = []

    word2id_dictionary = {}
    id2word_dictionary = {}

    tree = ET.parse(data_file_path)
    root = tree.getroot()
    words = []

    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15

    all_data_length = len(root)
    num_train = int(train_ratio * all_data_length)
    num_val = int(val_ratio * all_data_length)
    num_test = all_data_length - num_train - num_val#  test_ratio * all_data_length

    post_index = 0
    sentences_text = []
    post_ids = []
    target = []
    initial_comments = []

    train_ids = []
    val_ids = []
    test_ids = []

    global_sentence_idx = 0
    for index, post_item in tqdm(enumerate(root)):
        post_id = post_item.tag
        initial_comment = [item.text for item in post_item.findall('Body')][0].split('\n')
        # for sentence in initial_comment:
        #     words += hF.tokenize_text(sentence)

        summary_1 = [item.text for item in post_item.findall('Summary_1')][0]
        summary_2 = [item.text for item in post_item.findall('Summary_2')][0]

        selected_sentences = [item.text for item in post_item.findall('selected_sentences')][0].split('\n')
        initial_comment_str = ' '.join(initial_comment)
        sentences = initial_comment
        for comment_item in post_item.findall('Comment'):
            comment_body = [item.text for item in comment_item.findall('Body')][0].split('\n')
            # for sentence in comment_body:
            sentences += [x.strip() for x in comment_body]

        if index < num_train:
            num_found = 0
            for sentence in sentences:
                sentences_text.append(sentence)
                post_ids.append(post_index)
                initial_comments.append(initial_comment_str)
                if sentence in selected_sentences:
                    target.append({'label':1})
                    num_found += 1
                else:
                    target.append({'label':0})
                train_ids.append(global_sentence_idx)
                global_sentence_idx += 1
            if num_found == 0:
                print('Here')

        elif num_train <= index < num_train + num_val:
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
                val_ids.append(global_sentence_idx)
                global_sentence_idx += 1
            if num_found == 0:
                print('Here')
        else:
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
        post_index+=1

    save_dic = {'info':target, 'texts':sentences_text, 'val_ind':val_ids, 'train_ind':train_ids, 'test_ind':test_ids, 'post_ids':post_ids, 'posts':initial_comments}
    print('Saving data...')
    with open('./datasets/Forum/raw_forum.pickle', "wb") as output_file:
        pickle.dump(save_dic, output_file)



def read_cnn_dm_data(data_parent_dir):
    import json
    import os
    #
    # train_data = []
    # val_data = []
    # test_data = []
    #
    # test_count = 0
    # train_count = 0
    # val_count = 0


    post_index = 0
    sentences_text = []
    post_ids = []
    target = []
    human_summaries = []
    initial_comments = []
    train_ids = []
    val_ids = []
    test_ids = []

    global_sentence_idx = 0
    post_index = 0
    for data_dir in ['test', 'train', 'val']:
        file_names = os.listdir(data_parent_dir + '//' + data_dir)
        for fname in tqdm(file_names):
            with open(data_parent_dir + '//' + data_dir + '//' + fname) as json_file:
                data = json.load(json_file)
                article = [x.strip() for x in data['article']]
                if len(article) < 2:
                    continue
                human_summary = ' '.join([x.strip() for x in data['abstract']])
                summary = [article[x] for x in data['extracted']]

                article = [x for x in article if x.strip() != '' and len(x) > 2]
                if len(article) <= 1:
                    continue
                '''
                end Data filtering
                '''

                initial_comment_str = article[0]
                if data_dir == 'train':
                    num_found = 0
                    for sentence in article:
                        sentences_text.append(sentence)
                        post_ids.append(post_index)
                        initial_comments.append(initial_comment_str)
                        if sentence in summary:
                            target.append({'label': 1})
                            num_found += 1
                        else:
                            target.append({'label': 0})
                        train_ids.append(global_sentence_idx)
                        global_sentence_idx += 1
                        human_summaries.append(human_summary)
                    if num_found == 0:
                        print('Here')

                elif data_dir == 'test':
                    num_found = 0
                    for sentence in article:
                        sentences_text.append(sentence)
                        post_ids.append(post_index)
                        initial_comments.append(initial_comment_str)
                        if sentence in summary:
                            target.append({'label': 1})
                            num_found += 1
                        else:
                            target.append({'label': 0})
                        test_ids.append(global_sentence_idx)
                        global_sentence_idx += 1
                        human_summaries.append(human_summary)
                    if num_found == 0:
                        print('Here')


                elif data_dir == 'val':
                    num_found = 0
                    for sentence in article:
                        sentences_text.append(sentence)
                        post_ids.append(post_index)
                        initial_comments.append(initial_comment_str)
                        if sentence in summary:
                            target.append({'label': 1})
                            num_found += 1
                        else:
                            target.append({'label': 0})
                        val_ids.append(global_sentence_idx)
                        global_sentence_idx += 1
                        human_summaries.append(human_summary)
                    if num_found == 0:
                        print('Here')
                post_index+=1

    save_dic = {'info':target, 'texts':sentences_text, 'val_ind':val_ids, 'train_ind':train_ids, 'test_ind':test_ids, 'post_ids':post_ids, 'human_summaries':human_summaries, 'posts':initial_comments}
    print('Saving data...')
    with open('./datasets/CNN/raw_cnn.pickle', "wb") as output_file:
        pickle.dump(save_dic, output_file)


read_data('../SummRuNNer/forum_data/data_V2/Parsed_Data.xml')
read_cnn_dm_data('../SummRuNNer/cnn_data/finished_files/')