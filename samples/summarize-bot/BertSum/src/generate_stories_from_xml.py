from lxml import etree as ET
from tqdm import tqdm as tqdm_notebook
import codecs


class ForumItem(object):
    def __init__(self, owner_id, initial_comment_sentences, key, title_sentences, summary_1, summary_2, selected_sentences):
        self.owner_id = owner_id
        self.initial_comment_sentences = initial_comment_sentences
        self.key = key
        self.title_sentences = title_sentences
        self.comments = []
        self.summary_1 = summary_1
        self.summary_2 = summary_2

        self.selected_sentences_indcies = []
        self.selected_sentences_text = selected_sentences

        # self.annotated = False

    def add_comments(self, comments):
        self.comments += comments


def read_github_data(data_file_path):
    train_data = []
    val_data = []
    test_data = []

    word2id_dictionary = {}
    id2word_dictionary = {}

    tree = ET.parse(data_file_path)
    root = tree.getroot()
    words = []

    train_ratio = 0
    val_ratio = 0
    test_ratio = 1

    all_data_length = len(root)
    num_train = int(train_ratio * all_data_length)
    num_val = int(val_ratio * all_data_length)
    num_test = all_data_length - num_train - num_val#  test_ratio * all_data_length

    for index, post_item in tqdm_notebook(enumerate(root)):
        post_id = post_item.tag
        title = [item.text for item in post_item.findall('Title')][0].split('\n')

        owner_id = [item.text for item in post_item.findall('Owner')][0]

        initial_comment = [item.text for item in post_item.findall('Body')][0].replace('\r', '').split('\n')
        initial_comment = [x.strip() for x in initial_comment if x.strip() != '']

        summary_1 = initial_comment[0]
        summary_2 = initial_comment[0]
        selected_sentences = [initial_comment[0]]

        rthread = ForumItem(post_id, initial_comment, post_id, title, summary_1, summary_2, selected_sentences)

        for comment_item in post_item.findall('Comment'):

            comment_body = [item.text for item in comment_item.findall('Body')][0]
            if comment_body is None:
                continue
            comment_body = [x.strip() for x in comment_body.replace('\r', '').split('\n') if x.strip() != '']

            rthread.add_comments(comment_body)

        if len(rthread.comments) > 1:
            if index < num_train:
                train_data.append(rthread)
            elif num_train <= index < num_train + num_val:
                val_data.append(rthread)
            else:
                test_data.append(rthread)
        else:
            print('empty thread')

    return train_data, val_data, test_data


def write_data(data_dic, data_dir, data_part):
    for key, elem in enumerate(data_dic):
        title = ' '.join([x.replace('\n', '').replace('\r', '').strip() for x in data_dic[key].title_sentences])
        body = '\n'.join([x.replace('\n', '').replace('\r', '').strip() for x in data_dic[key].initial_comment_sentences])

        summary_1 = '\n'.join(data_dic[key].selected_sentences_text)

        comments = data_dic[key].comments
        comment_text = ''
        for comment_index, comment_body in enumerate(comments):
            comment_text += comment_body.replace('\n', '').replace('\r', '') + '\n'

        writer = codecs.open(data_dir + '//' + data_part + '_' + str(key) + '.story', 'w', encoding='utf8')
        writer.write(title + ' ' + body + '\n')
        # writer.write(body + '\n')
        writer.write(comment_text)
        writer.write('@highlight\n')
        writer.write(summary_1)
        writer.close()

        print(data_part + '_' + str(key))

train_data, val_data, test_data = read_github_data('../github_Data/issues_v2_combined.xml')
print('writing Training data')
write_data(train_data, '../github_Data/raw/', 'train')
print('writing validation data')
write_data(val_data, '../github_Data/raw/', 'val')
print('writing testing data')
write_data(test_data, '../github_Data/raw/', 'test')