from lxml import etree as ET
from SummarizationBot.Data.thread_object import ThreadObject
from tqdm import tqdm

def read_xml(file_path):
    data = []
    tree = ET.parse(file_path)
    root = tree.getroot()
    for post_item in tqdm(root):
        post_id = post_item.tag
        title_list = [item.text for item in post_item.findall('Title')]
        if title_list is not None:
            title = title_list[0]
        else:
            title = ''

        owner_list = [item.text for item in post_item.findall('Owner')]
        if owner_list is not None:
            owner_id = owner_list[0]
        else:
            owner_id = ''

        initial_post_list = [item.text for item in post_item.findall('Body')]
        if initial_post_list is not None:
            initial_post_sentences = initial_post_list[0].split('\n')
        else:
            initial_post_sentences = None

        human_summary1_list = [item.text for item in post_item.findall('Summary_1')]
        if human_summary1_list is not None:
            summary_1 = human_summary1_list[0]
        else:
            summary_1 = None

        human_summary2_list = [item.text for item in post_item.findall('Summary_2')]
        if human_summary2_list is not None:
            summary_2 = human_summary2_list[0]
        else:
            summary_2 = None

        selected_sentences_list = [item.text for item in post_item.findall('selected_sentences')]
        if selected_sentences_list is not None:
            selected_sentences = selected_sentences_list[0].split('\n')
        else:
            selected_sentences = None

        repository_list = [item.text for item in post_item.findall('Repository')]
        if repository_list is not None:
            repository = repository_list[0]
        else:
            repository = None

        thread = ThreadObject(post_id, title, initial_post_sentences)
        thread.set_selected_sentences(selected_sentences)
        thread.repository = repository
        thread.human_summary_1 = summary_1
        thread.human_summary_2 = summary_2
        thread.generated_full_summary = ''

        for comment_item in post_item.findall('Comment'):
            comment_body = [item.text for item in comment_item.findall('Body')][0]
            if comment_body is None:
                continue
            comment_body = comment_body.split('\n')
            thread.add_reply(comment_body)
        data.append(thread)

    return data


