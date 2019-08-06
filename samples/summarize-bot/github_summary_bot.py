
import os
import spacy
from config import DefaultConfig

nlp = spacy.load('en_core_web_lg')
os.environ['CORENLP_HOME'] = 'C:/Data/Stanford/stanford-corenlp-full-2018-10-05'
CoreNLPPath = 'E:/Data/Stanford/stanford-corenlp-full-2018-10-05'


from github3 import login
from nltk import sent_tokenize, word_tokenize

class IssueObj:
    def __init__(self, header, body, owner, repo_name):
        self.header = header
        self.body = body
        self.comments = []
        self.owner = owner
        self.repo_name = repo_name
        self.comment_summaries = []
        self.generated_full_summary = ''

    def add_comment(self, comment_body, comment_owner):
        self.comments.append([comment_body, comment_owner])

    def add_comment_summary(self, comment_body):
        self.comment_summaries.append(comment_body)


class MySummaryBot():
    def __init__(self):
        self.questions = ["Which model would you like to use [**LSA-clustering**, **Summy**, **BertSum**, **SummaRuNNer**, **siatl**] ?",
                          "Please enter a repository name",
                          "Now enter the owner user id",
                          "**These are the issues found, which one would you like to summarize?**\n\n{}\n\n**Copy and paste the issue you'd like to summarize below**"]
        self.repo_name = None
        self.owner_name = None
        self.issue_header = None

        self.history_sentences = []
        self.state = 0
        self.gh = login(DefaultConfig.GIT_USERNAME, password=DefaultConfig.GIT_PASSWORD)
        self.eSummarizer = None
        
    def extract_issue_names(self, repo_name, repo_owner):
        try:
            issues = [x.title for x in self.gh.issues_on(repo_owner, repo_name)]
            return issues
        except:
            return None
        return None

    def extract_issue(self, repo_name, repo_owner, _issue_header):
        try:
            issues = [x for x in self.gh.issues_on(repo_owner, repo_name)]
            for issue in issues:
                issue_header = issue.title
                # print(issue_header)
                if issue_header.lower().strip() == _issue_header.lower().strip():
                    # print('found')
                    issue_body = issue.body
                    iss = IssueObj(issue_header, issue_body, repo_owner, repo_name)
                    comments = [y for y in issue.comments()]
                    for comment in comments:
                        iss.add_comment(str(comment.body), str(comment.user))
                    return iss
        except:
            return None
        return None

    def summarize_issue(self, _repository_name, _owner_name, _issue_header):
        self.repo_name = _repository_name
        self.owner_name = _owner_name
        self.issue_header = _issue_header

        issue = self.extract_issue(_repository_name, _owner_name, _issue_header)
        if issue is None:
            return 'Couldn\'t extract the issue'
        else:
            final_text = ''
            docs = [x[0] for x in issue.comments]
            posts = [issue.body for x in docs]
            docs = ['\n'.join([' '.join(word_tokenize(y)) for y in sent_tokenize(x)]) for x in docs]
            ####################################################################
            summaries = self.eSummarizer.summarize(docs, posts, 1)

            combined_summary = ''
            index = 1
            for doc, summary in zip(docs, summaries):
                if summary.replace('\n', ' ').replace('\r', ' ').replace('  ', ' ').replace('  ', ' ').strip() == '':
                    summary = doc
                combined_summary += summary.replace('\n', ' ').replace('\r', ' ').replace('  ', ' ').replace('  ', ' ').strip() + '\n'
                final_text += '\n\ncomment {}:\t'.format(index) + doc + '\n\n'
                final_text += 'Summary {}:\t'.format(index) + summary + '\n\n'
                final_text += '-------------------------------\n\n'
                index += 1

            docs2 = ['\n'.join([' '.join(word_tokenize(y)) for y in sent_tokenize(x)]) for x in [combined_summary]]
            posts = [issue.body for x in docs2]
            reduced_summary = self.eSummarizer.summarize(docs2, posts, 3)
            sentences = sent_tokenize(reduced_summary[0])
            final_text += '\n\n**Final Summary:** \n\n'
            for sent in sentences:
                final_text += sent + '\n\n'
            return final_text

    def update_state_reply(self, text):
        text = text.strip()
        if self.state < len(self.questions):
            if self.state == -1:
                self.state += 1
                return ''

            elif self.state == 0:
                model_name = text.strip().lower()
                if model_name in ['lsa-clustering', 'summy', 'bertsum', 'summarunner', 'siatl']:
                    if model_name == 'lsa-clustering':
                        from LSA_Model.model import ExtractiveSummarizer as esum
                        self.eSummarizer = esum()
                        self.eSummarizer.init_model('forum')
                    elif model_name == 'summy':
                        from Sumy_Model.model import ExtractiveSummarizer as esum
                        self.eSummarizer = esum()
                        self.eSummarizer.init_model('lsa')
                    elif model_name == 'bertsum':
                        from BertSum.model import ExtractiveSummarizer as esum
                        self.eSummarizer = esum()
                    elif model_name == 'summarunner':
                        from SummaRuNNer.model import ExtractiveSummarizer as esum
                        self.eSummarizer = esum()
                    elif model_name == 'siatl':
                        from siatl.model import ExtractiveSummarizer as esum
                        self.eSummarizer = esum()

                    # self.eSummarizer = esum()
                    # self.eSummarizer.init_model('forum')
                    self.state += 1
                    return self.questions[self.state]
                else:
                    return 'Wrong Model name, ' + self.questions[self.state]

            elif self.state == 1:
                self.state += 1
                self.repo_name = text.strip()
                return self.questions[self.state]

            elif self.state == 2:
                self.state += 1
                self.owner_name = text.strip()
                issue_names = self.extract_issue_names(self.repo_name, self.owner_name)
                if issue_names is None:
                    self.repo_name = None
                    self.owner_name = None
                    self.state = -1
                    return 'No issues found in this repo.  Would you like to start over (yes/no)?'
                else:
                    issue_names = '- ' + '\n- '.join(issue_names)
                    return self.questions[self.state].format(issue_names)

        if self.state == len(self.questions) - 1:
            text = text.lower()
            issue_tile = text.strip()
            summary = self.summarize_issue(self.repo_name, self.owner_name, issue_tile)
            reply = summary
            self.repo_name = None
            self.owner_name = None
            self.state = -1
            if reply is None:
                reply = 'No summarization'
            return reply + '\n**Would you like to start over (yes/no)?**'
        if self.state >= len(self.questions):
            if text.lower().strip() == 'y':
                self.repo_name = None
                self.owner_name = None
                self.state = -1
            else:
                return 'Sure... Would you like to start over (yes/no) ?'

    def dummy_func(self):
        for model_name in ['lsa-clustering', 'bertsum', 'summy', 'summarunner', 'siatl']:
            print(model_name)
            if model_name == 'lsa-clustering':
                from LSA_Model.model import ExtractiveSummarizer as esum
                self.eSummarizer = esum()
                self.eSummarizer.init_model('forum')
            elif model_name == 'summy':
                from Sumy_Model.model import ExtractiveSummarizer as esum
                self.eSummarizer = esum()
                self.eSummarizer.init_model('lsa')
            elif model_name == 'bertsum':
                from BertSum.model import ExtractiveSummarizer as esum
                self.eSummarizer = esum()
            elif model_name == 'summarunner':
                from SummaRuNNer.model import ExtractiveSummarizer as esum
                self.eSummarizer = esum()
            elif model_name == 'siatl':
                from siatl.model import ExtractiveSummarizer as esum
                self.eSummarizer = esum()

            repository_name = 'OpenNMT-py'
            owner_name = 'OpenNMT'
            issue_header = 'Export (image) model to ONNX'

            issue = self.extract_issue(repository_name, owner_name, issue_header)
            if issue is None:
                print('Couldn\'t extract the issue')
            else:
                final_text = ''
                docs = [x[0] for x in issue.comments]
                posts = [issue.body for x in docs]
                docs = ['\n'.join([' '.join(word_tokenize(y)) for y in sent_tokenize(x)]) for x in docs]
                ####################################################################
                summaries = self.eSummarizer.summarize(docs, posts, 1)

                combined_summary = ''
                index = 1
                for doc, summary in zip(docs, summaries):
                    if summary.replace('\n', ' ').replace('\r', ' ').replace('  ', ' ').replace('  ', ' ').strip() == '':
                        summary = doc
                    combined_summary += summary.replace('\n', ' ').replace('\r', ' ').replace('  ', ' ').replace('  ', ' ').strip() + '\n'
                    final_text += '\n\ncomment {}:\t'.format(index) + doc + '\n\n'
                    final_text += 'Summary {}:\t'.format(index) + summary + '\n\n'
                    final_text += '-------------------------------\n\n'
                    index += 1

                docs2 = ['\n'.join([' '.join(word_tokenize(y)) for y in sent_tokenize(x)]) for x in [combined_summary]]
                posts = [issue.body for x in docs2]
                reduced_summary = self.eSummarizer.summarize(docs2, posts, 3)
                sentences = sent_tokenize(reduced_summary[0])
                final_text += '\n\nFinal Summary: \n\n'
                for sent in sentences:
                    final_text += sent + '\n\n'
                print ('finished')#final_text

if __name__ == "__main__":
    mybot = MySummaryBot()
    mybot.dummy_func()