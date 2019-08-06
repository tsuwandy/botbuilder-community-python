import wget
import os

if __name__ == "__main__":
    if not os.path.exists("./aux_data/Stanford"):
        os.mkdir("./aux_data/Stanford")

    print('downloading {}'.format('stanford-corenlp-full-2018-10-05.tar'))
    wget.download("https://ahmago.blob.core.windows.net/summarization-bot-files/aux_data/Stanford/stanford-corenlp-full-2018-10-05.tar", out="./aux_data/Stanford/stanford-corenlp-full-2018-10-05.tar")

    current_dir = os.getcwd()

    command = 'tar -xvf {}/aux_data/Stanford/stanford-corenlp-full-2018-10-05.tar -C {}/aux_data/Stanford/'.format(current_dir, current_dir)
    os.system(command)

    if not os.path.exists("./BertSum/temp_dir"):
        os.mkdir("./BertSum/temp_dir")

    if not os.path.exists("./BertSum/temp_dir/models"):
        os.mkdir("./BertSum/temp_dir/models")

    if not os.path.exists("./BertSum/temp_dir/models/bert_classifier"):
        os.mkdir("./BertSum/temp_dir/models/bert_classifier")

    print('downloading {}'.format('model_step_56000'))
    wget.download("https://ahmago.blob.core.windows.net/summarization-bot-files/BertSum/model_step_56000.pt", out="./BertSum/temp_dir/models/bert_classifier/model_step_56000.pt")

    print('downloading {}'.format('forum_bi_coatt_att_combined_aux_ft_gu_67'))
    wget.download("https://ahmago.blob.core.windows.net/summarization-bot-files/siatl/checkpoints/forum_bi_coatt_att_combined_aux_ft_gu_67.pt", out="./siatl/checkpoints/forum_bi_coatt_att_combined_aux_ft_gu_67.pt")

    print('downloading {}'.format('lm20m_70K'))
    wget.download("https://ahmago.blob.core.windows.net/summarization-bot-files/siatl/checkpoints/lm20m_70K.pt", out="./siatl/checkpoints/lm20m_70K.pt")

    print('downloading {}'.format('model_forum_30_75_19'))
    wget.download("https://ahmago.blob.core.windows.net/summarization-bot-files/SummaRuNNer/data/model_forum_30_75_19.pkl", out="./SummaRuNNer/checkpoints/model_forum_30_75_19.pkl")

    print('downloading {}'.format('model_forum_30_75_best'))
    wget.download("https://ahmago.blob.core.windows.net/summarization-bot-files/SummaRuNNer/data/model_forum_30_75_best.pkl", out="./SummaRuNNer/checkpoints/model_forum_30_75_best.pkl")

    print('downloading {}'.format('model_forum_30_75_bert_2_22'))
    wget.download("https://ahmago.blob.core.windows.net/summarization-bot-files/SummaRuNNer/data/model_forum_30_75_bert_2_22.pkl", out="./SummaRuNNer/checkpoints/model_forum_30_75_bert_2_22.pkl")

    print('downloading {}'.format('model_forum_30_75_bert_2_best'))
    wget.download("https://ahmago.blob.core.windows.net/summarization-bot-files/SummaRuNNer/data/model_forum_30_75_bert_2_best.pkl", out="./SummaRuNNer/checkpoints/model_forum_30_75_bert_2_best.pkl")

    print('downloading {}'.format('model_forum_30_75_bert_3_32'))
    wget.download("https://ahmago.blob.core.windows.net/summarization-bot-files/SummaRuNNer/data/model_forum_30_75_bert_3_32.pkl", out="./SummaRuNNer/checkpoints/model_forum_30_75_bert_3_32.pkl")

    print('downloading {}'.format('model_forum_30_75_bert_3_best'))
    wget.download("https://ahmago.blob.core.windows.net/summarization-bot-files/SummaRuNNer/data/model_forum_30_75_bert_3_best.pkl", out="./SummaRuNNer/checkpoints/model_forum_30_75_bert_3_best.pkl")

    print('downloading {}'.format('model_forum_30_75_coatt_33'))
    wget.download("https://ahmago.blob.core.windows.net/summarization-bot-files/SummaRuNNer/data/model_forum_30_75_coatt_33.pkl", out="./SummaRuNNer/checkpoints/model_forum_30_75_coatt_33.pkl")

    print('downloading {}'.format('model_forum_30_75_coatt_best'))
    wget.download("https://ahmago.blob.core.windows.net/summarization-bot-files/SummaRuNNer/data/model_forum_30_75_coatt_best.pkl", out="./SummaRuNNer/checkpoints/model_forum_30_75_coatt_best.pkl")

    print('downloading {}'.format('model_forum_30_75_coatt_bert_2_10'))
    wget.download("https://ahmago.blob.core.windows.net/summarization-bot-files/SummaRuNNer/data/model_forum_30_75_coatt_bert_2_10.pkl", out="./SummaRuNNer/checkpoints/model_forum_30_75_coatt_bert_2_10.pkl")

    print('downloading {}'.format('model_forum_30_75_coatt_bert_2_best'))
    wget.download("https://ahmago.blob.core.windows.net/summarization-bot-files/SummaRuNNer/data/model_forum_30_75_coatt_bert_2_best.pkl", out="./SummaRuNNer/checkpoints/model_forum_30_75_coatt_bert_2_best.pkl")

    print('downloading {}'.format('model_forum_30_75_coatt_bert_3_19'))
    wget.download("https://ahmago.blob.core.windows.net/summarization-bot-files/SummaRuNNer/data/model_forum_30_75_coatt_bert_3_19.pkl", out="./SummaRuNNer/checkpoints/model_forum_30_75_coatt_bert_3_19.pkl")

    print('downloading {}'.format('model_forum_30_75_coatt_bert_3_best'))
    wget.download("https://ahmago.blob.core.windows.net/summarization-bot-files/SummaRuNNer/data/model_forum_30_75_coatt_bert_3_best.pkl", out="./SummaRuNNer/checkpoints/model_forum_30_75_coatt_bert_3_best.pkl")

    if not os.path.exists("./aux_data/wiki"):
        os.mkdir("./aux_data/wiki")

    print('downloading {}'.format('lsa.model'))
    wget.download("https://ahmago.blob.core.windows.net/summarization-bot-files/aux_data/wiki/lsa.model", out="./aux_data/wiki/lsa.model")

    print('downloading {}'.format('own_lsi.model'))
    wget.download("https://ahmago.blob.core.windows.net/summarization-bot-files/aux_data/own_lsi.model", out="./aux_data/own_lsi.model")

    print('downloading {}'.format('own_lsi.model.projection'))
    wget.download("https://ahmago.blob.core.windows.net/summarization-bot-files/aux_data/own_lsi.model.projection", out="./aux_data/own_lsi.model.projection")

    if not os.path.exists("./aux_data/forum"):
        os.mkdir("./aux_data/forum")

    print('downloading {}'.format('lsa.model'))
    wget.download("https://ahmago.blob.core.windows.net/summarization-bot-files/aux_data/forum/lsa.model", out="./aux_data/forum/lsa.model")

    print('downloading {}'.format('stanford-english-corenlp-2018-10-05-models'))
    wget.download("https://ahmago.blob.core.windows.net/summarization-bot-files/aux_data/Stanford/stanford-english-corenlp-2018-10-05-models.jar", out="./aux_data/Stanford/stanford-english-corenlp-2018-10-05-models.jar")

    print('downloading {}'.format('pytorch_model'))
    wget.download("https://ahmago.blob.core.windows.net/summarization-bot-files/uncased_L-12_H-768_A-12/pytorch_model.bin", out="./SummaRuNNer/pytorch-pretrained-BERT/bert_models/uncased_L-12_H-768_A-12/pytorch_model.bin")
