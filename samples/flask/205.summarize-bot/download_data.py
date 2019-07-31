import wget
import os
import nltk

if __name__ == "__main__":
    nltk.download('punkt')
    os.system("python -m spacy download en")
    os.system("python -m spacy download en_core_web_lg")

    if not os.path.exists("./models/aux_data"):
        os.mkdir("./models/aux_data")

    if not os.path.exists("./models/aux_data/Stanford"):
        os.mkdir("./models/aux_data/Stanford")

    print('downloading {}'.format('stanford-corenlp-full-2018-10-05.tar'))
    wget.download("https://connectorprod.blob.core.windows.net/sdk-models/summarize-bot/aux_data/Stanford/stanford-corenlp-full-2018-10-05.tar", out="./models/aux_data/Stanford/stanford-corenlp-full-2018-10-05.tar")

    current_dir = os.getcwd()

    command = 'tar -xvf {}/models/aux_data/Stanford/stanford-corenlp-full-2018-10-05.tar -C {}/models/aux_data/Stanford/'.format(current_dir, current_dir)
    os.system(command)

    if not os.path.exists("./models/BertSum"):
        os.mkdir("./models/BertSum")

    if not os.path.exists("./models/BertSum/temp_dir"):
        os.mkdir("./models/BertSum/temp_dir")

    if not os.path.exists("./models/BertSum/temp_dir/models"):
        os.mkdir("./models/BertSum/temp_dir/models")

    if not os.path.exists("./models/BertSum/temp_dir/models/bert_classifier"):
        os.mkdir("./models/BertSum/temp_dir/models/bert_classifier")

    print('downloading {}'.format('model_step_56000'))
    wget.download("https://connectorprod.blob.core.windows.net/sdk-models/summarize-bot/BertSum/model_step_56000.pt", out="./models/BertSum/temp_dir/models/bert_classifier/model_step_56000.pt")

    if not os.path.exists("./models/siatl"):
        os.mkdir("./models/siatl")

    if not os.path.exists("./models/siatl/checkpoints"):
        os.mkdir("./models/siatl/checkpoints")

    print('downloading {}'.format('forum_bi_coatt_att_combined_aux_ft_gu_67'))
    wget.download("https://connectorprod.blob.core.windows.net/sdk-models/summarize-bot/siatl/checkpoints/forum_bi_coatt_att_combined_aux_ft_gu_67.pt", out="./models/siatl/checkpoints/forum_bi_coatt_att_combined_aux_ft_gu_67.pt")

    print('downloading {}'.format('lm20m_70K'))
    wget.download("https://connectorprod.blob.core.windows.net/sdk-models/summarize-bot/siatl/checkpoints/lm20m_70K.pt", out="./models/siatl/checkpoints/lm20m_70K.pt")

    if not os.path.exists("./models/SummaRuNNer"):
        os.mkdir("./models/SummaRuNNer")

    if not os.path.exists("./models/SummaRuNNer/checkpoints"):
        os.mkdir("./models/SummaRuNNer/checkpoints")

    print('downloading {}'.format('model_forum_30_75_19'))
    wget.download("https://connectorprod.blob.core.windows.net/sdk-models/summarize-bot/SummaRuNNer/data/model_forum_30_75_19.pkl", out="./models/SummaRuNNer/checkpoints/model_forum_30_75_19.pkl")

    print('downloading {}'.format('model_forum_30_75_best'))
    wget.download("https://connectorprod.blob.core.windows.net/sdk-models/summarize-bot/SummaRuNNer/data/model_forum_30_75_best.pkl", out="./models/SummaRuNNer/checkpoints/model_forum_30_75_best.pkl")

    print('downloading {}'.format('model_forum_30_75_bert_2_22'))
    wget.download("https://connectorprod.blob.core.windows.net/sdk-models/summarize-bot/SummaRuNNer/data/model_forum_30_75_bert_2_22.pkl", out="./models/SummaRuNNer/checkpoints/model_forum_30_75_bert_2_22.pkl")

    print('downloading {}'.format('model_forum_30_75_bert_2_best'))
    wget.download("https://connectorprod.blob.core.windows.net/sdk-models/summarize-bot/SummaRuNNer/data/model_forum_30_75_bert_2_best.pkl", out="./models/SummaRuNNer/checkpoints/model_forum_30_75_bert_2_best.pkl")

    print('downloading {}'.format('model_forum_30_75_bert_3_32'))
    wget.download("https://connectorprod.blob.core.windows.net/sdk-models/summarize-bot/SummaRuNNer/data/model_forum_30_75_bert_3_32.pkl", out="./models/SummaRuNNer/checkpoints/model_forum_30_75_bert_3_32.pkl")

    print('downloading {}'.format('model_forum_30_75_bert_3_best'))
    wget.download("https://connectorprod.blob.core.windows.net/sdk-models/summarize-bot/SummaRuNNer/data/model_forum_30_75_bert_3_best.pkl", out="./models/SummaRuNNer/checkpoints/model_forum_30_75_bert_3_best.pkl")

    print('downloading {}'.format('model_forum_30_75_coatt_33'))
    wget.download("https://connectorprod.blob.core.windows.net/sdk-models/summarize-bot/SummaRuNNer/data/model_forum_30_75_coatt_33.pkl", out="./models/SummaRuNNer/checkpoints/model_forum_30_75_coatt_33.pkl")

    print('downloading {}'.format('model_forum_30_75_coatt_best'))
    wget.download("https://connectorprod.blob.core.windows.net/sdk-models/summarize-bot/SummaRuNNer/data/model_forum_30_75_coatt_best.pkl", out="./models/SummaRuNNer/checkpoints/model_forum_30_75_coatt_best.pkl")

    print('downloading {}'.format('model_forum_30_75_coatt_bert_2_10'))
    wget.download("https://connectorprod.blob.core.windows.net/sdk-models/summarize-bot/SummaRuNNer/data/model_forum_30_75_coatt_bert_2_10.pkl", out="./models/SummaRuNNer/checkpoints/model_forum_30_75_coatt_bert_2_10.pkl")

    print('downloading {}'.format('model_forum_30_75_coatt_bert_2_best'))
    wget.download("https://connectorprod.blob.core.windows.net/sdk-models/summarize-bot/SummaRuNNer/data/model_forum_30_75_coatt_bert_2_best.pkl", out="./models/SummaRuNNer/checkpoints/model_forum_30_75_coatt_bert_2_best.pkl")

    print('downloading {}'.format('model_forum_30_75_coatt_bert_3_19'))
    wget.download("https://connectorprod.blob.core.windows.net/sdk-models/summarize-bot/SummaRuNNer/data/model_forum_30_75_coatt_bert_3_19.pkl", out="./models/SummaRuNNer/checkpoints/model_forum_30_75_coatt_bert_3_19.pkl")

    print('downloading {}'.format('model_forum_30_75_coatt_bert_3_best'))
    wget.download("https://connectorprod.blob.core.windows.net/sdk-models/summarize-bot/SummaRuNNer/data/model_forum_30_75_coatt_bert_3_best.pkl", out="./models/SummaRuNNer/checkpoints/model_forum_30_75_coatt_bert_3_best.pkl")

    if not os.path.exists("./models/aux_data/wiki"):
        os.mkdir("./models/aux_data/wiki")

    print('downloading {}'.format('lsa.model'))
    wget.download("https://connectorprod.blob.core.windows.net/sdk-models/summarize-bot/aux_data/wiki/lsa.model", out="./models/aux_data/wiki/lsa.model")

    print('downloading {}'.format('own_lsi.model'))
    wget.download("https://connectorprod.blob.core.windows.net/sdk-models/summarize-bot/aux_data/own_lsi.model", out="./models/aux_data/own_lsi.model")

    print('downloading {}'.format('own_lsi.model.projection'))
    wget.download("https://connectorprod.blob.core.windows.net/sdk-models/summarize-bot/aux_data/own_lsi.model.projection", out="./models/aux_data/own_lsi.model.projection")

    if not os.path.exists("./models/aux_data/forum"):
        os.mkdir("./models/aux_data/forum")

    print('downloading {}'.format('lsa.model'))
    wget.download("https://connectorprod.blob.core.windows.net/sdk-models/summarize-bot/aux_data/forum/lsa.model", out="./models/aux_data/forum/lsa.model")

    print('downloading {}'.format('stanford-english-corenlp-2018-10-05-models'))
    wget.download("https://connectorprod.blob.core.windows.net/sdk-models/summarize-bot/aux_data/Stanford/stanford-english-corenlp-2018-10-05-models.jar", out="./models/aux_data/Stanford/stanford-english-corenlp-2018-10-05-models.jar")