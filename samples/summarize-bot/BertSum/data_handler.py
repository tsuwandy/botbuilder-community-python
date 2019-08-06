import codecs
import os


def clear_directory(dir_name):
    story_files = os.listdir(dir_name)
    for fname in story_files:
        os.remove(dir_name + '\\' + fname)


def write_to_story_files(docs, data_dir, data_part):
    clear_directory(data_dir)
    write_names = open(data_dir + '/../urls/mapping_test.txt', 'w')
    for index, doc in enumerate(docs):
        sents = doc.split('\n')
        writer = codecs.open(data_dir + '//' + data_part + '_' + str(index) + '.story', 'w', encoding='utf8')
        for sent in sents:
            writer.write(sent + '\n')
        writer.write('@highlight\n')
        writer.write(data_part + '_' + str(index) + '.story')
        writer.close()
        write_names.write(data_part + '_' + str(index) + '\n')
    write_names.close()
	
    write_names = open(data_dir + '/../urls/mapping_train.txt', 'w')
    write_names.write('\n')
    write_names.close()
	
    write_names = open(data_dir + '/../urls/mapping_valid.txt', 'w')
    write_names.write('\n')
    write_names.close()
	

def convert_data_summarize(main_dir):
    current_dir = os.getcwd()
    main_dir = current_dir + '/' + main_dir
    main_dir = main_dir.replace('\\', '/')

    clear_directory('{}/bert/'.format(main_dir))
    clear_directory('{}/tokenized/'.format(main_dir))
    clear_directory('{}/json/'.format(main_dir))
    clear_directory('{}/results/bert_classifier/'.format(main_dir))

    command_1 = 'python ./BertSum/src/preprocess.py -mode tokenize -raw_path {}/raw -save_path {}/tokenized/ -log_file {}/preprocess.log'.format(main_dir, main_dir, main_dir)
    command_2 = 'python ./BertSum/src/preprocess.py -mode format_to_lines -raw_path {}/tokenized/ -save_path {}/json/temp -map_path {}/urls -lower -log_file {}/preprocess.log'.format(main_dir, main_dir, main_dir, main_dir)
    command_3 = 'python ./BertSum/src/preprocess.py -mode format_to_bert -raw_path {}/json/ -save_path {}/bert/ -oracle_mode greedy -n_cpus 4 -log_file {}/preprocess.log -min_nsents 1 '.format(main_dir, main_dir, main_dir)
    command_4 = 'python ./BertSum/src/train.py -mode test -bert_config_path {}/../bert_config_uncased_base.json -bert_data_path {}/bert/temp -model_path {}/models/bert_classifier  -test_from {}/models/bert_classifier/model_step_56000.pt -visible_gpus -1  ' \
                '-gpu_ranks -1 -batch_size 30000  -log_file {}/log_bert_classifier  -result_path {}/results/bert_classifier/test_temp_ -test_all -block_trigram true'.format(main_dir, main_dir, main_dir, main_dir, main_dir, main_dir)

    os.system(command_1)
    os.system(command_2)
    os.system(command_3)
    os.system(command_4)


def read_output(main_dir):
    current_dir = os.getcwd()
    main_dir = current_dir + '/' + main_dir + '/results/bert_classifier/'
    main_dir = main_dir.replace('\\', '/')

    file_names = []
    summaries = []
    for fname in os.listdir(main_dir):
        if '.gold' in fname:
            reader = open(main_dir + fname)
            for line in reader:
                line = line.replace('\n', '').replace('\r', '').replace('<q>', '').replace(' ', '').strip()
                line = line.replace('.story','').replace('test_','')
                file_names.append(int(line))
        elif '.candidate' in fname:
            reader = open(main_dir + fname)
            for line in reader:
                line = line.replace('\n', '').replace('\r', '').replace('<q>', ' ').strip()
                summaries.append(line)

    summaries_list = [[x, y] for x,y in zip(file_names, summaries)]
    summaries_list = sorted(summaries_list)
    return [x[1] for x in summaries_list]


