from pyrouge import Rouge155
from pyrouge.utils import log
import re
import os
from os.path import join
import logging
import tempfile
import subprocess as sp
from cytoolz import curry
import codecs
import shutil

if os.path.exists('/mnt/Work/Summarization/pyrouge/tools/ROUGE-1.5.5'):
    _ROUGE_PATH = '/mnt/Work/Summarization/pyrouge/tools/ROUGE-1.5.5'
elif os.path.exists('/mnt/Summarization/pyrouge/tools/ROUGE-1.5.5'):
    _ROUGE_PATH = '/mnt/Summarization/pyrouge/tools/ROUGE-1.5.5'
else:
    _ROUGE_PATH = '/mnt/e/Work/Summarization_Codes/pyrouge/tools/ROUGE-1.5.5'

def read_summaries(file_path):
    lines = []
    data_reader = open(file_path, 'r')
    for line in data_reader:
        line = line.replace('\n', '').replace('\r', '').strip()
        lines.append(line)
    return lines


def write_in_files(output_dir, data, extension):
    for index, summary in enumerate(data):
        if len(summary) == 1:
            data_writer = codecs.open(output_dir + '/{}.{}'.format(index + 1, extension), 'w', encoding='utf8')
            data_writer.write(summary[0])
            data_writer.close()
        elif len(summary) == 2:
            data_writer = codecs.open(output_dir + '/{}_1.{}'.format(index + 1, extension), 'w', encoding='utf8')
            data_writer.write(summary[0])
            data_writer.close()

            data_writer = codecs.open(output_dir + '/{}_2.{}'.format(index + 1, extension), 'w', encoding='utf8')
            data_writer.write(summary[1])
            data_writer.close()


def eval_rouge(tmp_dir, dec_pattern, dec_dir, ref_pattern, ref_dir,
               cmd='-c 95 -r 1000 -n 2 -m', system_id=1):
    """ evaluate by original Perl implementation"""
    # silence pyrouge logging
    assert _ROUGE_PATH is not None
    log.get_global_console_logger().setLevel(logging.WARNING)
    # with tempfile.TemporaryDirectory() as tmp_dir:
    #     tmp_dir = 'E:/Work/Summarization_samples/SummRunner/output/train/temp/'

    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.mkdir(tmp_dir)

    Rouge155.convert_summaries_to_rouge_format(
        dec_dir, join(tmp_dir, 'dec'))
    Rouge155.convert_summaries_to_rouge_format(
        ref_dir, join(tmp_dir, 'ref'))
    Rouge155.write_config_static(
        join(tmp_dir, 'dec'), dec_pattern,
        join(tmp_dir, 'ref'), ref_pattern,
        join(tmp_dir, 'settings.xml'), system_id
    )
    cmd = (join('sudo perl ' + _ROUGE_PATH, 'ROUGE-1.5.5.pl')
           + ' -e {} '.format(join(_ROUGE_PATH, 'data'))
           + cmd
           + ' -a {}'.format(join(tmp_dir, 'settings.xml')))
    output = sp.check_output(cmd, universal_newlines=True, shell=True)
    return output


def main(eval_output_director, input_dec_file_path, input_ref_file_path):

    if not os.path.exists(eval_output_director):
        os.mkdir(eval_output_director)

    decoded_summaries = read_summaries(input_dec_file_path)
    decoded_summaries = [[x] for x in decoded_summaries]
    reference_summaries = read_summaries(input_ref_file_path)
    reference_summaries = [[x] for x in reference_summaries]

    dec_dir = eval_output_director + '/dec/'
    ref_dir = eval_output_director + '/ref/'
    temp_dir = eval_output_director + '/tmp/'

    if not os.path.exists(dec_dir):
        os.mkdir(dec_dir)
    else:
        shutil.rmtree(dec_dir)
        os.mkdir(dec_dir)

    if not os.path.exists(ref_dir):
        os.mkdir(ref_dir)
    else:
        shutil.rmtree(ref_dir)
        os.mkdir(ref_dir)

    if not os.path.exists(temp_dir):
        os.mkdir(temp_dir)
    else:
        shutil.rmtree(temp_dir)
        os.mkdir(temp_dir)

    write_in_files(eval_output_director + '/dec/', decoded_summaries, 'dec')
    write_in_files(eval_output_director + '/ref/', reference_summaries, 'ref')

    dec_pattern = r'(\d+).dec'
    ref_pattern = '#ID#.ref'

    output = eval_rouge(temp_dir, dec_pattern, dec_dir, ref_pattern, ref_dir)
    return output


eval_output_dir = '../ROUGE_output/'
eval_output_director_abs = os.path.abspath(eval_output_dir)

max_rouge_index = 0
max_rouge = 0
max_output = ''

bert_results_main_dir = '../results/bert_classifier'

for j in [20000] + [x for x in range(37000, 51000, 1000)]:
    ref_file = '{}/test_{}_step{}.gold'.format(bert_results_main_dir, j, j)
    ref_file = os.path.abspath(ref_file)
    dec_file = '{}/test_{}_step{}.candidate'.format(bert_results_main_dir, j, j)
    dec_file = os.path.abspath(dec_file)

    output = main(eval_output_director_abs, dec_file, ref_file)
    print('step {} ROUGE scores: '.format(j))
    print(output)
    print('---------------------')
    lines = output.split('\n')

    current_rouge = 0
    for line in lines:
        if 'Average_F' in line:
            val = float(line.split('Average_F:')[1].split('(')[0].strip())
            current_rouge += val
    if current_rouge > max_rouge:
        max_output = output
        max_rouge = current_rouge
        max_rouge_index = j

print('Best Model...... step {}'.format(max_rouge_index))
print(max_output)


