import os
import subprocess
import torch
import h5py
import json
import nltk
import codecs
import argparse

parser = argparse.ArgumentParser(description='Neuron mask-out procedure')

# Required arguments
parser.add_argument('--model', 'Location of model file')
parser.add_argument('--src_dict', 'src.dict file location')
parser.add_argument('--targ_dict', 'targ.dict file location')
parser.add_argument('--source', 'Location of source file for testing')
parser.add_argument('--gold', 'Location of gold file for testing')
parser.add_argument('--ordering', 'Location of neuron ordering (as output by correlation.py or linreg.py)')
parser.add_argument('--output', 'Output location (json)')

# Optional arguments
parser.add_argument('--tmp_dir', '[optional; default=./tmp/] directory to store translations with killed neurons')
parser.add_argument('--projection_file', '[optional; default= ./projection.json] file through which to pass projection to seq2seq-attn')

with codecs.open(args.ordering, 'r', encoding = 'utf-8') as f:
    clusters = json.load(f)

network_orders = {network: [x[0] for x in clusters[network]] for network in clusters}

order = network_orders[network]

top_bleu_scores = {}
bottom_bleu_scores = {}

def test(projection_matrix, label):
    with codecs.open(projection_file, 'w', encoding = 'utf-8') as pf:
        json.dump(projection_matrix, pf)

    tmp_file_name = os.path.abspath(os.path.join(args.tmp_dir, '%s.txt' % label))

    # Run the tests
    subprocess.call(
        [   '/usr/bin/env'
            'th',
            os.path.abspath('seq2seq-attn/evaluate.lua'),
            '-model', os.path.abspath(args.model),
            '-src_file', os.path.abspath(args.source),
            '-output_file', tmp_file_name,
            '-projection', os.path.abspath(args.projection_file),
            '-src_dict', os.path.abspath(args.src_dict),
            '-targ_dict', os.path.abspath(args.targ_dict),
            '-replace_unk', '1',
            '-gpuid', '1'
        ],
        cwd = 'seq2seq-attn'
    )

    # Compute a BLEU score of the results.
    predicted_lines = []
    true_lines = []

    with codecs.open(tmp_file_name, encoding = 'utf-8') as tmp_file:
        with codecs.open(args.gold, encoding = 'utf-8') as true_file:
            for tmp_line, true_line in zip(tmp_file, true_file):
                tmp_line = tmp_line.split(' ')
                true_line = true_line.split(' ')

                predicted_lines.append(tmp_line)
                true_lines.append([true_line])

    return nltk.translate.bleu_score.corpus_bleu(true_lines, predicted_lines)


for threshold in [50 * x for x in range(0, 11)]:
    # Create the projection matrix and pass it
    # to the decoder

    # Starts as identity matrix
    projection_matrix = [[(1 if x == y else 0) for x in range(500)] for y in range(500)]

    # Zero out killed dimensions
    for i in range(min(threshold, 500)):
        projection_matrix[order[i]][order[i]] = 0

    top_bleu_scores[threshold] = test(projection_matrix, 'top-%d' % threshold)

    # Starts as identity matrix
    projection_matrix = [[(1 if x == y else 0) for x in range(500)] for y in range(500)]

    # Zero out killed dimensions
    for i in range(1, min(threshold, 500) + 1):
        projection_matrix[order[-i]][order[-i]] = 0

    bottom_bleu_scores[threshold] = test(projection_matrix, 'bottom-%d' % threshold)

json.dump((top_bleu_scores, bottom_bleu_scores), open(args.output, 'w'))
