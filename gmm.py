'''
Gaussian Mixture Model code.
============================

This code is meant to be imported by another Python script to be run.

Here is an example of usage. This will test all neurons in 'en-es-1.desc'
for f1 selectivity for words that end in 's'. It can also be found in gmm_example.py

```
import gmm
import json

gmm.initialize(['en-es-1.desc.t7'], 'en.tok')

def my_tagger(sentence):
    return [(word[-1] == 's') for word in sentence]

f1_scores = gmm.run_gmm(my_tagger, (True, False),
    desc='ends in s', scoring_function=gmm.make_f1_scorer(0))

json.dump(f1_scores, open('output.json', 'w'))
```

'''
import torch
from torch.utils.serialization import load_lua
from itertools import product as p
from tqdm import tqdm
import json
import os

languages = ['es', 'fr', 'ar', 'ru', 'zh']

networks = {}
lines = []

# INITIALIZE
# ==========
# Load data files and description files.
# description_fnames should be an iterable and source_file
# should be a string which is a filename.
def initialize(description_fnames, source_file):
    global networks
    global lines

    for fname in tqdm(description_fnames):
        nname = os.path.split(fname)[1]
        nname = nname[:nname.index('.')]

        # Load as 4000x(sentence_length)x500 matrix
        networks[nname] = load_lua(fname)['encodings']

        sample = torch.cat(networks[nname])

        # Normalize description
        mean = sample.mean(0)
        stdev = (sample - mean).pow(2).mean(0).sqrt()

        # In-place normalize
        for x in networks[nname]:
            x.sub_(mean).div_(stdev)

    with open(source_file) as f:
        lines = f.read().split('\n')[:-1]
        lines = [x.split(' ') for x in lines]
        lines = [line for line in lines if len(line) < 250]

# ACCURACY SCORE
# ==============
# A scoring function for use in run_gmm
def accuracy_score(indices, tag_tensor):
    return indices.eq(tag_tensor.unsqueeze(1).expand_as(indices)).float().mean(0)

# MAKE_F1_SCORER
# ==============
# A factory function that produces scoring functions for use in run_gmm
# Usage: run_gmm(my_tagger, my_tag_list, scoring_function = make_f1_scorer(0))
# Where 0 is the relevant class.
def make_f1_scorer(index):
    epsilon = 1e-7

    def f1_score(indices, tag_tensor):
        positives = indices.eq(index).float()
        retrieved = tag_tensor.eq(index).unsqueeze(1).expand_as(indices).float()

        precision = (positives * retrieved).sum(0) / (epsilon + retrieved.sum(0))
        recall = (positives * retrieved).sum(0) / (epsilon + positives.sum(0))

        return 2 * (precision * recall) / (precision + recall + epsilon)
    return f1_score

# RUN_GMM
# =======
# Given a manual tagger and a list of possible tags, train and run a GMM
# and see how it scores according to a scoring function.
def run_gmm(manual_tag, tags, desc='match', scoring_function=accuracy_score):
    # Tag to index:
    tag2idx = {tag: i for i, tag in enumerate(tags)}

    concatenated_tags = []

    used_tags = set()

    # Sort into buckets
    for i, line in tqdm(enumerate(lines), total=len(lines), desc=desc):
        line_tags = manual_tag(line)
        used_tags.update(line_tags)
        concatenated_tags.extend([tag2idx[tag] for tag in line_tags])

    tag_tensor = torch.Tensor(concatenated_tags).long().cuda()

    network_accuracies = {}

    for nname in networks:
        # tokens x dim_size
        data = torch.cat(networks[nname]).float().cuda()
        tokens, dim_size = data.size()

        # Get necessary data for mixed Gaussian model
        mean_tensor = torch.stack([
            data.index_select(0, tag_tensor.eq(i).nonzero().squeeze()).mean(0)
            for i in range(len(tags))
        ])

        stdev_tensor = torch.stack([
            data.index_select(0, tag_tensor.eq(i).nonzero().squeeze()).std(0)
            for i in range(len(tags))
        ])

        count_tensor = torch.Tensor([tag_tensor.eq(i).float().mean() for i in range(len(tags))]).cuda()

        count_tensor = torch.log(count_tensor)

        # Do predictions from mixed Gaussian model
        likelihoods = data.unsqueeze(0).expand(len(tags), tokens, dim_size)

        mean_tensor = mean_tensor.unsqueeze(1)
        stdev_tensor = stdev_tensor.unsqueeze(1)
        count_tensor = count_tensor.unsqueeze(1).unsqueeze(1)

        likelihoods = (-(
            (likelihoods - mean_tensor) / stdev_tensor
        ) ** 2) / 2 + count_tensor - torch.log(stdev_tensor) / 2

        # Indices here should be tokens x dim_size
        maxs, indices = torch.max(likelihoods, dim = 0)


        # Accuracies
        accuracies = scoring_function(indices, tag_tensor)

        scores, neurons = torch.sort(accuracies, descending = True)

        scores = scores.cpu().numpy().tolist()
        neurons = neurons.cpu().numpy().tolist()

        network_accuracies[nname] = list(zip(neurons, scores))

    return network_accuracies
