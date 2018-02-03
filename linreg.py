'''
FIND NEURONS IN NETWORKS PREDICTABLE WITH LINEAR REGRESSION

To use, generate description files using `describe.lua` from here: https://github.com/dabbler0/nmt-shared-information.

Then create a file listing the locations of the description files for your networks. For instance, I might create a file `myfile.txt` that reads:

```
../descriptions/en-es-1.desc.t7
../descriptions/en-es-2.desc.t7
../descriptions/en-fr-1.desc.t7
../descriptions/en-fr-2.desc.t7
```

Then invoke `python correlations.py --descriptions myfile.txt --output my_results.json`.
'''

import torch
import numpy
import json
from tqdm import tqdm
from numpy import newaxis as na
from torch.utils.serialization import load_lua
from itertools import product as p

import argparse

parser = argparse.ArgumentParser(description = 'Run linreg analysis')
parser.add_argument('--descriptions', dest='descriptions', description = 'File with list of locations of description files (one per line)')
parser.add_argument('--output', dest='output', description = 'Output file')

args = parser.parse_args()

# Load all the descriptions of networks
with open(parser.descriptions) as f:
    all_networks = {line: torch.cat(load_lua(line).cuda() for line in f}

# Normalize to have mean 0 and standard devaition 1.
means = {}
stdevs = {}
for network in tqdm(all_networks, desc='mu, sigma'):
    means[network] = all_networks[network].mean(0, keepdim=True)
    stdevs[network] = (
        all_networks[network] - means[network].expand_as(all_networks[network])
    ).pow(2).mean(0, keepdim=True).pow(0.5)

    all_networks[network] = (all_networks[network] - means[network]) / stdevs[network]

errors = {network: {} for network in all_networks}

# Get all correlation pairs
for network, other_network in tqdm(p(all_networks, all_networks), desc='correlate', total=len(all_networks)**2):
    # Don't match within one network
    if network == other_network:
        continue

    # Try to predict this network given the other one
    X = all_networks[other_network].clone()
    Y = all_networks[network].clone()

    coefs = X.t().mm(X).inverse().mm(X.t()).mm(Y)
    prediction = X.mm(coefs)
    error = (prediction - Y).pow(2).mean(0).squeeze()

    errors[network][other_network] = error

neuron_notated_sort = {}

# For each network, created an "annotated sort"
#
# Sort neurons by worst best correlation with another neuron
# in another network.
for network in tqdm(all_networks, desc='annotation'):
    neuron_sort = sorted(
        range(500),
        key = lambda i: max(
            errors[network][other][i]
            for other in errors[network] if other != 'position'
        )
    )

    # Annotate each neuron with its associated cluster
    neuron_notated_sort[network] = [
        (
            neuron,
            {
                other: errors[network][other][neuron]
                for other in errors[network]
            }
        )
        for neuron in neuron_sort
    ]

json.dump(neuron_notated_sort, open(args.output, 'w'), indent = 4)
