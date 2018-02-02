'''
FIND CORRELATED NEURONS BETWEEN NETWORKS

To use, generate description files using `describe.lua` from here: https://github.com/dabbler0/nmt-shared-information.

Then create a file listing the locations of the description files for your networks. For instance, I might create a file `myfile.txt` that reads:

```
../descriptions/en-es-1.desc.t7
../descriptions/en-es-2.desc.t7
../descriptions/en-fr-1.desc.t7
../descriptions/en-fr-2.desc.t7
```

Then invoke `python correlations.py --descriptions myfile.txt`.

If you want to run correlation-min instead of correlation-max, add the --min flag: `python correlations.py --descriptions myfile.txt --min`.
'''

import argparse
import torch
import numpy
import json
from tqdm import tqdm
from numpy import newaxis as na
from torch.utils.serialization import load_lua
from itertools import product as p

parser = argparse.ArgumentParser(description = 'Run correlation analysis')
parser.add_argument('--min', dest='pool', action='store_const', const=min, default=max)
parser.add_argument('--descriptions', dest='descriptions', description = 'File with list of locations of description files (one per line)')

args = parser.parse_args()

# Load all the descriptions of networks
with open(parser.descriptions) as f:
    all_networks = {line: torch.cat(load_lua(line).cuda() for line in f}

'''
Correlation-finding code. This should probably not need to be modified for ordinary use.
========================================================================================
'''

# Get means and stdevs so that we can whiten appropriately
means = {}
stdevs = {}
for network in tqdm(all_networks, desc='mu, sigma'):
    means[network] = all_networks[network].mean(0, keepdim=True)
    stdevs[network] = (
        all_networks[network] - means[network].expand_as(all_networks[network])
    ).pow(2).mean(0, keepdim=True).pow(0.5)

correlations = {network: {} for network in all_networks}

# Get all correlation pairs
for network, other_network in tqdm(p(all_networks, all_networks), desc='correlate', total=len(all_networks)**2):
    # Don't match within one network
    if network == other_network:
        continue

    # Correlate these networks with each other
    covariance = (
        torch.mm(
            all_networks[network].t(), all_networks[other_network] # E[ab]
        ) / all_networks[network].size()[0]
        - torch.mm(
            means[network].t(), means[other_network] # E[a]E[b]
        )
    )

    correlation = covariance / torch.mm(
        stdevs[network].t(), stdevs[other_network]
    )

    correlations[network][other_network] = correlation.cpu().numpy()

# Get all "best correlation pairs"
clusters = {network: {} for network in all_networks}
for network, neuron in tqdm(p(all_networks, range(500)), desc='clusters', total=len(all_networks)*500):
    clusters[network][neuron] = {
        other: max(
            range(500),
            key = lambda i: abs(correlations[network][other][neuron][i])
        ) for other in correlations[network]
    }

neuron_notated_sort = {}
# For each network, created an "annotated sort"
#
# Sort neurons by worst best correlation with another neuron
# in another network.
for network in tqdm(all_networks, desc='annotation'):
    neuron_sort = sorted(
        range(500),
        key = lambda i: -args.pool(
            abs(correlations[network][other][i][clusters[network][i][other]])
            for other in clusters[network][i]
        )
    )

    # Annotate each neuron with its associated cluster
    neuron_notated_sort[network] = [
        (
            neuron,
            {
                '%s:%d' % (other, clusters[network][neuron][other],):
                correlations[network][other][neuron][clusters[network][neuron][other]]
                for other in clusters[network][neuron]
            }
        )
        for neuron in neuron_sort
    ]

json.dump(neuron_notated_sort, open('results/annotated-clusters-min.json', 'w'), indent = 4)
