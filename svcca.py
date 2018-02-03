import torch
import numpy
import json
from tqdm import tqdm
from numpy import newaxis as na
from torch.utils.serialization import load_lua
from itertools import product as p

import argparse

parser = argparse.ArgumentParser(description='Run svcca analysis')
parser.add_argument('--descriptions', dest='descriptions', description='File with list of locations of description files (one per line)')
parser.add_argument('--output', dest='output', description='Output file')
parser.add_argument('--normalize_dimensions', dest='normalize_dimensions', description='Add flag to normalize dimensions first', action='store_const', const=True, default=False)
parser.add_argument('--percent_variance', dest='percent_variance', type=float, description='Percentage of variance to take in initial PCA')

args = parser.parse_args()

# Load all the descriptions of networks
with open(parser.descriptions) as f:
    all_networks = {line: torch.cat(load_lua(line).cuda() for line in f}

# Whiten dimensions
if args.normalize_dimensions:
    for network in tqdm(all_networks, desc='mu, sigma'):
        all_networks[network] -= all_networks[network].mean(0)
        all_networks[network] /= all_networks[network].std(0)

# PCA to get independent components
whitening_transforms = {}
for network in tqdm(all_networks, desc='pca'):
    X = all_networks[network]
    covariance = torch.mm(X.t(), X) / (X.size()[0] - 1)

    e, v = torch.eig(covariance, eigenvectors = True)

    # Sort by eigenvector magnitude
    magnitudes, indices = torch.sort(torch.abs(e[:, 0]), descending = True)
    se, sv = e[:, 0][indices], v.t()[indices].t()

    # Figure out how many dimensions account for 99% of the variance
    var_sums = torch.cumsum(se, 0)
    wanted_size = torch.sum(var_sums.lt(var_sums[-1] * args.percent_variance))

    print('For network', network, 'wanted size is', wanted_size)

    # This matrix has size (dim) x (dim)
    whitening_transform = torch.mm(sv, torch.diag(se ** -0.5))

    # We wish to cut it down to (dim) x (wanted_size)
    whitening_transforms[network] = whitening_transform[:, :wanted_size]

    #print(covariance[:10, :10])
    #print(torch.mm(whitening_transforms[network], whitening_transforms[network].t())[:10, :10])

# CCA to get shared space
transforms = {}
for a, b in tqdm(p(all_networks, all_networks), desc = 'cca', total = len(all_networks) ** 2):
    if a is b or (a, b) in transforms or (b, a) in transforms:
        continue

    X, Y = all_networks[a], all_networks[b]

    # Apply PCA transforms to get independent things
    X = torch.mm(X, whitening_transforms[a])
    Y = torch.mm(Y, whitening_transforms[b])

    # Get a correlation matrix
    correlation_matrix = torch.mm(X.t(), Y) / (X.size()[0] - 1)

    # Perform SVD for CCA.
    # u s vt = Xt Y
    # s = ut Xt Y v
    u, s, v = torch.svd(correlation_matrix)

    X = torch.mm(X, u).cpu()
    Y = torch.mm(Y, v).cpu()

    transforms[a, b] = {
        a: whitening_transforms[a].mm(u),
        b: whitening_transforms[b].mm(v)
    }

torch.save(transforms, args.output)
