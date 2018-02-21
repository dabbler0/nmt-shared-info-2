'''
DISSECTION AND VISUALIZATION SERVER
===================================

Run with given arguments. An example --models file is given in models-example.txt.

To run visualizations, go to `localhost:8080` (or `localhost:8080/index.html`).

To run modified translations, go to `localhost:8080/modify.html`.

'''
import torch
import numpy
import json
from tqdm import tqdm
from torch.utils.serialization import load_lua
from itertools import product as p
import codecs
import argparse
import os

parser = argparse.ArgumentParser(description='Visualization and dissection server')
parser.add_argument('--models', help='File with list of lines in format: `description_loc model_loc src_dict_loc targ_dict_loc`, space-separated')
parser.add_argument('--svcca', help='.pkl file output by svcca.py (optional)')
parser.add_argument('--source', help='tokenized source file for the description files')

args = parser.parse_args()

'''
LOAD NETWORKS
'''

# Get list of network filenames
with open(args.models) as f:
    network_fnames = [line.strip().split(' ') for line in f]

all_networks = {}
model_files = {}
src_dicts = {}
targ_dicts = {}

for fname, mname, sdict, tdict in tqdm(network_fnames):
    network_name = os.path.split(fname)[1]
    network_name = network_name[:network_name.index('.')]

    # Record the location of the model file for this description file
    model_files[network_name] = os.path.abspath(mname)
    src_dicts[network_name] = os.path.abspath(sdict)
    targ_dicts[network_name] = os.path.abspath(tdict)

    # Load as 4000x(sentence_length)x500 matrix
    all_networks[network_name] = load_lua(fname)#['encodings']

means = {}
variances = {}

# transforms
if args.svcca is not None:
    cca_transforms = torch.load(args.svcca)

# Get means and variances
for network in tqdm(all_networks, desc = 'norm, pca'):
    # large number x 500
    concatenated = torch.cat(all_networks[network], dim = 0).cuda()
    means[network] = concatenated.mean(0)
    variances[network] = concatenated.std(0)

    means[network] = means[network].cpu()
    variances[network] = variances[network].cpu()

'''
SERVER CLASS
'''

from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs
import os
import subprocess
from subprocess import PIPE

current_loaded_subprocess = None
current_network = None

class VisualizationServer(BaseHTTPRequestHandler):

    STATIC_CONTENT_PATH = {
        '': 'html/index.html',
        '/': 'html/index.html',
        '/modify.html': 'html/modify.html',
        '/lines.txt': args.source
    }

    MIME_TYPES = {
        '.html': 'text/html',
        '.tok': 'text/plain'
    }

    def do_GET(self):
        global current_loaded_subprocess
        global current_network

        # 200 OK
        self.send_response(200)

        # Parse path
        path = urlparse(self.path)
        query = parse_qs(path.query)

        # Determine if path leads to static content.
        if path.path in VisualizationServer.STATIC_CONTENT_PATH:
            _, ext = os.path.splitext(VisualizationServer.STATIC_CONTENT_PATH[path.path])

            self.send_header('Content-type', VisualizationServer.MIME_TYPES[ext])
            self.end_headers()

            with codecs.open(VisualizationServer.STATIC_CONTENT_PATH[path.path], encoding='utf8') as f:
                self.wfile.write(f.read().encode('ascii', 'xmlcharrefreplace'))

        # You need to make a request to this first in order to initialize the pipeline.
        if path.path == '/begin-modify':
            current_network = query['network'][0]

            model_name = model_files[current_network]
            src_dict = src_dicts[current_network]
            targ_dict = targ_dicts[current_network]

            if current_loaded_subprocess is not None:
                current_loaded_subprocess.kill()

            current_loaded_subprocess = subprocess.Popen(
                [   '/usr/bin/env',
                    'th', # Find user's torch
                    os.path.abspath('seq2seq-attn/dissect.lua'), # Find our modified seq2seq
                    '-model', model_name,
                    '-src_file', os.path.abspath(args.source),
                    '-src_dict', src_dict,
                    '-targ_dict', targ_dict,
                    '-replace_unk', '1',
                    '-gpuid', '1'
                ],
                cwd = 'seq2seq-attn',
                stdin = PIPE,
                stdout = PIPE
            )

            # Read the two "loading" lines out
            current_loaded_subprocess.stdout.readline()
            current_loaded_subprocess.stdout.readline()

            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({
                'success': True
            }).encode('ascii'))

        # Modification server
        # This is called from modify.html.
        if path.path == '/modify':
            sentence = query['sentence'][0]

            modifications = json.loads(query['modifications'][0])

            for modification in modifications:
                index, neuron = modification['position']

                modification['value'] = (
                    (modification['value'] + means[current_network][neuron]) *
                    variances[current_network][neuron]
                )

                modification['position'] = (index + 1, neuron + 1)

            #(json.dumps(modifications) + '\n').encode('ascii')
            # Put some things in
            current_loaded_subprocess.stdin.write(
                (sentence + '\n').encode('ascii')
            )
            current_loaded_subprocess.stdin.write(
                (json.dumps(modifications) + '\n').encode('ascii')
            )
            current_loaded_subprocess.stdin.flush()

            # Get response out
            response = current_loaded_subprocess.stdout.readline().decode('utf-8')

            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({
                'pred': response
            }).encode('utf-8'))

        # Otherwise, run one of our endpoints.
        # We can request visualizations for individual neurons.
        if path.path == '/activations':
            self.send_header('Content-type', 'application/json')
            self.end_headers()

            # Get standard identifier.
            identifier = query['neuron'][0]

            # Identify the wanted neuron
            network, neuron = identifier.split(':')

            neuron = int(neuron)

            # To get an SVCCA comparison, ask for network, e.g. en-es-1/en-es-2:3
            # for the fourth canonically correlated "neuron" between en-es-1 and en-es-2.
            if '/' in network and args.svcca is not None:
                a, b = network.split('/')

                pair = (a, b) if (a, b) in cca_transforms else (b, a)

                transform = cca_transforms[pair][a].cpu()

                # Perform the transform on the fly
                activations = [
                    torch.mm(x, transform)[:, neuron] for x in all_networks[a]
                ]

                # Get mean and variance on the fly
                concat = torch.cat(activations)
                mean = concat.mean()
                variance = concat.std()

                activations = [x.numpy().tolist() for x in activations]

                self.wfile.write(json.dumps({
                    'activations': activations,
                    'mean': mean,
                    'variance': variance
                }).encode('ascii'))

            # Ordinary neuron activations are specified using the notation en-XX-Y;Z
            else:
                # Get the desired neuron activations
                activations = [x[:, neuron].numpy().tolist() for x in all_networks[network]]
                mean = means[network][neuron]
                variance = variances[network][neuron]

                # Write
                self.wfile.write(json.dumps({
                    'activations': activations,
                    'mean': mean,
                    'variance': variance
                }).encode('ascii'))

# Run the server on 8080
print('Running server on 8080.')
server_address = ('', 8080)
httpd = HTTPServer(server_address, VisualizationServer)
httpd.serve_forever()

