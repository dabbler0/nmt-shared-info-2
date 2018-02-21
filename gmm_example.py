
import gmm
import json

gmm.initialize(['../descriptions/en-es-1.desc.t7'], '../data/testsets/tokenized-test/en.tok')

def my_tagger(sentence):
    return [(word[-1] == 's') for word in sentence]

f1_scores = gmm.run_gmm(my_tagger, (True, False),
    desc='ends in s', scoring_function=gmm.make_f1_scorer(0))

json.dump(f1_scores, open('output.json', 'w'))
