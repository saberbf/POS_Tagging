###################################
# CS B551 Fall 2017, Assignment #3
# D. Crandall
#
# There should be no need to modify this file, although you 
# can if you really want. Edit pos_solver.py instead!
#
# To get started, try running: 
#
#   python ./label.py bc.train bc.test.tiny
#


# from pos_scorer import Score
# from pos_solver import *
import sys
from HMM_tagger import *

# Read in training or test data file
#
# A generator funtion to skip blank lines
def nonblank_lines(fname):
    for lines in fname:
        line = lines.rstrip()
        if line:
            yield line

def read_data(fname):
    tagged_corpus_dict = dict(sentences=[],tags=[])
    tokens = []
    tags = []
    with open(fname, 'r') as file:
        for line in file:
            data = tuple([w for w in line.split()])

            if data:
                tokens += [data[0]]
                tags += [data[1]]
            else:
                tagged_corpus_dict['sentences'].append(list(tokens))
                tagged_corpus_dict['tags'].append(list(tags))
                tokens = []
                tags = []

    return tagged_corpus_dict 


####################
# Main program
#

# if len(sys.argv) < 3:
#     print ("Usage: ")
#     print ("    ./label.py training_file test_file")
#     sys.exit()

# (train_file, test_file) = sys.argv[1:3]
# train_file = sys.argv[1]
train_file = "./POS/test.pos"
print ("Learning model...")
train_data = read_data(train_file)
tag_set = set([item for sublist in train_data['tags'] for item in sublist])
hmm_object = HMM(tag_set)
tag_count = hmm_object.HMM_distributions(train_data)
print(hmm_object.log_emit_dist['1990']['CD'])
print(hmm_object.log_tran_dist['CD']['NNP'])
# solver = Solver()
# solver.train(train_data)

# print ("Loading test data...")
# test_data = read_data(test_file)

# print "Testing classifiers..."
# scorer = Score()
# Algorithms = ("Simplified", "HMM VE", "HMM MAP")
# Algorithm_labels = [str(i + 1) + ". " + Algorithms[i] for i in range(0, len(Algorithms))]
# for (s, gt) in test_data:
#     outputs = {"0. Ground truth": gt}

#     # run all algorithms on the sentence
#     for (algo, label) in zip(Algorithms, Algorithm_labels):
#         outputs[label] = solver.solve(algo, s)

#     posteriors = {o: solver.posterior(s, outputs[o]) for o in outputs}

#     Score.print_results(s, outputs, posteriors)

#     scorer.score(outputs)
#     scorer.print_scores()

#     print "----"


