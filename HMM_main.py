#!/usr/bin/env python
##################################
# COMP150 NLP Spring 2020, Programming Assignment #2
#
# Saber Bahranifard
# saber.bahranifard@tufts.edu
#
#
# Make sure that:
#    - Python 3 is being used
#    - input files are mentioned as the mentioned order
# Usage:
# $ python ./main_HMM.py training_file dev_file
# or
# $ python ./main_HMM.py training_file dev_file test_file
#

import sys
from HMM_tagger import HMM_numpy

# Read in data files
def read_data(fname):
    tagged_corpus_dict = dict(sentences=[],tags=[])
    tokens = []
    tags = []
    with open(fname, 'r') as file:
        for line in file:
            data = tuple([w for w in line.split()])
            if (len(data) > 1):
                if data:
                    tokens += [data[0]]
                    tags += [data[1]]
                else:
                    tagged_corpus_dict['sentences'].append(list(tokens))
                    tagged_corpus_dict['tags'].append(list(tags))
                    tokens = []
                    tags = []
            else:
                if data:
                    tokens += [data[0]]
                    tags += []
                else:
                    tagged_corpus_dict['sentences'].append(list(tokens))
                    tagged_corpus_dict['tags'].append(list(tags))
                    tokens = []
                    tags = []

    return tagged_corpus_dict 

# Write out data files
def write_data(fname, data):
    with open(fname, 'w') as file:
        for (sentence, tags) in data:
            for word, tag in zip(sentence, tags):
                file.write(str(word) + "\t" + str(tag) + "\n")
            file.write("\n")
    return



####################
# Main program
#
if __name__ == '__main__':
    if len(sys.argv) < 3:
        print ("Make sure that: ")
        print ("    - Python 3 is being used")
        print ("    - input files are mentioned as the mentioned order \n")
        print ("Usage: ")
        print ("$ python ./main_HMM.py training_file test_file \n\
__or__ \n\
$ python ./main_HMM.py training_file dev_file test_file")
        sys.exit()
    elif len(sys.argv) == 3:
        train_file = sys.argv[1]
        print ("Training HMM model...")
        train_data = read_data(train_file)
        # print(len(train_data['sentences']))
        test_file = sys.argv[2]
    else:
        train_file = sys.argv[1]
        print ("Training HMM model...")
        train_data = read_data(train_file)
        dev_file = sys.argv[2]
        dev_data = read_data(dev_file)
        train_data['sentences'] += dev_data['sentences']
        train_data['tags'] += dev_data['tags']
        test_file = sys.argv[3]

    tag_set = set([item for sublist in train_data['tags'] for item in sublist])
    hmm_object = HMM_numpy(tag_set)
    hmm_object.HMM_distributions(train_data)
    # print(hmm_object.log_init_dist['IN'])
    # print(hmm_object.log_emit_dist['1990']['CD'])
    # print(hmm_object.log_tran_dist['CD']['NNP'])

    print ("Loading test data...")
    test_data = read_data(test_file)
    # print(test_data)

    print ("Testing HMM tagger...")
    output_data = list()
    for sentence in test_data['sentences']:
        tags, _ = hmm_object.Viterbi(sentence)
        # print("Sentence: ", sentence)
        # print("Infered tags: ", tags)
        output_data.append((sentence, tags))
    print("-------------------------------")
    write_data('POS_test.pos', output_data)
    # write_data('my_test.pos', output_data)
