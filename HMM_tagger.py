#!/usr/bin/env python
##################################
# COMP150 NLP Spring 2020, Programming Assignment #2
#
# Saber Bahranifard
# saber.bahranifard@tufts.edu
#
#
#


import numpy as np
import sys
from collections import defaultdict

class HMM():
    """ 
    This class defines all the required distributions to setup 
    a Hidden Markov Model (HMM)
    The model-specific attrebutes are:
        - S :: scaler, number of sentences in the training corpus
        - T :: scaler, number of distinct tags in the training corpus
        - V :: scaler, number of words in the Vocabulary

        - init_dist :: Tx1 vector
        The initial distribution of hidden states (tags). This distribution shows the 
        probability of a given tag being observed at the begining of a sentence in the
        training corpus
        
        - tran_dist :: TxT matrix
        The Transition matrix gives the probability of a tag occurring given the previous tag 
        in the sentence is observed.

        - emit_dist ::  NxT matrix
        The Emission matrix gives the likelihood distribution of words given a tag

    """
    def __init__(self, tag_set):
        self.tag_set = list(tag_set)
        self.log_init_dist = dict()
        self.log_tran_dist = dict()
        self.log_emit_dist = dict()

        # total number of sentences in the training corpus
        self.S = 0
        # total number of distict tags in the training corpus
        self.T = 0
        # total number of distinct words in the training corpus
        self.V = 0

    def HMM_distributions(self, data):
        """
        This method calculates the initial, Transition, and 
        Emission distributions from the provided training corpus

        Arguments
        ---------
        
        """
        # to add smoothing 
        tag_counts = dict.fromkeys(self.tag_set,0)
        start_tag_counts = dict.fromkeys(self.tag_set,0)
        word_tag_counts = defaultdict(dict)
        tag_tag_counts = defaultdict(dict)
        for row_tag in self.tag_set:
            for col_tag in self.tag_set:
                tag_tag_counts[row_tag][col_tag] = 0

        for sentences, tags in zip(data['sentences'], data['tags']):
            # print(sentences, "\n", tags)
            for i,_ in enumerate(tags):
                if i == 0:
                    start_tag_counts[tags[i]] += 1
                else:
                    tag_tag_counts[tags[i]][tags[i-1]] += 1
            for word, tag in zip (sentences, tags):
                tag_counts[tag] += 1
                if tag not in word_tag_counts[word]:
                    word_tag_counts[word] = {key:0 for key in list(self.tag_set)}
                word_tag_counts[word][tag] += 1

        self.S = len(data['sentences'])
        self.T = len(tag_tag_counts)
        self.V = len(word_tag_counts)
        # print(self.S, self.T, self.V)

        # calculate Initial probability deistribution
        self.log_init_dist = self.Laplace_smoothing(start_tag_counts)
        # print(self.log_init_dist['IN'])

        # calculate Transition probability distribution
        self.log_tran_dist= self.Laplace_smoothing(tag_tag_counts)
        # print(self.log_emit_dist['1990']['CD'])

        # calculate Emission probability distribution
        self.log_emit_dist = self.Laplace_smoothing(word_tag_counts)
        # print(self.log_emit_dist['1990']['CD'])

        return

    def Laplace_smoothing(self, counts):
        dist_dict = dict()
        if type(list(counts.values())[0]) == int:
            dist_ary = np.array(list(counts.values())) + 1
            dist_ary = np.log(dist_ary) - np.log(np.sum(dist_ary))
            dist_dict = {key:value for key,value in zip(counts.keys(), dist_ary)}
        else:
            dist_ary = np.array([list(counts[key].values()) for key in list(counts.keys())]) + 1
            dist_ary = np.log(dist_ary) - np.log(np.sum(dist_ary, axis=0))
            for indx, key in enumerate(counts.keys()):
                dist_dict.update({key:{tag:value for tag,value in zip(counts[key].keys(), dist_ary[indx])}})
        return dist_dict
        
class HMM_numpy(HMM):
    def Viterbi(self, sentence):
        best_path_prob = list()
        best_path_tag = list()
        for word_indx, word in enumerate(sentence):
            if word in self.log_emit_dist.keys():
                emit_dist = self.log_emit_dist[word]
            else:
                emit_dist = {key:-np.log(self.V) for key in self.tag_set}

            init_dist_ary = np.array(list(self.log_init_dist.values()))
            emit_dist_ary = np.array(list(emit_dist.values()))
            tran_dist_ary = np.array([list(self.log_tran_dist[key].values()) for key in list(self.log_tran_dist.keys())])
            
            if word_indx ==0:
                # print("-------------------------------")
                # print("word: ", word)
                # print("emit_dist_ary: ", emit_dist_ary.shape, emit_dist_ary)
                # print("init_dist_ary: ", init_dist_ary.shape, init_dist_ary)
                viterbi_path = emit_dist_ary + init_dist_ary
                # print("emit_dist_ary + init_dist_ary: ", viterbi_path.shape, viterbi_path)
                # print("-------------------------------")
            else:
                # print("-------------------------------")
                # print("word: ", word)
                # print("tran_dist_ary: ", tran_dist_ary.shape, tran_dist_ary)
                # print("viter_path: ", viterbi_path.shape, viterbi_path)
                # print("np.amax(tran_dist_ary + viterbi_path, 1): ", np.amax(tran_dist_ary + viterbi_path, 1).shape, np.amax(tran_dist_ary + viterbi_path, 1))
                # print("emit_dist_ary: ", emit_dist_ary.shape, emit_dist_ary)
                viterbi_path = emit_dist_ary + np.amax(tran_dist_ary + viterbi_path, 1)
                # print("emit_dist_ary + np.amax(tran_dist_ary + viterbi_path, 1): ", viterbi_path.shape, viterbi_path)
                # print("-------------------------------")

            best_path_prob.append(np.amax(viterbi_path))
            best_path_tag.append(self.tag_set[np.argmax(viterbi_path)])
        return best_path_tag, best_path_prob



