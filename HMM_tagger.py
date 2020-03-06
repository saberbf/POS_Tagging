#!/usr/bin/env python
##################################
# COMP150 NLP Spring 2020, Programming Assignment #1
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
    The model-specific parameters are:
        - K (n_states) :: scaler, number of hidden states (tags)

        - init_dist :: Kx1 vector
        The initial distribution of hidden states
        
        - tran_dist :: KxK matrix
        The Transition matrix gives the probability of a tag occurring given the previous tag 
        in the sentence is observed
        The first row of this matrix show the initial distribution

        - emit_dist ::  NxK matrix
        The Emission matrix gives the likelihood distribution of words given a tag

    """
    def __init__(self, tag_set):
        self.tag_set = tag_set
        self.K = len(tag_set)
        # self.log_init_dist = np.log(np.ones(self.K)/self.K)
        # self.log_tran_dist = np.log(np.ones((self.K,self.K))/self.K**2)
        self.log_init_dist = dict()
        self.log_tran_dist = defaultdict(dict)
        for row_tag in self.tag_set:
            for col_tag in self.tag_set:
                self.log_tran_dist[row_tag][col_tag] = 0

        self.log_emit_dist = defaultdict(dict)

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
                #     word_tag_counts[word][tag] = 0
                    word_tag_counts[word] = {key:0 for key in list(self.tag_set)}
                word_tag_counts[word][tag] += 1

        # calculate Initial probability deistribution
        self.log_init_dist = np.array(list(start_tag_counts.values())) + 1
        self.log_init_dist = np.log(self.log_init_dist) - np.log(np.sum(self.log_init_dist))
        self.log_init_dist = {key:value for key,value in zip(start_tag_counts.keys(), self.log_init_dist)}
        # print(self.log_init_dist)
        # print(sum(self.log_init_dist.values()))

        # calculate Transition probability distribution
        self.log_tran_dist = np.array([list(tag_tag_counts[tag].values()) for tag in list(tag_tag_counts.keys())]) + 1
        self.log_tran_dist = np.log(self.log_tran_dist) - np.log(np.sum(self.log_tran_dist))
        _tmp = defaultdict(dict)
        for indx, tag in enumerate(tag_tag_counts.keys()):
            _tmp.update({tag:{key:value for key,value in zip(tag_tag_counts[tag].keys(), self.log_tran_dist[indx])}})
        self.log_tran_dist = _tmp
        # print(self.log_tran_dist['CD'])
        
        # calculate Emission probability distribution
        # self.log_emit_dist = np.array([list(word_tag_counts[word].values()) for word in list(word_tag_counts.keys())]) + 1
        # self.log_emit_dist = np.log(self.log_emit_dist) - np.log(np.sum(self.log_emit_dist))
        # _tmp = defaultdict(dict)
        # for indx, word in enumerate(word_tag_counts.keys()):
        #     _tmp.update({word:{key:value for key,value in zip(word_tag_counts[word].keys(), self.log_emit_dist[indx])}})
        # self.log_emit_dist = _tmp
        # print(self.log_emit_dist['1990']['CD'])
        self.log_emit_dist = self.Laplace_smoothing(word_tag_counts)
        # print(self.log_emit_dist['1990']['CD'])
        # print(self.Laplace_smoothing(word_tag_counts)['1990'])
        return

    def Laplace_smoothing(self, counts):
        # calculate Emission probability distribution
        dist_array = np.array([list(counts[key].values()) for key in list(counts.keys())]) + 1
        dist_array = np.log(dist_array) - np.log(np.sum(dist_array))
        dist_dict = defaultdict(dict)
        for indx, key in enumerate(counts.keys()):
            dist_dict.update({key:{tag:value for tag,value in zip(counts[key].keys(), dist_array[indx])}})
        return dist_dict
        
# class HMM_numpy(HMM):


