# POS Tagging using Hidden Markov Models

Natural language processing (NLP) is an important research area in artificial intelligence, dating back to at least the 1950’s. One of the most basic problems in NLP is part-of-speech tagging, in which the goal is to mark every word in a sentence with its part of speech (noun, verb, adjective, etc.). This is a first step towards extracting semantics from natural language text.

The goal of this project is to tag all the words in an unseen Test corpus. To achieve this goal, a tagged Train corpus along with a tagged Development corpus are provided to train and evaluate a model. As model, for this sequential task, a bi-gram Hidden Markov Model is used. The HMM model is optimized using a Viterbi Algorithm. Also, to avoid zero probability for the words in the Test corpus which are not already observed in the training set, a Laplace Smoothing method is used. The Development set is tagged to be able to score the model. The accuracy of the POS tagging is reported at the end of this brief. 

__Code Structure:__
A Part-of-Speech (POS) tagging is assumed to be a Markov Process with tags being assumed as hidden states and each observed word in a sentence as observed data. The Markov Process is relying on the facts that:
* At each state, the observed data is conditionally independent of previous states and is just conditionally dependent on the current state.
* Each state is just conditionally dependent on the immediate previous state and is conditionally independent of any state beyond or before.
Using Hidden Markov Model (HMM), this Markov Process can be learned in a stochastic approach. The basic components of a HMM model for POS
tagging are three distributions:
* Initial Distribution, PS(Ti): Probability of a sentence in the corpus be-
gins with Ti in the Tag Set.
Transition distribution, PT (Tj jTi): Probability of a transition to tag Tj
given the current state is tag Ti
Emission distribution, PE(Wj jTi): Probability of observing the word Wj
given the current state is tag Ti
The purpose of the POS tagger is to nd the most likely sequence of
tags for a given sentence. For this purpose, we need to maximize the joint
posterior distribution of latent variables, in this case tags, given a sequence
of observations, in this case words in a training corpus. To be able to imple-
ment such optimizer on the HMM model, a dynamic programming algorithm,
called Viterbi Algorithm is used. This algorithm takes in the aforemen-
tioned distributions and iteratively nds a sequence of tags which maximizes
this probability.
In the provided code, inside the HMM tagger.py le, a HMM class is
dened which does all the word counts required to setup the Initial distri-
bution log init dist , Transition probability matrix log tran dist , and
Emission distributions log emit dist . Also, to avoid overtting to the
training set and under-weighting the unseen words, a Laplace Smoothing
method is used. The ( Laplace smoothing ) method under the HMM class

__Data:__ The dataset is a large corpus of labeled training and testing data,
consisting of nearly 1 million words and 50,000 sentences. The file format of the datasets is:
each line consists of a word, followed by a space, followed by one of 12 part-of-speech tags: ADJ (adjective),
ADV (adverb), ADP (adposition), CONJ (conjunction), DET (determiner), NOUN, NUM (number), PRON
(pronoun), PRT (particle), VERB, X (foreign word), and . (punctuation mark). Sentence boundaries are
indicated by blank lines. </br></br>

label.py is the main program, pos scorer.py, which has the scoring code, and pos solver.py, which contains the actual
part-of-speech estimation code. The program takes as input two filenames: a training file and a testing file and displays accuracy using simple probability, Bayes net variable elimination method and Viterbi algorithm to find the maximum a posteriori (MAP). </br> </br>
It also displays the logarithm of the posterior probability for each solution it finds, as well as a
running evaluation showing the percentage of words and whole sentences that have been labeled correctly
according to the ground truth. </br></br>
To run the code:</br>
__python label.py part2 training_file testing_file__

