There are 2 possible versions of each file in this dataset.

1) file.pos -- there are two columns separated by a tab:
   1st column: token
   2nd column: POS tag
   Blank lines separate sentences.

   This is the format of training files, system output, and development
   or test files used for scoring purposes.

2) file.words -- one token per line, with blank lines between sentences.
   Format of an input file for a tagging program.

The following files are ditributed in for this project:

POS_train.pos  -- to use as the training corpus

POS_dev.words   -- to use as development set

POS_dev.pos     -- to use to check how well the system is doing

POS_test.words -- to run the model on.  The output file is in
	     	the .pos format .

scorer.py -- the scorer is to evaluate the output files
