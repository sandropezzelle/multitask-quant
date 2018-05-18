# models

This repository contains 7 models, each in a separate folder. In particular, you can find all the necessary codes for running and testing 3 one-task-frozen, 3 one-task-end2end, and 1 multi-task-prop models.

The script 'compute-vagueQ-correlation.py' can be used to compute Pearson correlations given the predictions for vagueQ task. The predictions by each model are in the corresponding folder. The use is: python compute-vagueQ-correlation.py

The file 'Q-probabilities.txt' contains the probability for each quantifier at any given ratio.
