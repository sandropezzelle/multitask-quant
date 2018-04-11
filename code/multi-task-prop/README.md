This folder contains the code for multi-task-prop model: it is the end2end multitask model in order to predict all 3 tasks

complete_model.py - it contains the implementation of the multitask model
train_complete.py - contains the training code
test_complete.py - contains the code to evaluate a trained model


In order to train a model, you need to run the command
python train_complete.py "repository path" "data path"

For testing a pretrained model, you need to create the folder "best_model" in the current directory, copy the pretrained model
as "weight.best.hdf5" and you can run python test_complete.py "repository path" "data path"
