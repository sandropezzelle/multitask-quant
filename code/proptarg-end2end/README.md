This folder contains the code for prop-targ-end2end model: it is the end2end model for predicting the correct ratio of animals in a given image

complete_ratio.py - it contains the implementation of the model for predicting the ratio
train_ratio.py - contains the training code
test_ratio.py - contains the code to evaluate a trained model


In order to train a model, you need to run the command
python train_ratio.py "repository path" "data path"

For testing a pretrained model, you need to create the folder "best_model" in the current directory, copy the pretrained model
as "weight.best.hdf5" and you can run python test_ratio.py "repository path" "data path"
