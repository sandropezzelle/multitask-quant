This folder contains the code for setcomp-end2end model: it is the end2end model for predicting if the are more/same/less animals in an image

complete_msl.py - it contains the implementation of the model for predicting more/same/less 
train_msl.py - contains the training code
test_msl.py - contains the code to evaluate a trained model


In order to train a model, you need to run the command
python train_msl.py "repository path" "data path"

For testing a pretrained model, you need to create the folder "best_model" in the current directory, copy the pretrained model
as "weight.best.hdf5" and you can run python test_msl.py "repository path" "data path"
