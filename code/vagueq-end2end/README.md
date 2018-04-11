This folder contains the code for vagueq-end2end model: it is the end2end model for predicting a probability distribution of the quantifiers 

complete_quant.py - it contains the implementation of the model for predicting a quantifier distribution 
train_quant.py - contains the training code
test_quant.py - contains the code to evaluate a trained model


In order to train a model, you need to run the command
python train_quant.py "repository path" "data path"

For testing a pretrained model, you need to create the folder "best_model" in the current directory, copy the pretrained model
as "weight.best.hdf5" and you can run python test_quant.py "repository path" "data path"
