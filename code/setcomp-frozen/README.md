This folder contains the code for one-task-frozen setComp task: it uses pretrained (frozen) visual features to perform the task of set comparison (more, same, less)

train_setcomp.py - contains the training code
test_setcomp.py - contains the code to evaluate a trained model

In order to train a model, you need to run the command:
python train_setcomp.py "data path"
where "data path" is the path containing the .txt files with the frozen vectors

For testing a pretrained model, you need to create the folder "best_model" in the current directory, copy the pretrained model
as "weight.best.hdf5" and you run: python test_setcomp.py "data path"
