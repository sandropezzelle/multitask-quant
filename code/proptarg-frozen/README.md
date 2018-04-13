This folder contains the code for one-task-frozen propTarg task: it uses pretrained (frozen) visual features to perform the task of proportional estimation (0%, 20%, 50%, 100%, etc.)

train_proptarg.py - contains the training code
test_proptarg.py - contains the code to evaluate a trained model

In order to train a model, you need to run the command:
python train_proptarg.py "data path"
where "data path" is the path containing the .txt files with the frozen vectors

For testing a pretrained model, you need to create the folder "best_model" in the current directory, copy the pretrained model
as "weight.best.hdf5" and you run: python test_proptarg.py "data path"
