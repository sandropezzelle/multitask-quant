This folder contains the code for one-task-frozen vagueQ task: it uses pretrained (frozen) visual features to perform the task of vague quantification (few, many, all, etc.)

train_vagueq.py - contains the training code
test_vagueq.py - contains the code to evaluate a trained model

In order to train a model, you need to run the command:
python train_vagueq.py "data path"
where "data path" is the path containing the .txt files with the frozen vectors

For testing a pretrained model, you need to create the folder "best_model" in the current directory, copy the pretrained model
as "weight.best.hdf5" and you run: python test_vagueq.py "data path"
