The project aims to train two ANNs using human eye gaze data distributed along with
the source code.

Data set contains two file:
1. fzis_training.csv - the training set
2. fzis_testing.csv - the testing set

Two ANNs resides in below python scripts:

1. NN_NoBimodalRemoval.py - the control group ANN
2. NN_BimodalRemoval.py - the experiment group ANN, basically has everything the same 
as control group except for it has implemented BDR during training

To train and obtain test results of each ANN, simply run the python script.

Both scripts contains commented code that plot summary data to help better visualise
statistics that are useful, user should uncomment those lines should they find it useful.