# Software Dependencies

- Python 2.7 
- Tensorflow 0.12
- OpenCV 3.1.0
- Numpy
- Matplotlib
- Pandas
- Pillow
- h5py
- argparse

And for making use of the real-time camera app, you will need a computer with 
a webcam plugged in. 


# Data

The data used for training the models in this project are not included. The data 
can be downloaded from the following links. 

- http://ufldl.stanford.edu/housenumbers/train.tar.gz
- http://ufldl.stanford.edu/housenumbers/test.tar.gz
- http://ufldl.stanford.edu/housenumbers/extra.tar.gz

The scripts assume that the contents of these tar files are extracted within a subdirectory "data". This should result in the three subdirectories: 

- data/train
- data/extra
- data/test

To process the data to be used by the neural network, simply run the following 
in the command line: 

    python create_data_pickles.py
    
This will create several pickle files within the `data` subdirectory that will 
be used by the neural network directly. 


# Training

    python trainer.py -m a -e 10 -b 32 -a 0.001 -o "A_01"


| argument | what it is         | possible values |
|----------|--------------------|----------------|
| -m       | model architecture | "a", "b", "c", or "d" |
| -e       | number of epochs   | any integer value |
| -b       | batch-size         | any integer value |
| -a       | alpha learning rate| a float between 0-1 |
| -o       | output name *      | any string that could legally be used as a directory name. |

- \* the output name will be used as a place where all snapshots, evaluation files, and visualisations for the model will be stored, in a subdirectory `results/output_name` 


# Evaluate on Test Dataset

    python tester.py

This will print out the Per Digit Accuracy, Whole Number Accuracy, and Intersect 
of Union score on the test dataset. It will also save images of the following 
things in the `imgs` subdirectory: 

- Grid of predictions on test set
- Grid of worst performing bounding box predictions
- Grid of incorrect digit predictions



# Real Time Prediction

    python live_predictor.py

Will launch an application that displays the captured video feed from the 
default webcam on the computer, and overlays the predicted bounding boxes and 
digits on the screen in real time. 



# Pre-Trained Models

Only the The checkpoint files for the best performing model (model A) are 
packaged here in the interest of keeping file size to a minimum. These are 
located in `results/A_02/checkpoints`.



