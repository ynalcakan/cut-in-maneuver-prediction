# Monocular Vision-based Prediction of Cut-in Maneuvers with LSTM Networks

## Abstract
Advanced driver assistance and automated driving systems should be capable of predicting and avoiding
dangerous situations. 
In this study, we propose a method to predict the potentially dangerous lane changes (cut-ins) of the vehicles in front.
We follow a computer vision-based approach that only employs a single in-vehicle RGB camera and we classify the target vehicle's maneuver based on the recent video frames. 
Our algorithm consists of a CNN-based vehicle detection and tracking step and an LSTM-based maneuver classification step.
It is computationally efficient compared to other vision-based methods since it exploits a small number of features for the classification step rather than feeding CNNs with RGB frames.
To evaluate our approach, we have worked on a publicly available dataset and tested several classification models.
Experiment results reveal that 0.9325 accuracy can be obtained with side-aware two-class (cut-in vs. lane-pass) classification models.

Proposed method architecture:
![pipeline](https://github.com/ynalcakan/cut-in-maneuver-prediction/blob/main/figures/pipeline_v5.png?raw=true)

## Information about the repository:

Proposed pipeline's classification code is available in "**maneuver_prediction_train_and_test_LSTM.py**".

After you extracted target vehicle bounding box features and created your train-validation-test X and y csv files as mentioned in the paper, you can get evaluation results from the code.

About the data we used:<br/>

 - First, you should create a user at Berkeley Deep Drive Dataset portal (https://bdd-data.berkeley.edu/).
 - After you download the training data from BDD-100K, you can cut cut-in and lane-pass maneuvers from them by using the data.csv file.

**Note:** information about data.csv file will be here after the ITSC.
