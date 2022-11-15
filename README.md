# Data Science Task
This is a repository containing code for downloading, processing, exploring, training and evaluating models on the [ptb-xl dataset](https://physionet.org/content/ptb-xl/1.0.3/).

I possess no previous knowledge on ECG data. I have developed this code using my knowledge on Data Science and Machine Learning, without having done an exhaustive literature search on the field.
I have selected a ResNet18 model following other similar approaches for this task (1d-signal data).

The focus on this repository would be on predicting the superclass labels (NORM, MI, STTC, CD and HYP), using the samplerate-100 data (as opposed to the samplerate-500 data). It is also possible to predict other labels (such as subclasses), but a 5-output prediction can be complicated enough, and this is just a week-project with some prototype code on the first steps I would follow if having to deal with this task.

The value of this repository is to be able to explore the dataset, with figures that can be visualized by experts to gain some domain knowledge. The trained models in this repository can also be used to predict abnormalities on the ECG data (with an AUC of 0.8-0.9). This could ease the cardiac diagnosis of cardiac abnormalities, and so increase the chances of successful treatments, without the need of experts annotating the data manually.

An example use of this code could be to predict a score on cardiac abnormalities (labels MI, STTC, CD and HYP), and set a low thresholds (have quite some false positives), and then pass this ECG data to experts, which could make a final decision.
The models presented here should not be used for diagnostic purposes without expert supervision. The models have not been fine-tuned, nor they are understood once trained. 

I have also added a (naive) explainability pipeline, such that experts could use it to see why the model is making such diagnostic predictions.

Further work is needed in all directions of this repository.

# Structure

The structure of the code can be seen in four main blocks:
- Block 0: The [ptb-xl dataset](https://physionet.org/content/ptb-xl/1.0.3/) is downloaded, and processed using *wfdb* functions. Here, I implement the following tasks regarding the _ECG_ data
  - Read the _ECG_ files and corresponding annotations
  - Plot the _ECG_ signal in appropriate manner to be read by a doctor
  - Identify the heart beat of the signal, average and total heart beat in the signal
  - Identify the complex QRS in the signal and annotate on it
- Block 1: An exploration phase of the data, both for the _ECG_ signals and for the *.csv* files present. The tasks present here are:
  - Some data wrangling, on the annotations and demographics of the _ECG_ signals.
  - An exploration of the _ECG_ data according to lower-dimensional features.
  - Training of some simple algoritms [(Support Vector Machines)](https://en.wikipedia.org/wiki/Support_vector_machine) on these features.
- Block 2: Training on the _ECG_ signals, predicting the superclass labels (NORM, MI, STTC, CD and HYP) using a ResNet model.
- Block 3: Evaluation of the models trained in Block 2.

# Environments
Due to the dataset being "big", and my personal laptop being almost agonal and with limited memory, I have decided to use [google colab](https://colab.research.google.com/) for running most of the notebooks, google drive to store the data (as numpy arrays) and [Weights&Biases](https://wandb.ai/site) for experiment tracking. 

However, the initial notebooks (*0_Create_arrays.ipynb* and *0_Read_ECG.ipynb*) are run locally. This is due to the fact that the data had been dowloaded locally. These local notebooks save the required data to [numpy arrays](https://numpy.org/doc/stable/reference/generated/numpy.array.html) and uploaded manually to Google Drive, to be loaded in the colab environments.

To run the local notebooks, the dependencies are as following:
## Dependencies for local environments
- *wfdb* to read and process ECG data
- *matplotlib* for plotting
- *numpy*
- *scipy* to do FFT transforms, etc
The versions are not specified here. Better documentation on the dependencies is needed, and a *requirements.txt* file should be created.

For the colab notebooks, the dependencies are installed and imported in each notebook. It is recommended to use a GPU (cuda) environment.

# References
No exhaustive literature study has been made.
For the model selection, I have followed this literature:
- [ECG Heartbeat Classification Based on ResNet and Bi-LSTM](https://iopscience.iop.org/article/10.1088/1755-1315/428/1/012014)
- [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf)

For saliency maps (trying to understand the model decisions), I follow
- [Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps](https://arxiv.org/abs/1312.6034)
and for "smoothed" saliency maps I follow
- [SmoothGrad: removing noise by adding noise](https://arxiv.org/pdf/1706.03825.pdf)

Better models exists, and better pipelines for explainability for ECG data and for time series. In this repository I made a prototype / first steps, considering the time-frame and expectations.

# Lessons learned


