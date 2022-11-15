# Data Science Task
This is a repository containing code for downloading, processing, exploring, training and evaluating models on the [ptb-xl dataset](https://physionet.org/content/ptb-xl/1.0.3/).

I possess no previous knowledge on ECG data. I have developed this code using my knowledge on Data Science and Machine Learning, without having done an exhaustive literature search on the field.
I have selected a ResNet18 model following other similar approaches for this task (1d-signal data).

The focus on this repository would be on predicting the superclass labels (NORM, MI, STTC, CD and HYP). It is also possible to predict other labels (such as subclasses), but a 5-output prediction can be complicated enough, and this is just a week-project with some prototype code on the first steps I would follow if having to deal with this task.

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

## Dependencies
- *wfdb*
- *matplotlib*


## What to do
We've tried to keep this task as similar to working here as possible. With that in mind, we think you'll know better than us what can be achieved with this. So there's no specific "thing" we want you to find or do. We want you to explore it as you would if you were working here.

We'd like you to analyse it and give us some insights. The insights should be useful and actionable in some way.

We ask data scientist do want to join **Idoven** to work with anonymised patient data, and on the basis of this data be able to:
- Be able to read the _ECG_ files and corresponding annotations
- Show how they will work on the signal and plot the signal in appropriate manner to be read by a doctor
- Identify the heart beat of the signal, average and total heart beat in the signal
- Identify the complex QRS in the signal and been able to annotate on it

As a result we expect a github project with and extructure that will include:
- Reference documentation used
- Jupyter Notebook, in an running environment, Colab, Docker.
- An explanation of the work done and lessons learned.


## Timeframe
It would be great if you could have this done within a week. If that's not doable for you, let us know early.

Also, we don't know how long this should take you, but we're not looking to reward the person that spends the most time on it. We believe in working smarter not harder.

## Tips
In case it's helpful, here's some other tips for you:

You can ask questions. This isn't a "bonus points if they ask questions" thing, just that we'll answer what we can if you need us. Like we would when working together.

We like to have a real work example work flow, we wencorage you to do a pull request and send the pull request for evaluation. 

You can request more information/data, but we'd rather you didn't. If you really need more, let us know, but there'd need to be a compelling reason for it.
## Summary
We want to see what it's like to work with you, and the quality of work you'd produce. This is a chance for both sides to see how that is.

We will be making a decision based on these tests, so do give it your best.

Thanks for giving this a go, we can't wait to see what you come up with.
