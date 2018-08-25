# MachineLearningDigits

# !!! WORK IN PROGRESS. COME BACK LATER!!!

## Introduction
The key idea of this project is the following: suppose you have to digitalise some documents, but an high quality scanner is not available, or high quality scans are too expensive in terms of time or storage memory.
Is it still possible to recover the information contained on the (bad) scans? 


In this project we want to build an Optical Character Recognition (OCR) machine using techniques form Machine Learning (ML). To do so we created our own dataset using a low quality scan of a sheet of paper containing a long sequence of digits. Each number on the sheet is then extracted using techniques from image processing and a list of the pixels is saved. The pixels are then analyzed with different ML classifiers and the results are plotted    


## Methodology
#### Dataset creation
To create our dataset we generated a random sequence of 4290 space separated digits that can be observed in the file [`sequence.dat`](https://github.com/dario-marvin/MachineLearningDigits/blob/master/sequence.dat) and pasted them in a normal A4 PDF file using LaTeX. The resulting sheet of paper looked like in the next figure

<p align="center">
  <img width = 600 src="https://github.com/dario-marvin/MachineLearningDigits/blob/master/sequence_ex.png">
</p>

The sheet of paper was then scanned at the lowest resolution available (75 dpi) and saved as a PNG image

<p align="center">
  <img src="https://github.com/dario-marvin/MachineLearningDigits/blob/master/page1_ex.png">
</p>

<p align="center">
  <img width = 300 src="https://github.com/dario-marvin/MachineLearningDigits/blob/master/page1_particular.png">
</p>

Of course, it would be unthinkable to manually save the pixels of every single digit in the sequence, so we used image processing instead. As a first step, for every pixel row in the image we compute the mean of the values of the pixels composing it, and select those with a value smaller than a fixed threshold, i.e. whose pixels are generally light and thus do not contain numbers. To visualize this process, we colored in white the lines that are supposed not contain numbers in the previous image

<p align="center">
  <img src="https://github.com/dario-marvin/MachineLearningDigits/blob/master/page1_modified_ex.png">
</p>


#### Data analysis


## Results

```

Accuracy of Decision Tree classifier on training set: 1.0
Accuracy of Decision Tree classifier on test set: 0.872

Accuracy of K-NN classifier on training set: 0.998480243161
Accuracy of K-NN classifier on test set: 0.997

/usr/local/lib/python3.5/dist-packages/sklearn/discriminant_analysis.py:442: UserWarning: The priors do not sum to 1. Renormalizing
  UserWarning)
Accuracy of LDA classifier on training set: 0.948024316109
Accuracy of LDA classifier on test set: 0.901

Accuracy of GNB classifier on training set: 0.887841945289
Accuracy of GNB classifier on test set: 0.858

Accuracy of SVM classifier on training set: 1.0
Accuracy of SVM classifier on test set: 0.09

```

<p align="center">
  <img src="https://github.com/dario-marvin/MachineLearningDigits/blob/master/classifier_comparison.png">
</p>

```             precision    recall  f1-score   support

          0       0.98      1.00      0.99        85
          1       0.99      1.00      0.99        98
          2       1.00      1.00      1.00        99
          3       1.00      1.00      1.00        95
          4       1.00      0.99      1.00       115
          5       1.00      1.00      1.00       101
          6       1.00      0.98      0.99        95
          7       1.00      1.00      1.00       113
          8       1.00      1.00      1.00        90
          9       1.00      1.00      1.00       109

avg / total       1.00      1.00      1.00      1000


[[ 85   0   0   0   0   0   0   0   0   0]
 [  0  98   0   0   0   0   0   0   0   0]
 [  0   0  99   0   0   0   0   0   0   0]
 [  0   0   0  95   0   0   0   0   0   0]
 [  0   1   0   0 114   0   0   0   0   0]
 [  0   0   0   0   0 101   0   0   0   0]
 [  2   0   0   0   0   0  93   0   0   0]
 [  0   0   0   0   0   0   0 113   0   0]
 [  0   0   0   0   0   0   0   0  90   0]
 [  0   0   0   0   0   0   0   0   0 109]]


```

<p align="center">
  <img src="https://github.com/dario-marvin/MachineLearningDigits/blob/master/wrong_predictions.png">
</p>
