# MachineLearningDigits

# !!! WORK IN PROGRESS. COME BACK LATER!!!

## Introduction
The key idea of this project is the following: suppose you have to digitalise and extract information from some documents, and a high quality scanner is not available, or high quality scans are too expensive in terms of time or storage memory. Is it then still possible to recover the information contained on the (bad) scans? 

In practice, we want to build a homemade Optical Character Recognition (OCR) machine using techniques form supervised Machine Learning (ML).
For the moment, we only consider digits, but an upgrade using a similar approach for recognizing letters and words is easily generalizable and will be implemented in the near future.
To verify our idea, we created a dataset using a low quality scan of a sheet of paper containing a long sequence of digits. Each number on the sheet is individually extracted using techniques from image processing and the list of its pixel values is saved. The data are then analyzed with different ML classifiers and the results are plotted and commented.    


## Methodology
### Document creation
To create our dataset we generated a random sequence of 4290 single digits that can be observed in the file [`sequence.dat`](https://github.com/dario-marvin/MachineLearningDigits/blob/master/sequence.dat). This sequence was then pasted into a normal A4 paper and a PDF file was created using LaTeX. The resulting file was printed in draft quality and subsequently scanned at lowest quality setting (75 dpi) and saved as a PNG image. 

In the next images you can see part of the resulting PNG image, and a detail of the upper left corner.

<p align="center">
  <img src="https://github.com/dario-marvin/MachineLearningDigits/blob/master/page1_ex.png">
</p>

<p align="center">
  <img width = 300 src="https://github.com/dario-marvin/MachineLearningDigits/blob/master/page1_particular.png">
</p>

### Dataset extraction

Of course, it would be unthinkable to manually save the pixels of every single digit in the sequence, so we used image processing instead. As a first step, for every row of pixels composing the image we compute its mean. If there are only clear pixels, i.e. it is not a row containing numbers, its mean will be relatively high (remember white RBG value is 255 and black is 0). If instead we cross some darker pixels, the mean will be lower.

Thus we select all rows with mean pixel value higher than a fixed threshold and color them in white, to show how the process works.

<p align="center">
  <img src="https://github.com/dario-marvin/MachineLearningDigits/blob/master/page1_modified_ex.png">
</p>


### Data analysis


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
