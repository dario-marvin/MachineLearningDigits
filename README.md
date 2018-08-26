# MachineLearningDigits

## Introduction
The idea of this project is the following: suppose you have to digitalise and extract information from some documents, and a high quality scanner is not available, or high quality scans are too expensive, in terms of scanning time or storage memory. Is it then still possible to recover the information contained on the (bad) scans? 

In short, we want to build a homemade Optical Character Recognition (OCR) machine using techniques form supervised Machine Learning (ML). For the moment, we only try to recover single digits, but an upgrade using a similar approach for recognizing letters and words is easily generalizable and will be implemented in the near future.  
~To verify our idea~, we create a dataset using a low quality scan of a sheet of paper containing a long sequence of digits. Each number on the sheet is individually extracted using techniques from image processing and the list of its pixel values is saved. The data are then analyzed with different ML classifiers and the results are plotted and commented.

## Methodology
### Document creation
To create our dataset we generated a random sequence of 4290 single digits, which can be observed in the file [`sequence.dat`](https://github.com/dario-marvin/MachineLearningDigits/blob/master/sequence.dat). A PDF file containing this sequence was then compiled using LaTeX and the resulting file was printed in draft quality, scanned at lowest quality setting (75 dpi) and finally saved as a PNG image.  
In the next figures you can see the upper part of the resulting PNG, and a detail of the upper left corner.

<p align="center">
  <img src="https://github.com/dario-marvin/MachineLearningDigits/blob/master/page1_ex.png">
</p>

<p align="center">
  <img width = 300 src="https://github.com/dario-marvin/MachineLearningDigits/blob/master/page1_particular.png">
</p>

### Dataset extraction

Of course, it is unthinkable to manually save the pixels of every single digit in the sequence, so we used image processing instead. As a first step, we compute the mean of the pixel values for every row of pixels composing the image. If there are only clear pixels, i.e. it is not a row containing numbers, its mean will be relatively high (remember white RBG value is 255 and black is 0). If instead we cross some darker pixels, the mean will be lower.  
Thus we select all rows with mean pixel value higher than a fixed threshold and to illustrate how the process works, we color them in white on the previous image.

<p align="center">
  <img src="https://github.com/dario-marvin/MachineLearningDigits/blob/master/page1_modified_ex.png">
</p>

At this point, the remaining stripes of pixels separated by white lines should contain all our numbers. In a perfect world, all stripes would have the same height, but sadly it's not our case, as it ranges from 7 to 9 pixel. So we decided to set universal height 7 pixels for every number image, and in case a stripe was 8 pixel tall we ignored the uppermost or lowermost row depending on which one has the highest pixel mean value (i.e. the clearest between the two, which should in theory contain less information).  

To capture the digit images on the stripes, the idea is similar to the proceeding method: for each column in the stripe we compute the mean of the pixel values. We start with the leftmost column and search until we obtain a mean value smaller than a fixed threshold, meaning that the column contains some dark pixels, hence a number. From there we select the previous column (even if it has only clear pixels) and the following 5, as that's the width in pixel of the average number. From there we start the search again until we find another number or we reach the end of the stripe.

In the end, each image of a digit will be composed of 6 pixels in width and 7 in height. We show here the images extracted for the first 8 digits in the sequence, together with their real value.

<p align="center">
  <img width=600 src="https://github.com/dario-marvin/MachineLearningDigits/blob/master/list_beginning.png">
</p>

### Data analysis

Since it is well known there is not a perfect general learning algorithm for every problem, we explore many different approaches and select the best one for this problem according to the results. We try some of the most common methods for ML classification, including:
- Decision tree learning (DT)
- k-nearest neighbors algorithm (KNN)
- Linear discriminant analysis (LDA)
- Gaussian naive Bayes (GNB)
- Support Vector Machines (SVM)

For each of these approaches we compare the predictions of the model with the real values for the 1000 images in the test set, and we also ask the algorithm to classify the images it used to train itself.

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

The results of the classification show some relatively good values, around 90% of success, although a couple of things should be noted: in the LDA analysis we receive a warning saying the priors do not sum to one and thus will be renormalized. Secondly, the SVM classifier performs poorly, as it classifies every image as the same value every time, thus getting the prediction right only 1/10 of the time.

The method that performs the best in both the train and test sets is the k-nearest neighbors algorithm, with a score of 0.997 in the test set, which means only 3 images were misclassified over the 1000 analyzed.  
For this classifier we print classification report and confusion matrix.

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
From the confusion matrix we understand that 2 images whose real value was 6 were wrongly classified as 0 instead, and one 4 was classified as 1. We plot the images in question, together with their real and predicted values.

<p align="center">
  <img src="https://github.com/dario-marvin/MachineLearningDigits/blob/master/wrong_predictions.png">
</p>


## Conclusion and future works
We showed it is possible to retrieve the correct values of some badly scanned digits with a precision of 99.7%. The classifier that seems to work best for this problem is the k-nearest neighbors approach.  
In the near future we plan to continue using this approach on more and more complicated data, such as single badly scanned letters of the alphabet, and later complete words extracted from a real scanned book.
