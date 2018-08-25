# MachineLearningDigits
## Introduction
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
