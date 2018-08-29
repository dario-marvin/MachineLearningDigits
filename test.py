from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Load the number sequence that will be used partly for training the classifier and partly for verifying the
# exactness of the predictions

sequence = np.genfromtxt('sequence.dat', delimiter=' ', dtype=int)

# Load the scanned image 

im = Image.open('page1.png')
pixels = im.load()
[x, y] = im.size

# To recognize where the numbers are, collect the pixels rows whose mean value is smaller than a certain threshold.
# Save the image for visual purposes

clear_lines = []

for row in range(y):

    mean = 0

    for column in range(x):
        mean += pixels[column, row][0]
    if mean / x > 215:
        clear_lines.append(row)
        for column in range(x):
            pixels[column, row] = (255, 255)

im.save('page1_modified.png')

# Transform the list of clear_lines we just computed into a list of intervals for better  treatment of the data
intervals = []
current = clear_lines[0]
for i in range(len(clear_lines) - 1):
    if clear_lines[i] + 1 != clear_lines[i + 1]:
        intervals.append(list(range(current, clear_lines[i] + 1)))
        current = clear_lines[i + 1]
intervals.append(list(range(current, y)))

# Now that we know where the numbers are more or less located, we can analyze each row to retrieve the pixels
# composing the numbers. The idea is to store each number in a list of 42 numbers, each image being 6 pixels wide and
# 7 tall
pix = []

for i in range(len(intervals) - 1):
    start = intervals[i][-1] + 1
    end = intervals[i + 1][0] - 1

    # In case the height is not 7 pixels already, ignore the row containing the least information

    if end - start == 6:
        pass

    elif end - start == 7:
        mean1 = 0
        mean2 = 0
        for column in range(x):
            mean1 += pixels[column, start][0]
            mean2 += pixels[column, end][0]

        mean1 /= x
        mean2 /= x

        if mean1 > mean2:
            end = end - 1
        else:
            start = start + 1

    elif end - start == 8:
        start = start + 1
        end = end - 1

    # Now we search where a number starts by computing the mean over the columns until we find one whose value is
    # lower than a threshold, ie it contains some dark pixels, then we skip 5 pixels, as that is the width of the
    # average number and then start the search again

    column = 0
    while column < x:
        mean = 0
        for row in range(start, end + 1):
            mean += pixels[column, row][0]
        if mean / 7 < 200:
            current_column = column - 1
            pixel_list = []
            for yy in range(start, end + 1):
                for xx in range(current_column, current_column + 6):
                    pixel_list.append(pixels[xx, yy][0])
            pix.append(pixel_list)
            column += 5
        column += 1

# Some assertions to be sure we did everything right

assert (len(pix) == len(sequence))

for p in pix:
    assert (len(p) == 42)

# Random check for testing every image is linked to the correct corresponding number in the sequence

# ~ for i in range(10):
# ~ r = randint(0, 4290)
# ~ p = np.reshape(pix[r], (7,6))
# ~ array = np.array(p, dtype=np.uint8)
# ~ filename = 'check' + str(r) + '.png'
# ~ new_image = Image.fromarray(array)
# ~ new_image.save(filename)
# ~ print(r, sequence[r])

# Plot some images of the first 8 images in the sequence

#~ for i in range(8):
    #~ image = np.reshape(pix[i], (7, 6))
    #~ plt.subplot(2, 4, i + 1)
    #~ plt.xticks([])
    #~ plt.yticks([])
    #~ plt.imshow(image, cmap=cm.Greys_r, interpolation='nearest')
    #~ plt.title('index: ' + str(i + 1))
    #~ plt.xlabel('value: ' + str(sequence[i]))
#~ plt.show()

# We use the first 3290 numbers to train the predictor, and leave the remaining 1000 as test 

n = len(pix)

y_train = sequence[: n - 100]
X_train = pix[: n - 100]

y_test = sequence[n - 100:]
X_test = pix[n - 100:]

### Decision Tree

#~ DTclassifier = DecisionTreeClassifier().fit(X_train, y_train)
#~ DTclassifier_score_train = DTclassifier.score(X_train, y_train)
#~ DTclassifier_score_test = DTclassifier.score(X_test, y_test)
#~ print('Accuracy of Decision Tree classifier on training set: ' + str(DTclassifier_score_train))
#~ print('Accuracy of Decision Tree classifier on test set: ' + str(DTclassifier_score_test))
#~ print()

### K Nearest Neighbors

#~ KNNclassifier = KNeighborsClassifier().fit(X_train, y_train)
#~ KNNclassifier_score_train = KNNclassifier.score(X_train, y_train)
#~ KNNclassifier_score_test = KNNclassifier.score(X_test, y_test)
#~ print('Accuracy of K-NN classifier on training set: ' + str(KNNclassifier_score_train))
#~ print('Accuracy of K-NN classifier on test set: ' + str(KNNclassifier_score_test))
#~ print()

### Linear Discriminant Analysis

#~ LDAclassifier = LinearDiscriminantAnalysis().fit(X_train, y_train)
#~ LDAclassifier_score_train = LDAclassifier.score(X_train, y_train)
#~ LDAclassifier_score_test = LDAclassifier.score(X_test, y_test)
#~ print('Accuracy of LDA classifier on training set: ' + str(LDAclassifier_score_train))
#~ print('Accuracy of LDA classifier on test set: ' + str(LDAclassifier_score_test))
#~ print()

### Gaussian Naive Bayes

#~ GNBclassifier = GaussianNB().fit(X_train, y_train)
#~ GNBclassifier_score_train = GNBclassifier.score(X_train, y_train)
#~ GNBclassifier_score_test = GNBclassifier.score(X_test, y_test)
#~ print('Accuracy of GNB classifier on training set: ' + str(GNBclassifier_score_train))
#~ print('Accuracy of GNB classifier on test set: ' + str(GNBclassifier_score_test))
#~ print()

### Support Vector Machines

SVCclassifier = SVC().fit(X_train, y_train)
SVCclassifier_score_train = SVCclassifier.score(X_train, y_train)
SVCclassifier_score_test = SVCclassifier.score(X_test, y_test)
print('Accuracy of SVM classifier on training set: ' + str(SVCclassifier_score_train))
print('Accuracy of SVM classifier on test set: ' + str(SVCclassifier_score_test))
print()

# Plot some images of the data

#~ N = 5
#~ train_set = (DTclassifier_score_train, KNNclassifier_score_train, LDAclassifier_score_train, GNBclassifier_score_train,
             #~ SVCclassifier_score_train)
#~ test_set = (DTclassifier_score_test, KNNclassifier_score_test, LDAclassifier_score_test, GNBclassifier_score_test,
            #~ SVCclassifier_score_test)

#~ ind = np.arange(N) + .15  # the x locations for the groups
#~ width = 0.35  # the width of the bars
#~ fig, ax = plt.subplots(figsize=(8, 6))

#~ extra_space = 0.05
#~ ax.bar(ind, train_set, width, color='r', label='train')
#~ ax.bar(ind + width + extra_space, test_set, width, color='b', label='test')

#~ ax.set_ylabel('Score')
#~ ax.set_title('Classifiers comparison')
#~ ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=2)
#~ ax.set_xticks(ind + width + extra_space)
#~ ax.set_xticklabels(('DT', 'KNN', 'LDA', 'GNB', 'SVC'))

#~ plt.show()

# Print report and confusion matrix for KNN classifier     

predicted = SVCclassifier.predict(X_test)
print(classification_report(y_test, predicted))
print()
print(confusion_matrix(y_test, predicted))

# Retrieve and plot the images that confuse the KNN predictor

#~ wrong_predictions = []

#~ for i in range(len(predicted)):
    #~ if predicted[i] != y_test[i]:
        #~ img = [pix[3290 + i], predicted[i], y_test[i]]
        #~ wrong_predictions.append(img)

#~ for index, image in enumerate(wrong_predictions):
    #~ img = np.reshape(image[0], (7, 6))
    #~ plt.figure.figsize = (8, 4)
    #~ plt.subplot(1, 3, index + 1)
    #~ plt.xticks([])
    #~ plt.yticks([])
    #~ plt.imshow(img, cmap=cm.Greys_r, interpolation='nearest')
    #~ plt.title('Real: ' + str(image[2]))
    #~ plt.xlabel('Predicted: ' + str(image[1]))
#~ plt.show()
