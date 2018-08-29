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

        if mean1 < mean2:
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

# Experiments

n = len(pix)
sizes = list(range(50,3300,50)) + [3290]
scores_dt = []
scores_knn = []
scores_lda = []
scores_gnb = []
scores_svm = []

for size in sizes: 
    y_train = sequence[:size]
    X_train = pix[:size]
    y_test = sequence[n - 1000:]
    X_test = pix[n - 1000:]

    DTclassifier = DecisionTreeClassifier().fit(X_train, y_train)
    scores_dt.append(DTclassifier.score(X_test, y_test))

    KNNclassifier = KNeighborsClassifier().fit(X_train, y_train)
    scores_knn.append(KNNclassifier.score(X_test, y_test))

    LDAclassifier = LinearDiscriminantAnalysis().fit(X_train, y_train)
    scores_lda.append(LDAclassifier.score(X_test, y_test))

    GNBclassifier = GaussianNB().fit(X_train, y_train)
    scores_gnb.append(GNBclassifier.score(X_test, y_test))

    #~ SVCclassifier = SVC().fit(X_train, y_train)
    #~ scores_svm.append(SVCclassifier.score(X_test, y_test))


plt.figure()
plt.plot(sizes, scores_dt, 'r-*', label='DT')
plt.plot(sizes, scores_knn, 'b-^', label='KNN')
plt.plot(sizes, scores_lda, 'g-s', label='LDA')
plt.plot(sizes, scores_gnb, 'm-<', label='GNB')
#~ plt.plot(sizes, scores_svm, 'c->', label='SVM')

plt.axvline(x=210, color= 'k')
plt.ylabel('Score')
plt.xlabel('Train set size')
plt.legend(loc='center right')
plt.show()

		








