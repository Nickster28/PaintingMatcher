import csv
import urllib
from PIL import Image
import random
from scipy import misc
import numpy as np

# Outputs the number of paintings under each theme for the given dataset file
def outputThemeCounts(dataset, output):
    with open(dataset, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        atStart = True
        themeDict = {}

        for row in reader:
            if atStart:
                atStart = False
                continue

            # Theme
            if row[0] in themeDict:
                themeDict[row[0]] += 1
            else:
                themeDict[row[0]] = 1

        # Output theme counts
        with open(output, 'wb') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')

            # Theme
            writer.writerow(["Themes"])
            for key in themeDict:
                key = unicode(key, "utf-8")
                writer.writerow([key, themeDict[key]])

# Prints out the row numbers with malformed data
def outputMalformedRows(dataset):
    with open(dataset, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        counter = 0

        for row in reader:
            counter += 1
            if counter == 1:
                continue

            if not row[6].startswith("https://"):
                print(counter)

# Downloads count random images for the dataset and puts them in images/
# Makes a new dataset file recording the filenames for the entries we chose.
def downloadImages(dataset, output, thumbnailSize, count=None):
    with open(dataset, 'rb') as datasetFile:
        with open(output, 'wb') as outputFile:
            writer = csv.writer(outputFile, delimiter=',')
            reader = csv.reader(datasetFile, delimiter=',')

            counter = 0
            rows = [row for row in reader if row[6].startswith("https://")]
            writer.writerow(rows[0] + ['Filename'])
            rows = rows[1:]

            if count == None: count = len(rows)

            while count > 0:
                index = random.randint(0, len(rows)) # Pick random index
                row = rows[index]
                filename = str(index) + ".jpg"

                try:
                    urllib.urlretrieve(row[6], "images/" + filename)
                    img = Image.open("images/" + filename)
                    img = img.resize(thumbnailSize)
                    img.save("images/" + filename)
                    writer.writerow(row + [filename])
                    print(str(count) + " - " + row[1])
                    del rows[index]
                    count -= 1
                except Exception as e:
                    print(row[1] + " - error: " + str(e))

# Returns an N x 200 x 400 x 3 tensor representing our input
def readData(filename, pairs=100):
    with open(filename, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        rows = [row for row in reader]

        # We want 50% same, 50% different
        numSame = pairs / 2
        numDifferent = pairs / 2

        input = []   # Build up our input tensor
        ys = []      # The labels for each input

        # Add random pairs
        while (numSame + numDifferent > 0):
            index1 = random.randint(0, len(rows) - 1)
            index2 = random.randint(0, len(rows) - 1)
            sameTheme = rows[index1][0] == rows[index2][0]

            # Add them if we still need another same or different
            if (sameTheme and numSame > 0) or (not sameTheme and numDifferent > 0):
                if sameTheme:
                    numSame -= 1
                else:
                    numDifferent -= 1
                input.append(loadImages(rows[index1][1], rows[index2][1]))
                ys.append(1 if sameTheme else 0)

        # Shuffle
        indices = np.arange(len(input))
        np.random.shuffle(indices)
        shuffled_input = []
        shuffled_ys = []
        for index in indices:
            shuffled_input.append(input[index])
            shuffled_ys.append(ys[index])

        return (np.concatenate(shuffled_input), np.array(shuffled_ys))


# Loads the two images in and returns a tensor of them concatenated together on axis 1
def loadImages(filename1, filename2):
    image1 = misc.imread("images/" + filename1)
    image2 = misc.imread("images/" + filename2)
    c = np.concatenate((image1, image2), axis=1)
    return c.reshape([1] + list(c.shape))

#outputThemeCounts('dataset.csv', 'theme-counts.csv')
#outputMalformedRows('dataset.csv')
#downloadImages('dataset.csv', 'datset-mini.csv', (200, 200), count=100)

# SAMPLE CODE

# Returns (100, 200, 400, 3) and (100,) tensors
# (100, 200, 400, 3) is 100 concatenated images of size 200x200x3
# (100,) is a 1-hot vector of labels where 1 means same theme, 0 means different
# input, labels = readData('dataset-mini.csv')


