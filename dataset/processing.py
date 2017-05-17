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

def createMiniDataset(dataset, newFilename, size=100):
    with open(dataset, 'rb') as datasetFile:
        with open(newFilename, 'wb') as outputFile:
            writer = csv.writer(outputFile, delimiter=',')
            reader = csv.reader(datasetFile, delimiter=',')

            counter = 0
            rows = [row for row in reader if row[6].startswith("https://")][1:]

            while size > 0:
                index = random.randint(0, len(rows) - 1) # Pick random index
                row = rows[index]
                filename = str(index) + ".jpg"

                try:
                    urllib.urlretrieve(row[6], "images/" + filename)
                    img = Image.open("images/" + filename)
                    img = img.resize((200, 200))
                    img.save("images/" + filename)
                    writer.writerow([row[0], filename])
                    print(str(size) + " - " + row[1])
                    del rows[index]
                    size -= 1
                except Exception as e:
                    print(row[1] + " - error: " + str(e))   


# Returns an N x 200 x 400 x 3 tensor representing our input
def readData(filename, pairs=100):
    with open(filename, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')

        input = []   # Build up our input tensor
        ys = []      # The labels for each input
    
        for row in reader:
            input.append(loadImages(row[0], row[1]))
            ys.append(row[2])

        return (np.concatenate(input), np.array(ys))


# Loads the two images in and returns a tensor of them concatenated together on axis 1
def loadImages(filename1, filename2):
    image1 = misc.imread("images/" + filename1)
    if len(image1.shape) != 3:
        newImage1 = np.zeros(image1.shape + (3,))
        for row in range(image1.shape[0]):
            for col in range(image1.shape[1]):
                newImage1[row][col][0] = image1[row][col]
                newImage1[row][col][1] = image1[row][col]
                newImage1[row][col][2] = image1[row][col]
        image1 = newImage1

    image2 = misc.imread("images/" + filename2)
    if len(image2.shape) != 3:
        newImage2 = np.zeros(image2.shape + (3,))
        for row in range(image2.shape[0]):
            for col in range(image2.shape[1]):
                newImage2[row][col][0] = image2[row][col]
                newImage2[row][col][1] = image2[row][col]
                newImage2[row][col][2] = image2[row][col]
        image2 = newImage2

    c = np.concatenate((image1, image2), axis=1)
    return c.reshape([1] + list(c.shape))


def createPairsDataset(datasetFile, output, pairs=100):
    with open(datasetFile, 'rb') as csvfile:
        with open(output, 'wb') as outputFile:
            writer = csv.writer(outputFile, delimiter=',')
            reader = csv.reader(csvfile, delimiter=',')

            rows = [row for row in reader]
            newRows = []

            # We want 50% same, 50% different
            numSame = pairs / 2
            numDifferent = pairs / 2

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
                    newRows.append([rows[index1][1], rows[index2][1], 1 if sameTheme else 0])

            # Shuffle
            np.random.shuffle(newRows)
            for row in newRows:
                writer.writerow(row)


def loadPaintingsDataset():
    X_train, y_train = readData('train-final.csv')
    X_val, y_val = readData('dev-final.csv')
    X_test, y_test = readData('test-final.csv')
    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test
    }

# SAMPLE CODE

# Returns (100, 200, 400, 3) and (100,) tensors
# (100, 200, 400, 3) is 100 concatenated images of size 200x200x3
# (100,) is a 1-hot vector of labels where 1 means same theme, 0 means different
# input, labels = readData('dataset-mini.csv')

# createMiniDataset("dataset.csv", "train.csv")
# createMiniDataset("dataset.csv", "dev.csv")
# createMiniDataset("dataset.csv", "test.csv")
# createPairsDataset("train.csv", "train-final.csv")
# createPairsDataset("dev.csv", "dev-final.csv")
# createPairsDataset("test.csv", "test-final.csv")

