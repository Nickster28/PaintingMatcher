import csv
import urllib
from PIL import Image
import random
from scipy import misc
import numpy as np
from collections import Counter


# Trims the dataset with the given filename to the top 10 painting themes,
# and outputs this trimmed dataset to the given output filename.
def trimDataset(filename, output):
    topTenThemes = getTopTenThemes(filename)
    trimDatasetToThemes(filename, output, topTenThemes)


# Returns the top 10 themes in the dataset with the given filename
def getTopTenThemes(filename):
    with open(filename, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        themesCounter = Counter([row[0] for row in reader][1:])
        return [entry[0] for entry in themesCounter.most_common(10)]

# Outputs only the rows in filename where the theme is contained in themes.
# Outputs these rows to the output filename.
def trimDatasetToThemes(filename, output, themes):
    with open(filename, 'rb') as csvfile:
        with open(output, 'wb') as outputfile:
            reader = csv.reader(csvfile, delimiter=',')
            writer = csv.writer(outputfile, delimiter=',')
            rows = [row for row in reader][1:]

            for row in rows:
                if row[0] in themes:
                    writer.writerow(row)

# Outputs the number of paintings under each theme for the given dataset file
def outputThemeCounts(filename, output):
    with open(filename, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        themesCounter = Counter([row[0] for row in reader][1:])

        # Output theme counts
        with open(output, 'wb') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            for themeTuple in themesCounter.most_common():
                writer.writerow(themeTuple)





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