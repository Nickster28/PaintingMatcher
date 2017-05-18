import csv
import urllib
from PIL import Image
import random
from scipy import misc
import numpy as np
from collections import Counter
import os


# Trims the dataset with the given filename to the top 10 painting themes,
# and outputs this trimmed dataset to the given output filename.
def trimDataset(filename, output, headerRow=False):
    topTenThemes = getTopTenThemes(filename, headerRow=headerRow)
    trimDatasetToThemes(filename, output, topTenThemes)


# Returns the top 10 themes in the dataset with the given filename
def getTopTenThemes(filename, headerRow=False):
    with open(filename, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        rows = [row[0] for row in reader]
        themesCounter = Counter(rows[1:] if headerRow else rows)
        return [entry[0] for entry in themesCounter.most_common(10)]

# Outputs only limit rows per theme in filename where the theme is contained in
# themes.  Picks rows randomly.  Outputs these rows to the output filename.
def trimDatasetToThemes(filename, output, themes, limit=1000):
    with open(filename, 'rb') as csvfile:
        with open(output, 'wb') as outputfile:
            reader = csv.reader(csvfile, delimiter=',')
            writer = csv.writer(outputfile, delimiter=',')
            rows = [row for row in reader if row[6].startswith("https://")]
            newRows = []

            themesCounter = Counter()
            np.random.shuffle(rows)
            for i, row in enumerate(rows):
                print(i)
                if row[0] in themes and themesCounter[row[0]] < limit:
                    filename = str(i) + ".jpg"
                    success = saveImage(row[6], "images/" + filename)
                    if success:
                        newRows.append([row[0], filename])
                        themesCounter[row[0]] += 1

            # Shuffle before writing out
            np.random.shuffle(newRows)
            for row in newRows:
                writer.writerow(row)

# Outputs the number of paintings under each theme for the given dataset file
def outputThemeCounts(filename, output, headerRow=False):
    with open(filename, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        rows = [row[0] for row in reader]
        themesCounter = Counter(rows[1:] if headerRow else rows)

        # Output theme counts
        with open(output, 'wb') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            for themeTuple in themesCounter.most_common():
                writer.writerow(themeTuple)

# Outputs to file a list of randomly-chosen pairs from filename.  Downloads the
# images for the pairs, and records their filenames and whether they are the
# same or different thematically.
def makePairings(filename, output, pairs=100):
    with open(filename, 'rb') as csvfile:
        with open(output, 'wb') as outputfile:
            reader = csv.reader(csvfile, delimiter=',')
            writer = csv.writer(outputfile, delimiter=',')
            rows = [row for row in reader]
        
            for i in range(pairs):
                row1 = rows[random.randint(0, len(rows) - 1)]
                row2 = rows[random.randint(0, len(rows) - 1)]
                sameTheme = row1[0] == row2[0]
                writer.writerow([row1[1], row2[1], 1 if sameTheme else 0])


# Downloads image.
# Returns true/false for success/failure.
def saveImage(url, filename):
    try:
        urllib.urlretrieve(url, filename)
        return True
    except Exception as e:
        return False

#trimDataset('dataset.csv', 'dataset-trimmed.csv', headerRow=True)
#outputThemeCounts('dataset-trimmed.csv', 'themes-tr.csv')
#makePairings('dataset-trimmed.csv', 'dataset-trimmed-pairs.csv', pairs=100000)
