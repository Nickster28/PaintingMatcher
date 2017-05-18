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
    trimDatasetToThemes(filename, output, topTenThemes, headerRow=headerRow)


# Returns the top 10 themes in the dataset with the given filename
def getTopTenThemes(filename, headerRow=False):
    with open(filename, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        rows = [row[0] for row in reader]
        themesCounter = Counter(rows[1:] if headerRow else rows)
        return [entry[0] for entry in themesCounter.most_common(10)]

# Outputs only limit rows per theme in filename where the theme is contained in
# themes.  Picks rows randomly.  Outputs these rows to the output filename.
def trimDatasetToThemes(filename, output, themes, headerRow=False, limit=1000):
    with open(filename, 'rb') as csvfile:
        with open(output, 'wb') as outputfile:
            reader = csv.reader(csvfile, delimiter=',')
            writer = csv.writer(outputfile, delimiter=',')
            rows = [row for row in reader]
            if headerRow:
                rows = rows[1:]

            themesCounter = Counter()
            np.random.shuffle(rows)
            for row in rows:
                if row[0] in themes and themesCounter[row[0]] < limit:
                    writer.writerow(row)
                    themesCounter[row[0]] += 1

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
def makePairings(filename, output, headerRow=False, pairs=100, size=(200,200)):
    with open(filename, 'rb') as csvfile:
        with open(output, 'wb') as outputfile:
            reader = csv.reader(csvfile, delimiter=',')
            writer = csv.writer(outputfile, delimiter=',')
        
            rows = [row for row in reader if row[6].startswith("https://")]
            if headerRow:
                rows = rows[1:]
            counter = 0

            # Sample until we have enough pairs
            while counter < pairs:
                print(str(counter) + " of " + str(pairs))
                index1 = random.randint(0, len(rows) - 1)
                index2 = random.randint(0, len(rows) - 1)
                row1 = rows[index1]
                row2 = rows[index2]
                filename1 = "images/" + str(index1) + ".jpg"
                filename2 = "images/" + str(index2) + ".jpg"
                success1 = saveImage(row1[6], filename1, size)
                if not success1: continue
                success2 = saveImage(row2[6], filename2, size)
                if not success2:
                    os.remove(filename1)
                    continue

                sameTheme = row1[0] == row2[0]
                writer.writerow([filename1, filename2, 1 if sameTheme else 0])
                counter += 1


# Downloads image and resizes them.
# Returns true/false for success/failure.
def saveImage(url, filename, size):
    try:
        urllib.urlretrieve(url, filename)
        img = Image.open(filename)
        img = img.resize(size)
        img.save(filename)
        return True
    except Exception as e:
        return False

trimDataset('dataset.csv', 'dataset-trimmed.csv', headerRow=True)
outputThemeCounts('dataset-trimmed.csv', 'themes-tr.csv')
makePairings('dataset-trimmed.csv', 'dataset-trimmed-pairs.csv', pairs=100000)
