import csv
import urllib
from PIL import Image
import random

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


#outputThemeCounts('dataset.csv', 'theme-counts.csv')
#outputMalformedRows('dataset.csv')
downloadImages('dataset.csv', 'datset-new.csv', (200, 200), count=100)




