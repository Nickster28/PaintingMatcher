import csv
import urllib
from PIL import Image

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

def downloadImages(dataset, output):
    with open(dataset, 'rb') as datasetFile:
        with open(output, 'wb') as outputFile:
            writer = csv.writer(outputFile, delimiter=',')
            reader = csv.reader(datasetFile, delimiter=',')

            counter = 0
            for row in reader:
                counter += 1
                if counter == 1:
                    writer.writerow(row + ['Filename'])
                    continue

                if not row[6].startswith("https://"): continue
                filename = str(counter) + ".jpg"

                try:
                    urllib.urlretrieve(row[6], "images/" + filename)
                    img = Image.open("images/" + filename)
                    img = img.resize((300,300))
                    img.save("images/" + filename)
                    writer.writerow(row + [filename])
                    print("(" + str(counter) + " of " + str(35748) + ") - " + row[1])
                except Exception as e:
                    print(row[1] + " - error: " + str(e))


#outputThemeCounts('dataset.csv', 'theme-counts.csv')
#outputMalformedRows('dataset.csv')
downloadImages('dataset.csv', 'datset-new.csv')




