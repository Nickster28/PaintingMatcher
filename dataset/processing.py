import csv

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

#outputThemeCounts('dataset.csv', 'theme-counts.csv')
outputMalformedRows('dataset.csv')


