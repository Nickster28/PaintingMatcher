import csv

def outputDict(theme):
	with open('output.csv', 'wb') as csvfile:
		writer = csv.writer(csvfile, delimiter=',')

		# Theme
		writer.writerow(["Themes"])
		for key in theme:
			key = unicode(key, "utf-8")
			writer.writerow([key, theme[key]])


with open('by-topic.csv', 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    counter = 0
    
    startIndex = -1
    length = 0

    theme_dict = {}
    for row in reader:
    	counter += 1;
        if counter == 1: continue

        # Theme
        if row[0] in theme_dict:
        	theme_dict[row[0]] += 1
        else:
        	theme_dict[row[0]] = 1

        if not row[5].startswith("https://") or not row[6].startswith("https://"):
            if startIndex == -1:
                startIndex = counter
            elif counter == startIndex + length + 1:
                length += 1
            elif startIndex != -1:
                print(str(startIndex) + " to " + str(startIndex + length))
                startindex = counter
                length = 1

    outputDict(theme_dict)