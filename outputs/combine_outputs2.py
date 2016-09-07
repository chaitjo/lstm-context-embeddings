import csv

with open('outputs-cv.csv', 'ab') as of:
	writer = csv.writer(of)
	with open("06.09.16/output-normal.csv", "rb") as f:
		reader = csv.reader(f)
		outputs = []
		for row in reader:
			outputs.append(row[2])
		writer.writerow(['No CV'] + outputs)	
	with open("06.09.16/output-cnn.csv", "rb") as f:
		reader = csv.reader(f)
		outputs = []
		for row in reader:
			outputs.append(row[2])
		writer.writerow(['CNN only'] + outputs)
