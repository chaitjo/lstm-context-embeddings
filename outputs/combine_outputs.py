import csv

with open('outputs-cv.csv', 'wb') as of:
	writer = csv.writer(of)
	for i in range(10):
		with open("output-cv-{}.txt".format(i), "r") as file:
			lines = file.readlines()
			outputs = []
			for line_no in range(len(lines)):
				if lines[line_no] == 'Evaluation:\n':
					line_no += 1
					outputs.append(lines[line_no][-9:-2])
		writer.writerow(['Fold-{}'.format(i)] + outputs)


