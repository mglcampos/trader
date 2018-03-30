
import csv


class CsvReader():
	"""
		Attributes:
			filepath (string): path where the file will be read.
	"""

	def __init__(self, filepath):
		self.name = "csv_reader"
		self.filepath = filepath
		self.fieldnames = None

	def read(self):
		"""
			Retruns:
				dictList[]: Returns a list with a dictionary for each line in the read file
			Raises:
				Input/output error in case something fails reading the file.
		"""
		dictList = []
		try:
			with open(self.filepath) as csvfile:
				reader = csv.DictReader(csvfile, delimiter=';', dialect='excel')
				self.fieldnames = reader.fieldnames
				for row in reader:
					dictList.append(row)
				return dictList

		except IOError as e:
			print("Error reading csv file.")
			print("{} - {}".format(e.errno, e.strerror))
