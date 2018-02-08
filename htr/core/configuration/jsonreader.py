
import json

class JsonReader:
    """
        Attributes:
            filepath (string): path where the file will be read.
    """
    def __init__(self, filepath):
        self.name = "json_reader"
        self.filepath = filepath
        self.fieldnames = None

    def read(self):
        """
            Retruns:
                dictList[]: Returns a list with the information in the json file as a dictionary
            Raises:
                Input/reports error in case something fails reading the file.
        """
        dictList = []
        try:
            with open(self.filepath) as data_file:
                dictList = json.load(data_file)
                return dictList
        except IOError as e:
            print("Error reading json file.")
            print("{} - {}".format(e.errno,e.strerror))