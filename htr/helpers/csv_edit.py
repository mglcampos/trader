import pandas as pd
import os
class CSVEdition():
    def __init__(self):
        self.header = ['Type', 'Day', 'Time', 'Open', 'High', 'Low', 'Close']
        self.dframes = []

    def load_csv(self, path, header = None):
        if header is not None:
            self.header = header
        if path is not None:
            for root, dirs, files in os.walk(self.path):
                for file in files:
                    if file.endswith('.csv') or file.endswith('.txt'):
                        self.dframes.append(pd.read_csv(os.path.join(root, file), names=self.header))
            # df = pd.read_csv(path, names=self.header)
            # print(len(df.columns))
            # print(df.head())
        else:
            ##todo set default path for csv's
            pass

        return self.dframes
