import os
import pandas as pd

class read:
    files = []
    path = ""

    def __init__(self, path: str = "", fileFilter: list[str] = [], typeFilter: list[str] = []):
        self.path = path
        self.files = os.listdir(path)
        typeFilter += ["py"]
        for i in range(len(self.files) - 1, -1, -1):
            file = self.files[i]
            if file.split(".")[-1] in typeFilter or file in fileFilter:
                self.files.pop(i)

class readCSV(read):
    def toPandas(self, skip: int = 0, last: int = 0) -> pd.DataFrame:
        result = []
        for file in self.files:
            data = pd.read_csv(os.path.join(self.path, file), skiprows=skip)
            data = data[:-last]
            data.insert(data.shape[1], "fileName", file)
            result.append(data)
        
        return pd.concat(result)