import os, sys
import pandas as pd

sys.path.append(".\\.")
from Data.readAllFile import readCSV
from Function.general import StoD

class rainFall:
    data = pd.DataFrame()

    def __init__(self, path: str = "", threshold: int = 0):
        self.data = readCSV(path, fileFilter=["rainfallStation.xlsx", "result.csv"]).toPandas(2, 4)
        self.data.drop(columns=["數據完整性/data Completeness"], inplace=True)
        self.data["fileName"] = self.data["fileName"].str.split('.', expand=True)[0].str.split('_', expand=True)[1]
        self.data.columns = ["Year", "Month", "Day", "Value", "StationName"]
        self.data["Year"] = self.data["Year"].astype(int)
        self.data["Month"] = self.data["Month"].astype(int).astype(str)
        self.data["Day"] = self.data["Day"].astype(int).astype(str)
        # Change Trace into 0
        self.data.loc[self.data["Value"].isin(["Trace", "***"]), ["Value"]] = 0
        self.data = self.data.loc[self.data["Year"] >= threshold]

        # Delete data when values in one day are all zero
        self.data["date"] = self.data["Year"].astype(str) + self.data["Month"].str.zfill(2) + self.data["Day"].str.zfill(2)
        self.data = self.data.groupby(["date"]).filter(lambda x: (x["Value"].astype(float) != 0).any())
    
    def addStationCord(self, path: str) -> None:
        station = pd.read_excel(path)
        station = station[["shortName", "E", "N"]]
        station.set_index("shortName", inplace=True)
        station["E"] = StoD(station["E"].copy())
        station["N"] = StoD(station["N"].copy())
        self.data = self.data.join(station, on="StationName")
        
        return

    def toCSV(self, path: str) -> None:
        self.data.to_csv(path, encoding="utf-8", index=False)

        return

if __name__ == "__main__":
    a = rainFall("Data\\rainfall", 2018)
    a.addStationCord("Data\\rainfall\\rainfallStation.xlsx")
    a.toCSV("Data\\rainfall\\result.csv")