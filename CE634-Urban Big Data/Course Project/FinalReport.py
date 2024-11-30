import pandas as pd
import numpy as np
from decimal import Decimal,ROUND_HALF_UP
import matplotlib.pyplot as plt
import calendar, os, gc

#The original round function of Python has different defination, 
#like result of round(1.335,2) is 1.33 rather than 1.34,
#So we need use decimal to creat a right round function
def rightRound(num: float, keep_n: int=0) -> Decimal:
    if isinstance(num,float):
        num=str(num)
    return Decimal(num).quantize((Decimal('0.' + '0'*keep_n)),rounding=ROUND_HALF_UP)

def fyearToRyear(nameList: list, *path: str) -> None:
    count=len(path)
    usecols=[
        "Summons Number", "Registration State", "Plate Type", "Issue Date", "Violation Code", "Vehicle Body Type", "Vehicle Make","Issuing Agency",
        "Violation Time", "Street Code1", "Vehicle Expiration Date", "Street Name"
    ]
    streetSuffix={
        " AVE$": " AVENUE", " AVE.$": " AVENUE", " AV$": " AVENUE", " AVENU$": " AVENUE",
        " ST$": " STREET", " ST.$": " STREET", " STR$": " STREET", " STREE$": " STREET", " STRET$": " STREET",
        " RD$": " ROAD", " RD.$": " ROAD",
        " DR":" DRIVE",
        " BDWAY$": " BROADWAY", " BDWY$": " BROADWAY", " BROAD$": " BROADWAY",
        " BL$": " BOULEVARD", " BLD$": " BOULEVARD", " BLV$": " BOULEVARD", " BLVD$": " BOULEVARD", " BLVD.$": " BOULEVARD", " BVD$": " BOULEVARD",
        " BRG$": " BRIDGE", " BRID$": " BRIDGE", " BRIDG$": " BRIDGE",
        " DR$": " DRIVEWAY", " DR.$": " DRIVEWAY", " DRIVE$": " DRIVEWAY",
        " EXP$": " EXPRESSWAY", " EXP$": " EXPRESSWAY", " EXPRE$": " EXPRESSWAY", " EXPRESS$": " EXPRESSWAY", " EXPRESSW$": " EXPRESSWAY",
        " EXPRESSWA$": " EXPRESSWAY", " EXPWAY$": " EXPRESSWAY", " EXPWY$": " EXPRESSWAY", " EXPY$": " EXPRESSWAY", " EXWAY$": " EXPRESSWAY", " EXWY$": " EXPRESSWAY",
        " LN$": " LANE",
        " LO$": " LOOP",
        " PK$": " PARK",
        " PKW$": " PARKWAY", " PKWAY$": " PARKWAY", " PKWY$": " PARKWAY", " PKY$": " PARKWAY", " PWAY$": " PARKWAY", " PWY$": " PARKWAY", " PY$": " PARKWAY",
        " PL$": " PLACE", " PL.$": " PLACE", " PLA$": " PLACE", " PLC$": " PLACE", " PLCE$": " PLACE",
        " PLZ$": " PLAZA", " PLZA$": " PLAZA",
        " ":"",
    }

    def normalized(df: pd.DataFrame) -> pd.DataFrame:
        df.dropna(how="all", inplace=True)
        df.drop(df.loc[(df["Street Name"].isna()) & (df["Street Code1"] == '0')].index, inplace=True)
        df.drop(df.loc[(df["Street Name"] == "--") | (df["Street Name"].str.isdigit())].index, inplace=True)
        df["Street Name"]=df["Street Name"].str.upper()
        df["Street Name"].replace(streetSuffix, regex=True, inplace=True)

        #Change the data whoes Street Code1 is 0 using the data who has the same Street Name
        codeMapping=df.loc[df["Street Code1"] != '0'].groupby("Street Code1")["Street Code1"].first()
        df.loc[df["Street Code1"] == '0', "Street Code1"]=df["Street Name"].map(codeMapping)

        # #Comapre with RND
        # #Too large to implement
        # df["fuzzName"]=df.apply(fuzzy_match, axis=1, df2=rowRoad)
        # df["fuzzName"]=df["fuzzName"].str[0]
        # print(df.shape[0])

        return df

    if len(nameList) + 1 != count:
        print("Wrong number of name list")
        return 1

    dfList=[]
    dfA=pd.read_csv(path[0], index_col=0, usecols=usecols, dtype=str)
    dfA=normalized(dfA)

    for i in range(count-1):
        dfB=dfA
        dfA=pd.read_csv(path[i + 1], index_col=0, usecols=usecols, dtype=str)
        dfA=normalized(dfA)
        fileName=nameList[i]
        df=pd.concat([dfA, dfB])
        df["Registration State"]=np.where(df["Registration State"] == "NY", "local", "nonlocal")
        df=df.loc[df["Issue Date"].str[-4:] == fileName]
        dfList.append(df)
    
    df=pd.concat(dfList)
    df.drop(df.loc[df["Street Code1"] == '0'].index, inplace=True)
    
    #Standardized time
    df["Violation Time"].fillna("TE", inplace=True)
    df["hr"]=pd.to_numeric(df["Violation Time"].str[0:2], errors='coerce')
    a=df["Violation Time"].str.contains("P|A")
    wrongVioTime=df[~a]["Violation Time"].unique().tolist()
    rightVioTime=df[a]["Violation Time"].unique().tolist()
    df.loc[(df["hr"]>12) & (df["Violation Time"].isin(rightVioTime)), ["Violation Time"]]=df.loc[(df["hr"]>12) & (df["Violation Time"].isin(rightVioTime))]["Violation Time"].str[0:2]
    rightVioTime=df[a]["Violation Time"].unique().tolist()
    df.loc[(df["hr"]>24) & (df["Violation Time"].isin(rightVioTime)), ["Violation Time"]]="TE"
    df.loc[df["Violation Time"].isin(wrongVioTime), ["Violation Time"]]=df.loc[df["Violation Time"].isin(wrongVioTime)]["Violation Time"].str[0:2]
    a=df["Violation Time"].str.contains("P|A")
    rightVioTime=df[a]["Violation Time"].unique().tolist()
    df.loc[df["Violation Time"].isin(rightVioTime), ["Violation Time"]]=df.loc[df["Violation Time"].isin(rightVioTime), ["Violation Time"]] + 'M'
    rightVioTime=df[a]["Violation Time"].unique().tolist()
    df.loc[df["Violation Time"].isin(rightVioTime), ["Violation Time"]]=df.loc[df["Violation Time"].isin(rightVioTime)]["Violation Time"].replace({"^00":"12"}, regex=True)
    rightVioTime=df[a]["Violation Time"].unique().tolist()
    df.loc[df["Violation Time"].isin(rightVioTime), ["Violation Time"]]=pd.to_datetime(df.loc[df["Violation Time"].isin(rightVioTime)]["Violation Time"], format="%I%M%p").dt.hour.astype(str)
    df.loc[df["Violation Time"] == "TE"]=None
    df["Violation Time"]=df["Violation Time"].astype(float)
    df.drop("hr", axis=1, inplace=True)
    df.to_csv("All.csv", encoding="utf-8")
    
    return 0

class analysis:
    def __init__(self, df: pd.DataFrame) -> None:
        self.issues=df
        self.num=df.shape[0]

        self.issues["year"]=self.issues["Issue Date"].str[-4:]
        self.issues["month"]=self.issues["Issue Date"].str[:2]
    
    def singlePattern(self, *years: str) -> None:
        numLocal=self.issues.loc[self.issues["Registration State"] == "local"].shape[0]
        perLocal=rightRound(numLocal / self.num * 100, 2)
        print("The total violation is {}".format(self.num))

        for i in years:
            n=self.issues.loc[self.issues["year"] == i].shape[0]
            print("The number of violation in {} is {}.".format(i, n))
        
        print("The number of violation of local is {}, which contains {}% of total issues.".format(numLocal, perLocal))
        print("The number of of violation of nonlocal is {}, which contains {}% of total issues.".format(self.num - numLocal, 100 - perLocal))
        print("The most common violation code is {}".format(self.issues["Violation Code"].mode()[0]))
        print("Most of the issues are issued by {}".format(self.issues["Issuing Agency"].mode()[0]))

        return 0

    def streetNum(self, *years: str) -> None:
        streetAll=[]

        for i in years:
            yearData=self.issues.loc[self.issues["year"] == i]
            print(yearData.shape[0])
            
            street=yearData.groupby(["Street Code1"]).size().reset_index(name=i + "Num")
            street.set_index(["Street Code1"], inplace=True)
            streetAll.append(street)
        
        for i in range(len(streetAll)-1):
            streetAll[0]=streetAll[0].join(streetAll[i+1], how="outer")
        streetAll[0].to_csv("streetVioNum.csv", encoding="utf-8")

        return 0

    def monthPattern(self, *years: str) -> None:
        monthList=[x for x in range(1, 13)]

        for i in years:
            yearData=self.issues.loc[self.issues["year"] == i]
            monthSet=yearData.groupby(["month"]).size().reset_index(name="num")
            monthSet.set_index("month")
            ava=rightRound(monthSet["num"].mean(), 0)
            
            # Draw Monthly Distribution
            distribution=monthSet["num"].plot.bar()
            distribution.set_xticks([x - 1 for x in monthList], (calendar.month_abbr[x] for x in monthList), rotation=0)
            distribution.set_xlabel("Months")
            distribution.set_ylabel("Number of issues")
            distribution.axhline(ava) #Draw line of average number
            path=os.path.join("monthPattern", i + "monthPattern.jpg")
            distribution.get_figure().savefig(path)
            plt.close()
        
        return 0
    
    def monthPatternSingle(self, *years: str) -> None:
        monthList=[x for x in range(1, 13)]
        allSet=[]

        for i in years:
            yearData=self.issues.loc[self.issues["year"] == i]
            monthSet=yearData.groupby(["month"]).size().reset_index(name=i)
            monthSet.set_index("month")
            allSet.append(monthSet)
        
        tempDf=pd.concat(allSet, axis=1)
        distribution=tempDf.plot.bar(rot=0)
        
        distribution.set_xticks([x - 1 for x in monthList], (calendar.month_abbr[x] for x in monthList), rotation=0)
        distribution.set_xlabel("Months")
        distribution.set_ylabel("Number of issues")
        plt.legend(years)
        
        path=os.path.join("monthPattern", "monthPattern.jpg")
        distribution.get_figure().savefig(path)
        plt.close()
        
        return 0
    
    def hourlyPattern(self, *years: str) -> None:     
        for year in years:
            issuesYear=self.issues.loc[self.issues["year"] == year]
            timeSeries=issuesYear.groupby(["Violation Time"]).size().reset_index(name="Num")
            timeSeries["Violation Time"]=timeSeries["Violation Time"].astype(float)
            timeSeries.sort_values(by="Violation Time", ignore_index=True, inplace=True)
            distribution=timeSeries["Num"].plot()
        distribution.set_xlabel("Hour of day (24 hours format)")
        distribution.set_ylabel("Summons Count")
        plt.legend(years)
        distribution.set_xticks([x for x in range(24)])
        
        path=os.path.join("monthPattern", "hourlyPatternALL.jpg")
        distribution.get_figure().savefig(path)
        plt.close()

        return 0

    # def hourlyPattern(self, year: list) -> None:
    #     issuesYear=self.issues.loc[self.issues["year"] == year]
    #     timeSeries=issuesYear.groupby(["Violation Time"]).size().reset_index(name="Num")
    #     timeSeries["Violation Time"]=timeSeries["Violation Time"].astype(float)
    #     timeSeries.sort_values(by="Violation Time", ignore_index=True, inplace=True)
    #     distribution=timeSeries["Num"].plot()
    #     distribution.set_xlabel("Hour of day (24 hours format)")
    #     distribution.set_ylabel("Summons Count")
    #     distribution.set_xticks([x for x in range(24)])
        
    #     path=os.path.join("monthPattern", "hourlyPattern" + year + ".jpg")
    #     distribution.get_figure().savefig(path)
    #     plt.close()

    #     return 0

if __name__ == "__main__":
    fyearToRyear(
        ["2021", "2022", "2023"],
        r"Parking_Violations_Issued_-_Fiscal_Year_2021_20240920.csv",
        r"Parking_Violations_Issued_-_Fiscal_Year_2022_20240920.csv",
        r"Parking_Violations_Issued_-_Fiscal_Year_2023_20240920.csv",
        r"Parking_Violations_Issued_-_Fiscal_Year_2024_20240920.csv"
    )
    
    df=pd.read_csv("All.csv", dtype=str, index_col=0)
    ana=analysis(df)
    years=["2021","2022","2023"]
    del df
    gc.collect()
    ana.singlePattern("2021","2022","2023")
    ana.streetNum("2021","2022","2023")
    # ana.monthPattern("2021","2022","2023")
    ana.monthPatternSingle("2021","2022","2023")
    ana.hourlyPattern("2021","2022","2023")