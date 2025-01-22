import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from matplotlib.ticker import ScalarFormatter
from decimal import Decimal,ROUND_HALF_UP
from datetime import datetime
import calendar,os

#The original round function of Python has different defination, 
#like result of round(1.335,2) is 1.33 rather than 1.34,
#So we need use decimal to creat a right round function
def rightRound(num,keep_n: float=0) -> Decimal:
    if isinstance(num,float):
        num=str(num)
    return Decimal(num).quantize((Decimal('0.' + '0'*keep_n)),rounding=ROUND_HALF_UP)

#Distance calculation function using haversine
def getDistanceHav(lat0: float, long0: float, lat1: float, long1: float, codType: str) -> float:
    from numpy import sin, cos, arcsin, radians, fabs, sqrt
    EARTH_RADIUS=6371
    lat0=radians(lat0)
    long0=radians(long0)
    lat1=radians(lat1)
    long1=radians(long1)
    dlat=fabs(lat0 - lat1)
    dlong=fabs(long0 - long1)

    def hav(theta):
        s=sin(theta / 2)
        return s * s
    
    #WGS-84
    if codType == "WGS84":
        h=hav(dlat) + cos(lat0) * cos(lat1) * hav(dlong)
        distance=2 * EARTH_RADIUS * arcsin(sqrt(h)) * 1000
    else:
        return 0

    return distance

class assignment:
    #It has been checked that all data in datasets is valid. Or we need do some data cleaning.
    def __init__(self, datasets: pd.DataFrame) -> None:
        self.taxi=datasets
        self.taxi.dropna(inplace=True)

class assignment1(assignment):
    def __init__(self, datasets: pd.DataFrame) -> None:
        super().__init__(datasets)
        self.taxiIDSet=pd.DataFrame(index=self.taxiID)
        self.dateSet=pd.DataFrame(index=pd.date_range('2011-01-01', periods=365, freq='D'))
        self.taxiID=datasets["taxi_id"].unique()
        self.tripNum=self.taxi.shape[0]
        self.taxiNum=len(self.taxiID)
        self.avaTripPreDay=rightRound(self.tripNum / 365, 0)

    def task1(self):
        print("The number of unique taxis in this dataset is %d" % self.taxiNum)
        print("The number of trips recorded is %d" % self.tripNum)

        return 0

    def task2(self, savePath: str) -> None:
        # Add a new column to count the number of trips per taxi
        num=self.taxi["taxi_id"].value_counts()
        self.taxiIDSet=self.taxiIDSet.join(num)
        self.taxiIDSet.rename(columns={"count": "trip"}, inplace=True)

        # Find the top performers
        maxTrip=self.taxiIDSet.max().iloc[0]
        topPerformers=self.taxiIDSet.idxmax().iloc[0]
        commonTrip=self.taxiIDSet.mode().iloc[0, 0]
        print("The top performers is the taxi with ID %d who has %d trips in total." % (topPerformers, maxTrip))
        print("The most common trip number is %d ." % (commonTrip))

        # Distributuion
        # Set x-axis interval
        i=0
        step=int(self.taxiNum / 4000) * 1000 #Dived the x-axis into 8 parts
        xticks=[0]
        while i < maxTrip:
            i += step
            xticks.append(i)
        # Draw the histogram
        distribution=self.taxiIDSet["trip"].plot.hist(bins=len(xticks) * 5, xticks=xticks) #int(self.taxiNum / 100)
        distribution.set_ylim(0, 900)
        yticks=distribution.get_yticks()
        self.taxiIDSet.boxplot(column="trip", ax=distribution, sym='', positions=[820], widths=10, color='red', vert=False)
        distribution.set_yticks(yticks, yticks)
        distribution.yaxis.set_major_formatter(ScalarFormatter())
        distribution.ticklabel_format(style="sci", axis='y', scilimits=(0, 0))
        distribution.set_xlabel("Number of trips")
        distribution.get_figure().savefig(savePath)
        plt.close()

        return 0

    def task3(self, savePath1: str, savePath2: str) -> None:
        print("The average number of trips per day is %d." % self.avaTripPreDay)

        dateSeries=pd.DataFrame({"date": self.taxi["pick_up_time"].dt.date})
        num=dateSeries["date"].value_counts()
        self.dateSet=self.dateSet.join(num)
        self.dateSet.rename(columns={"count": "trip"}, inplace=True)

        dayExceed=self.dateSet.loc[self.dateSet["trip"] > self.avaTripPreDay].shape[0]
        print("There are %d days' number of trip exceed the average." % dayExceed)

        cov=rightRound(self.dateSet["trip"].std() / self.dateSet["trip"].mean() * 100, 2)
        print("The coefficient of variation of the number of trips per day is " + str(cov) + "%.")

        # Daily Distribution
        distribution=self.dateSet["trip"].plot.bar()
        monthList=[x for x in range(1, 13)]
        monthRange=pd.date_range("1969-12-31", periods=12, freq="M")
        distribution.set_xticks(monthRange,(calendar.month_abbr[x] + "1" for x in monthList), rotation=0)
        distribution.set_xlabel("Dates")
        distribution.set_ylabel("Number of trips")
        distribution.axhline(self.avaTripPreDay) #Draw line of average number
        for i in monthRange:
            distribution.axvline(i, linestyle="--", linewidth=0.5, color="black") #Draw Distribution line
        distribution.get_figure().savefig(savePath1)
        plt.close()

        # Monthly Distributuion
        self.dateSet["month"]=self.dateSet.index
        self.dateSet["month"]=self.dateSet["month"].dt.month
        num=self.dateSet.groupby(by=["month"])["trip"].sum()
        monthSet=pd.DataFrame(index=monthList)
        monthSet=monthSet.join(num)
        avaTripPreMonth=rightRound(monthSet["trip"].mean(), 0)
        print("The average number of trips per month is %d." % avaTripPreMonth)
        monthExceed=monthSet.loc[monthSet["trip"] > avaTripPreMonth].shape[0]
        print("There are %d months' number of trip exceed the average." % monthExceed)
        cov=rightRound(monthSet["trip"].std() / monthSet["trip"].mean() * 100, 2)
        print("The coefficient of variation of the number of trips per month is " + str(cov) + "%.")

        # Draw Monthly Distribution
        distribution=monthSet["trip"].plot.bar()
        distribution.set_xticks([x - 1 for x in monthList], (calendar.month_abbr[x] for x in monthList), rotation=0)
        distribution.set_xlabel("Months")
        distribution.set_ylabel("Number of trips")
        distribution.axhline(avaTripPreMonth) #Draw line of average number
        distribution.get_figure().savefig(savePath2)
        plt.close()

        return 0
    
    def task4(self, intersectionSet: pd.DataFrame, savePath: str) -> None:
        pickUpNum=self.taxi["pick_up_intersection"].value_counts()
        dropOffNum=self.taxi["drop_off_intersection"].value_counts()
        intersectionSet=intersectionSet.join(pickUpNum)
        intersectionSet.rename(columns={"count": "pickUpNum"}, inplace=True)
        intersectionSet=intersectionSet.join(dropOffNum)
        intersectionSet.rename(columns={"count": "dropOffNum"}, inplace=True)
        intersectionSet=intersectionSet.fillna(0)
        intersectionSet.to_csv(savePath, encoding="utf-8")

        return 0
    
    def task5(self, dates: list, savePath: str=r"") -> None:
        pickUpDate=pd.DataFrame({"date": self.taxi["pick_up_time"].dt.date, "time": self.taxi["pick_up_time"].dt.hour})
        dropOffDate=pd.DataFrame({"date": self.taxi["drop_off_time"].dt.date, "time": self.taxi["drop_off_time"].dt.hour})
        DateSet=[pickUpDate, dropOffDate]
        timeSeries=[x for x in range(0,24)]
        timeSet=pd.DataFrame(index=timeSeries)
        for date in dates:
            differenceSet=[]
            for i in [0,1]:
                subSetPick=DateSet[i].loc[DateSet[i]["date"] == datetime.strptime(date, "%Y/%m/%d").date()]
                num=subSetPick.value_counts("time")
                subTimeSet=timeSet.join(num)
                distribution=subTimeSet["count"].plot()
                differenceSet.append(subTimeSet)
            
            #Set plot style
            plt.legend(["Pick Up", "Drop Off"])
            plt.grid(axis='x', linestyle="--")
            distribution.set_xticks(timeSeries)
            distribution.set_xlabel("Hours")
            distribution.set_ylabel("Number of trips")
            outputPath=os.path.join(savePath, date.replace("/","-") + ".jpg")
            distribution.get_figure().savefig(outputPath)
            plt.close()
            
            #Draw daliy difference
            difference=differenceSet[0] - differenceSet[1]
            differencePlot=difference.plot(legend=None)
            plt.grid(axis='x', linestyle="--")
            differencePlot.axhline(0, linestyle="--")
            differencePlot.set_xticks(timeSeries)
            differencePlot.set_xlabel("Hours")
            differencePlot.set_ylabel("Difference of trips")
            outputPath=os.path.join(savePath, date.replace("/","-") + "Difference.jpg")
            differencePlot.get_figure().savefig(outputPath)
            plt.close()

        return 0
    
    def task6(self, intersectionSet: pd.DataFrame, savePath1: str, savePath2: str) -> None:
        # Drop same destination and pickup? DO NOT need.
        # self.newTaxi=self.taxi.drop(self.taxi.loc[self.taxi["pick_up_intersection"]==self.taxi["drop_off_intersection"]].index)
        self.taxi["travelTime"]=self.taxi["drop_off_time"] - self.taxi["pick_up_time"]
        self.taxi=self.taxi.join(intersectionSet, on="pick_up_intersection")
        #This may need a large memary more than 32GB...
        self.taxi.rename(columns={"latitude": "pickLat", "longitude": "pickLong"}, inplace=True)
        self.taxi=self.taxi.join(intersectionSet, on="drop_off_intersection")
        self.taxi.rename(columns={"latitude": "dropLat", "longitude": "dropLong"}, inplace=True)

        #Calculate the stright-line distance
        self.taxi["distance"]=getDistanceHav(self.taxi["pickLat"], self.taxi["pickLong"], self.taxi["dropLat"], self.taxi["dropLong"], "WGS84")
        
        #Distance distribution
        maxDistance=self.taxi["distance"].max()
        topPerformers=self.taxi["taxi_id"].loc[self.taxi["distance"].idxmax()]
        print("The longest trip in distance is the taxi with ID %d who dorve %.2f KM in one trip." % (topPerformers, rightRound(maxDistance / 1000, 2)))
        avaDistance=rightRound(self.taxi["distance"].mean()/1000, 2)
        print("The average distance pre trips is %.2f KM." % avaDistance)
        # Set x-axis interval
        i=0
        step=int(maxDistance / 5000) * 1000
        xticks=[0]
        while i < maxDistance:
            i += step
            xticks.append(i)
        # Draw the histogram
        distribution=self.taxi["distance"].plot.hist(bins=len(xticks) * 5, xticks=xticks)
        yticks=distribution.get_yticks()
        self.taxi.boxplot(column="distance", ax=distribution, sym='', positions=[3.2*10**7], widths=1.25*10**6, color='red', vert=False)
        distribution.set_yticks(yticks, yticks)
        distribution.yaxis.set_major_formatter(ScalarFormatter())
        distribution.ticklabel_format(style="sci", axis='y', scilimits=(0, 0))
        distribution.set_xlabel("Distances of trips (Meter)")
        distribution.get_figure().savefig(savePath1)
        plt.close()

        #Drop time < 0d. Speed may not be a suitbale index.
        self.newTaxi=self.taxi.drop(self.taxi.loc[self.taxi["travelTime"]<=pd.Timedelta(0)].index)

        #Delete the data exceed x+_3std
        self.newTaxi["travelTime"]=self.newTaxi["travelTime"].dt.seconds
        timeMean=self.newTaxi["travelTime"].mean()
        timeStd=self.newTaxi["travelTime"].std()
        threshold=3 * timeStd
        self.newTaxi["isOutlier"]=abs(self.newTaxi["travelTime"] - timeMean) > threshold
        self.newTaxi=self.newTaxi[~self.newTaxi["isOutlier"]]

        #Travel Time, using newTaxi
        maxTime=self.newTaxi["travelTime"].max()
        print("The longest trip in time is " + str(maxTime // 60) + "mins" + str(maxTime % 60) + "seconds.")
        avaTime=self.newTaxi["travelTime"].mean()
        print("The average time pre trips is " + str(int(avaTime // 60)) + "mins" + str(rightRound(avaTime % 60, 0)) + "seconds.")
        
        # Set x-axis interval
        i=0
        step=600
        xticks=[0]
        while i < maxTime:
            i += step
            xticks.append(i)
        # Draw the histogram
        distribution=self.newTaxi["travelTime"].plot.hist(bins=len(xticks) * 5, xticks=xticks)
        yticks=distribution.get_yticks()
        self.newTaxi.boxplot(column="travelTime", ax=distribution, sym='', positions=[1.9*10**7], widths=1.25*10**6, color='red', vert=False)
        distribution.set_yticks(yticks, yticks)
        distribution.yaxis.set_major_formatter(ScalarFormatter())
        distribution.ticklabel_format(style="sci", axis='y', scilimits=(0, 0))
        distribution.set_xlabel("Trips time (Seconds)")
        distribution.get_figure().savefig(savePath2)
        plt.close()

        self.newTaxi=self.newTaxi.drop(self.newTaxi.loc[self.newTaxi["distance"] == 0].index)
        self.newTaxi["avaSpeed"]=self.newTaxi["distance"] / self.newTaxi["travelTime"] * 3.6
        avaSpeed=self.newTaxi["avaSpeed"].mean()
        print("The average time pre trips is " + str(rightRound(avaSpeed,2)) + "KM/h.")

        return 0
    
if __name__ == "__main__":
    df=pd.read_csv("taxi_id.csv", names=["taxi_id", "pick_up_time", "drop_off_time", "pick_up_intersection", "drop_off_intersection"])
    df["pick_up_time"]=pd.to_datetime(df["pick_up_time"],unit='s')
    df["drop_off_time"]=pd.to_datetime(df["drop_off_time"],unit='s')
    assignment1=assignment1(df)
    assignment1.task1()
    assignment1.task2("DistributionOfTheNumOfTripsPerTaxi.jpg")
    assignment1.task3("DistributionOfDailyTrip.jpg", "DistributionOfMonthlyTrip.jpg")
    dfIntersection=pd.read_csv("intersections.csv", names=["latitude", "longitude"])
    assignment1.task4(dfIntersection,"intersectionPoints.csv")
    assignment1.task5(["2011/8/19", "2011/11/11", "2011/12/16"])
    assignment1.task6(dfIntersection, "DistributionOfDistance.jpg","DistributionOfTravelTime.jpg")