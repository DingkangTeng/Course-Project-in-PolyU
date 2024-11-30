from typing_extensions import override
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import PartialDependenceDisplay
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os,gc

class RF:
    def __init__(self, df: pd.DataFrame) -> None:
        self.issues=df
        self.issues["year"]=self.issues["Issue Date"].str[-4:]
    
    def evaluate(self, true: pd.Series, pred: pd.Series) -> None:
        print("accuracy:{:.2%}".format(metrics.accuracy_score(true, pred)))
        print("precision:{:.2%}".format(metrics.precision_score(true, pred, average="weighted")))
        print("recall:{:.2%}".format(metrics.recall_score(true, pred, average="weighted")))
        print("f1-score:{:.2%}".format(metrics.f1_score(true, pred, average="weighted")))

        return 0
    
    def classResult(self, rfModel: type[RandomForestClassifier | RandomForestRegressor], XTrain: pd.Series, path: str, top :int=0) -> pd.DataFrame:
        importances=rfModel.feature_importances_
        index=XTrain.columns
        ax=pd.DataFrame({'Feature': index, 'Importance': importances})
        ax.sort_values(by='Importance', inplace=True, ignore_index=True)
        ax.to_csv(path + ".csv", encoding="utf-8")
        if top != 0:
            ax=ax.tail(top)
            
        #Draw Bar
        graph=ax.plot.barh(x="Feature", figsize=(16, 6))
        graph.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: "{:.2%}".format(x)))
        for p in graph.patches:
            b=p.get_bbox()
            val="{:.2%}".format(b.x1)
            graph.annotate(val, (b.x1 + 0.002, b.y1 - 0.4))
        graph.set_ylabel("Variables")
        graph.set_xlabel("Importance")
        yTicks=plt.yticks()[1]
        yTicks=[label.get_text().capitalize() for label in yTicks]
        graph.set_yticks([x for x in range(len(yTicks))], yTicks)
        graph.get_figure().savefig(path + ".jpg")
            
        plt.close()
    
        return ax
    
    def cleanOutlier(self, data: pd.Series) -> pd.Series:
        data=data.str.upper()
        typeNum=data.value_counts()
        # Values whose repeated number is less than last skew will be regarded as invalid data
        # threshold=typeNum.skew()
        # threshold=typeNum.mean()
        wrongValue=typeNum[typeNum < 100].index.to_list()
        data.loc[data.isin(wrongValue)]=None

        return data

    def plotScatter(self, spec: str, source: list, target: str) -> None:
        data=self.issues[[spec, target]].loc[self.issues[spec].isin(source)].copy()
        data[target]=data[target].astype(int)
        data.sort_values(by=[spec, target], inplace=True)
        data.plot.scatter(x=spec, y=target)

        return 0

    # #Sensitivity analysis
    # def tuning(clf, XTrain, XTest, yTrain, yTest, parameter, parameter_name, parameterRange):
    #     def plot_metrics(parameter, accuracyAll, precisionAll, recallAll, f1All):
    #         plt.figure(figsize=(12, 6))
    #         plt.plot(accuracyAll, label="accuracy")
    #         plt.plot(precisionAll, label="precision")
    #         plt.plot(recallAll, label="recall")
    #         plt.plot(f1All, label="f1-score")
    #         plt.legend()
    #         plt.xlabel(parameter)
    #         plt.title(f"Evaluation Metrics for Models with Different {parameter}")
    #         plt.show()

    #     accuracyAll=pd.Series(index=parameterRange, dtype="float")
    #     precisionAll=pd.Series(index=parameterRange, dtype="float")
    #     recallAll=pd.Series(index=parameterRange, dtype="float")
    #     f1All=pd.Series(index=parameterRange, dtype="float")
    #     for parameterValue in parameterRange:
    #         clf=RandomForestClassifier(
    #             **{parameter: parameterValue}, random_state=42, n_jobs=-1
    #         )
    #         clf.fit(XTrain, yTrain)
    #         yPred=clf.predict(XTest)
    #         accuracyAll[parameterValue]=metrics.accuracy_score(yTest, yPred)
    #         precisionAll[parameterValue]=metrics.precision_score(yTest, yPred)
    #         recallAll[parameterValue]=metrics.recall_score(yTest, yPred)
    #         f1All[parameterValue]=metrics.f1_score(yTest, yPred)
    #     plot_metrics(parameter_name, accuracyAll, precisionAll, recallAll, f1All)

    def driverPattern(self) -> None:
        variable=["Registration State", "Plate Type", "Vehicle Body Type", "Vehicle Make", "Violation Time"]
        data=self.issues[variable + ["Violation Code", "year"]].copy()

        #Clean
        data.dropna(how="any", inplace=True)
        for i in variable[1:-1]:
            data[i]=self.cleanOutlier(data[i])
        data.dropna(how="any", inplace=True)
        gc.collect()

        for i in variable:
            data[i]=data[i].astype("category")
            if i in variable[1:-1]:
                data[i]=pd.factorize(data[i])[0]
        
        for year in ["2021","2022","2023"]:
            dataYear = data.loc[data["year"] == year]
            X=pd.get_dummies(dataYear[variable[0]])
            X=pd.concat([X, dataYear[variable[1:]]], axis=1)
            y=dataYear["Violation Code"]
            del dataYear
            gc.collect()

            XTrain, XTest, yTrain, yTest=train_test_split(X, y, test_size=0.2, random_state=42)
            rfModel=RandomForestClassifier(n_estimators=128, random_state=42, verbose=1, n_jobs=-1)
            rfModel.fit(XTrain, yTrain)
            yPred=rfModel.predict(XTest)
            self.evaluate(yTest, yPred)
            path=os.path.join("driver","varCode"+year)
            self.classResult(rfModel, XTrain, path)

        return 0
    
    def driverSpecific(self, spec: str) -> None:
        data=self.issues[[spec, "Violation Code"]].copy()

        #Clean
        data.dropna(how="any", inplace=True)
        data[spec]=self.cleanOutlier(data[spec])
        data.dropna(how="any", inplace=True)
        data[spec]=data[spec].astype("category")
        gc.collect() # Memory management
        
        X=pd.get_dummies(data[spec])
        # X=X.loc[:,~(X == False).all(axis=0)] # Delete columns whoes values are all False
        gc.collect() # Memory management
        y=data["Violation Code"]

        XTrain, XTest, yTrain, yTest=train_test_split(X, y, test_size=0.2, random_state=42)
        rfModel=RandomForestClassifier(n_estimators=128, random_state=42, verbose=1, n_jobs=-1)
        rfModel.fit(XTrain, yTrain)
        yPred=rfModel.predict(XTest)
        self.evaluate(yTest, yPred)
        path=os.path.join("driver", spec)
        self.classResult(rfModel, XTrain, path, 10)        

        return 0

class parkingRF(RF):
    def __init__(self, df: pd.DataFrame, interval: list) -> None:
        super().__init__(df)
        self.vioName=["Y2021", "Y2022", "Y2023", "allViolation", "Y22_Y21", "Y23_Y22"]
        self.vioNum=self.issues[["ORIG_FID"]+self.vioName].copy()
        self.vioNum.drop_duplicates(inplace=True)
        self.vioNum.set_index("ORIG_FID", inplace=True)
        self.vioNum.fillna(0, inplace=True)
        # self.vioNum["allViolationCat"]=pd.cut(self.vioNum["allViolation"], bins=[0, 930, 9700, 18000, 27000, 36000, 45000, 490000], labels=[x for x in range(7)], right=True)
        # self.vioNum["Y22_Y21"]=pd.cut(self.vioNum["Y22_Y21"], bins=[-7000, -2000, -1700, -1300, -910, -540, -160, -210, 580, 950, 1300, 1700, 26000], labels=[x for x in range(12)], right=True)
        self.interval=interval
        self.yearName=["Y2021", "Y2022", "Y2023", "allViolation"]
    
    def modify(self, df: pd.DataFrame, distance: str, num: list) -> pd.DataFrame:
        dfOut=[]
        for i in self.interval:
            intDf=df.loc[df[distance] == i].drop(columns=distance)
            intDf.columns=[x + " (" + str(i) + "M)" for x in intDf.columns]
            dfOut.append(intDf)
        dfOut=pd.concat(dfOut + [self.vioNum[num]], axis=1)
        dfOut.fillna(0, inplace=True)

        return dfOut
    
    @override
    def evaluate(self, true: np.ndarray, pred: np.ndarray, path: str) -> None: #true:yTest, pred:yPred
        ax=pd.DataFrame({"True Values": true, "Predictions": pred})
        m, b = np.polyfit(true, pred, 1)
        graph=ax.plot.scatter(x="True Values", y="Predictions", label="Predictions")
        plt.plot(true.values, m * true.values + b, color="red", label="Regression line")

        R2=metrics.r2_score(true, pred)
        #sklearn.__version__ == 1.3.2
        try:
            RMSE=metrics.mean_squared_error(true, pred)**0.5
        except:
            #sklearn.__version__ >=1.4
            RMSE=metrics.root_mean_squared_error(true, pred)**0.5
        print("R2 is {:.4f} and MSE is {:.4f}".format(R2, RMSE))
        
        legend = [
            Line2D([0], [0], color="blue", marker="o", linestyle="None", label="Data Points"),
            Line2D([0], [0], color="red", linestyle="-", label="Regression Line"),
            Line2D([0], [0], color="none", label="R2 = {:.4f} \nRMSE = {:.4f}".format(R2, RMSE))  # Additional text
        ]
        plt.legend(handles=legend)

        graph.get_figure().savefig(path + ".jpg")
        plt.close()

        return 0
    
    def PDP(self, rfModel: type[RandomForestRegressor], X: pd.DataFrame, path: str, num: int=2) -> None:
        fig, ax = plt.subplots(figsize=(15, 32))
        fig.subplots_adjust(wspace=0.2)
        PDP=PartialDependenceDisplay.from_estimator(rfModel, X, X.columns, n_cols=num, n_jobs=-1, ax=ax)
        fig.savefig(path + ".jpg")
        plt.close()
        
        return 0
    
    def liner(self, XTrain: pd.Series, yTrain: pd.Series, xlabel: str, ylabel:str, path: str) -> None:
        lineModel=LinearRegression(n_jobs=-1)
        lineModel.fit(XTrain, yTrain)

        plt.scatter(XTrain, yTrain, color="blue", label="Training Data")
        plt.plot(XTrain, lineModel.predict(XTrain), color="green", label="Regression Line")

        legend = [
            Line2D([0], [0], linestyle="-", color="green", label="Regression Line"),
            Line2D([0], [0], color="none", label="y = {:.4f}x + {:.4f}\nR2 = {:.4f}".format(lineModel.coef_[0], lineModel.intercept_, lineModel.score(XTrain, yTrain)))  # Additional text
        ]
        plt.legend(handles=legend)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        plt.savefig(path + ".jpg")
        plt.close()
        
        return 0
    
    def parkingPattern(self):
        parkPattern=self.issues[["ORIG_FID", "PkLotNum2022", "PkMeterNum", "distance"]]
        parkPattern.set_index("ORIG_FID", inplace=True)
        parkPattern.columns=["Parking Lot", "Parking Meter", "distance"]
        parkPattern=self.modify(parkPattern, "distance", ["Y2022"])

        X=parkPattern.drop(columns=["Y2022"])
        y=parkPattern["Y2022"]

        XTrain, XTest, yTrain, yTest=train_test_split(X, y, test_size=0.2, random_state=42)
        rfModel=RandomForestRegressor(n_estimators=128, random_state=42, verbose=1, n_jobs=-1)
        rfModel.fit(XTrain, yTrain)
        yPred=rfModel.predict(XTest)
        path=os.path.join("parking","parkPatternAccuracy")
        self.evaluate(yTest, yPred, path)
        path=os.path.join("parking","parkPattern")
        self.classResult(rfModel, XTrain, path)

        return 0

    def lotPattern(self, distance: int, tag: str) -> None:
        self.issues["pkLotDiff"]=self.issues["PkLotNum2022"] - self.issues["PkLotNum2021"]
        lotPattern=self.issues.loc[self.issues["distance"] == distance, ["pkLotDiff", "Y22_Y21"]].copy()
        
        if tag == "n":
            lotPattern=lotPattern.loc[lotPattern["Y22_Y21"]<0]
        elif tag == "p":
            lotPattern=lotPattern.loc[lotPattern["Y22_Y21"]>=0]
        
        XTrain=lotPattern["pkLotDiff"].values.reshape(-1, 1)
        yTrain=lotPattern["Y22_Y21"]
        xlabel="Change in the Number of Parking Lot from 2021 to 2022"
        ylabel="Change in the Number of Violations from 2021 to 2022"
        path=os.path.join("parking","lotPattern" + tag)
        
        self.liner(XTrain, yTrain, xlabel, ylabel, path)
        
        return 0
    
    def landPattern(self):
        landPattern=self.issues[["ORIG_FID", "distance", "ResNum", "PublicNum", "TransportNum", "IndusNum", "CommNum"]]
        landPattern.set_index("ORIG_FID", inplace=True)
        landPattern.columns=["distance", "Resident", "Public", "Transportation", "Industry", "Commerce"]
        landPattern=self.modify(landPattern, "distance", self.yearName)

        X=landPattern.drop(columns=self.yearName)

        for year in self.yearName:
            gc.collect()
            y=landPattern[year]

            XTrain, XTest, yTrain, yTest=train_test_split(X, y, test_size=0.2, random_state=42)
            rfModel=RandomForestRegressor(n_estimators=128, random_state=42, verbose=1, n_jobs=-1)
            rfModel.fit(XTrain, yTrain)
            yPred=rfModel.predict(XTest)
            path=os.path.join("land","landPattern" + year + "Accuracy")
            self.evaluate(yTest, yPred, path)
            path=os.path.join("land","landPattern" + year)
            ax = self.classResult(rfModel, XTrain, path, 10)
            
            #PDP
            path=os.path.join("land","landPattern" + year + "PDP")
            self.PDP(rfModel, X, path, 5)

        return 0
    
    def linerCor(self, a: str, xlabel:str, b: str, ylabel: str) -> None:
        linerCor=self.issues[["ORIG_FID", a, b]]
        
        XTrain=linerCor[a].values.reshape(-1, 1)
        yTrain=linerCor[b]
        
        path=os.path.join("land",a + "LinerWith" + b)
        
        self.liner(XTrain, yTrain, xlabel, ylabel, path)

if __name__ == "__main__":
    df=pd.read_csv("All.csv", dtype=str, index_col=0)
    ana=RF(df)
    del df
    gc.collect()

    ana.driverPattern()
    ana.driverSpecific("Violation Time")
    ana.plotScatter("Violation Time", ["8.0", "9.0", "11.0", "19.0"], "Violation Code")
    ana.driverSpecific("Vehicle Body Type")
    ana.plotScatter("Vehicle Body Type", ["VAN", "UT", "DELV", "SD"], "Violation Code")

    df=pd.read_csv("streetAtt.csv")
    parking=parkingRF(df, [50, 100, 200, 500, 1000, 2000, 5000])
    parking.parkingPattern()
    for i in ["n","p","all"]:
        parking.lotPattern(5000, i)
    parking.landPattern()
    parking.linerCor("CommNum", "Number of Commercial Lands", "TransportNum", "Number of Transportation Lands")