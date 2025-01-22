import pandas as pd
import geopandas as gpd
import igraph as ig
import os

# Run in relative path due to VSCode runs in the user directory by defult
import os,sys
os.chdir(sys.path[0])
sys.path.append(os.getcwd())

INFINITY=4294967295

class assignment:
    # It has been checked that all data in datasets is valid. Or we need do some data cleaning.
    def __init__(self, datasets: pd.DataFrame) -> None:
        self.taxi=datasets
        self.taxi.dropna(inplace=True)

class assignment2(assignment):
    def __init__(self, datasets: pd.DataFrame, zone: pd.DataFrame) -> None:
        super().__init__(datasets)
        self.taxi=self.taxi.join(zone, on="pick_up_intersection")
        self.taxi.rename(columns={"zone_id": "pickZone"}, inplace=True)
        self.taxi=self.taxi.join(zone, on="drop_off_intersection")
        self.taxi.rename(columns={"zone_id": "dropZone"}, inplace=True)
        # Data cleaning
        self.exception(self.taxi)
        self.taxi.drop(columns=["pick_up_intersection", "drop_off_intersection"], inplace=True)
        self.taxi.set_index("pickZone", inplace=True)
        self.taxi["month"]=self.taxi["pick_up_time"].dt.month
        self.taxi["pick_up_time"]=self.taxi["pick_up_time"].dt.hour
        # Because some trips ended on the second day, the time of every trip will be counted based on the pick-up time.
        self.t7Tot9=self.taxi.loc[(self.taxi["pick_up_time"] >= 7) & (self.taxi["pick_up_time"] <= 9)]
        self.t16Tot18=self.taxi.loc[(self.taxi["pick_up_time"] >= 16) & (self.taxi["pick_up_time"] <= 18)]
    
    def toGraph(self, zone: pd.DataFrame, path: str="") -> None:
        zoneList=zone["zone_id"].unique()
        zoneMat=pd.DataFrame(index=zoneList, columns=zoneList)
        time=["m"+str(x) for x in range(1,13)]
        time.extend(["t7Tot9", "t16Tot18"])
        dfAll=[self.taxi.loc[self.taxi["month"] == x] for x in range(1,13)]
        dfAll.extend([self.t7Tot9, self.t16Tot18])
        for i in range(14):
            zoneMat[:]=INFINITY
            fileName=time[i] + ".csv"
            df=dfAll[i].drop(columns=["pick_up_time","month"])
            print("Is generating " + fileName)
            # O(n2), j is pick up and k is drop off
            for j in zoneList:
                for k in zoneList:
                    print("Calculateing [%d][%d]" % (j, k))
                    dfSub=df.loc[int(j)]
                    dfSub=dfSub.loc[dfSub["dropZone"] == int(k)]
                    zoneMat.loc[j][k]=dfSub.shape[0]
            # # O(n) but may consumpt more time?
            # for j in range(df.shape[0]):
            #     row=int(df.loc[j])
            #     col=int(df.iloc[j]["dropZone"])
            #     if zoneMat[row][col] == INFINITY:
            #         zoneMat[row][col]=1
            #     else:
            #         zoneMat[row][col] +=1
            savePath=os.path.join(path, fileName)
            zoneMat.to_csv(savePath, encoding="utf-8")

        return 0

class calCommunity:
    def __init__(self, adjMat: str) -> None:
        adjMat=pd.read_csv(adjMat, index_col=0)
        # Different defination of unconnected nodes in igraph
        adjMat.replace(INFINITY, 0, inplace=True)
        matrix=adjMat.values.tolist()
        # plus means change the graph into undirect and the weight of the edge between vertex i and j is A(i, j) + A(j, i)
        self.graph=ig.Graph.Weighted_Adjacency(matrix, mode="plus", attr="weight", loops="once")
        self.label=adjMat.index.values
        self.graph.vs["name"]=self.label
        # edge_width=self.graph.es["weight"]
        self.attribute=None
    
    def showGraph(self) -> None:
        import matplotlib.pyplot as plt
        layout=self.graph.layout("kk")
        fig, ax=plt.subplots()
        ig.plot(self.graph, layout=layout, target=ax)
        plt.show()

        return 0
    
    def creatCommunity(self) -> None:
        result=ig.Graph.community_multilevel(self.graph, weights="weight")
        membership=result.membership
        self.attribute=pd.DataFrame({
            "Vertex": self.label,
            "Community": membership,
        })
        
        return 0

    def linkShp(self, baseVector: gpd, savePath: str) -> None:
        joinResult=baseVector.join(self.attribute.set_index("Vertex"), on="LocationID")
        output=gpd.GeoDataFrame(joinResult, crs="EPSG:4326")
        output.set_geometry("geometry")
        output.to_file(savePath, driver="ESRI Shapefile", encoding="utf-8")

        return 0
    
if __name__ == "__main__":
    # Derive the flow network at the level of taxi zones and generate directed adjacency matritx
    df=pd.read_csv("taxi_id.csv", usecols=[1,3,4], names=["pick_up_time", "pick_up_intersection", "drop_off_intersection"])
    df["pick_up_time"]=pd.to_datetime(df["pick_up_time"],unit='s')
    dfZone=pd.read_csv("intersection_to_zone.csv", index_col="inter_id")
    assignment2=assignment2(df, dfZone)
    assignment2.toGraph(dfZone, "adjacentMat")

    # Community detection
    fileList=["m"+str(x) for x in range(1,13)]
    fileList.extend(["t7Tot9", "t16Tot18"])
    dbFile=r"D:\\OneDrive - The Hong Kong Polytechnic University\\Subject\\Urban Big Data\\Report\\GISProject\\CE634Assignment.gdb"
    gdb=gpd.read_file(dbFile, layer="taxiZonesMa")
    for fileName in fileList:
        path=os.path.join(r"adjacentMat",fileName + ".csv")
        print("Generating %s" % path)
        community=calCommunity(path)
        community.creatCommunity()
        savePath=os.path.join("joinShp", fileName + ".shp")
        community.linkShp(gdb, savePath)