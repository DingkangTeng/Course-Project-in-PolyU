import arcpy, os, sys
from datetime import datetime, timedelta

sys.path.append(".") # Set path to the roots
from Function.general import randomName

CODEBLOCK_1 = """
def calculate(a):
 if a == 0:
  return 0
 else:
   return 1
"""

class creatFishnet:
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Creating fishnet"
        self.description = "This tool are used to fusion different data into one fishnet for analysis."

    def getParameterInfo(self):
        """Define the tool parameters."""
        fishnet = arcpy.Parameter(
            displayName="Fishnet Layer",
            name="fishnetLayer",
            datatype="GPFeatureLayer",
            parameterType="Required",
            direction="Input"
        )
        # Landslide
        landslide = arcpy.Parameter(
            displayName="Landslide Layer",
            name="landslideLayer",
            datatype="GPFeatureLayer",
            parameterType="Required",
            direction="Input"
        )
        landslideType = arcpy.Parameter(
            displayName="Select field of landslide type",
            name="landslideType",
            datatype="Field",
            parameterType="Required",
            direction="Input"
        )
        landslideType.parameterDependencies = [landslide.name]
        landslideType.filter.list = ["Text"]
        landslideDate = arcpy.Parameter(
            displayName="Select field of landslide date",
            name="landslideDate",
            datatype="Field",
            parameterType="Required",
            direction="Input"
        )
        landslideDate.parameterDependencies = [landslide.name]
        landslideDate.filter.list = ["Text"]
        # Rainfall
        rainfall = arcpy.Parameter(
            displayName="Rainfall Layer (Daily)",
            name="rainfallLayer",
            datatype="GPFeatureLayer",
            parameterType="Required",
            direction="Input"
        )
        rainfallClass = arcpy.Parameter(
            displayName="Select field of rainfall classification",
            name="rainfallCalss",
            datatype="Field",
            parameterType="Required",
            direction="Input"
        )
        rainfallClass.parameterDependencies = [rainfall.name]
        rainfallClass.filter.list = ["Text"]
        rainfallDate = arcpy.Parameter(
            displayName="Select field of rainfall date",
            name="rainfallDate",
            datatype="Field",
            parameterType="Required",
            direction="Input"
        )
        rainfallDate.parameterDependencies = [rainfall.name]
        rainfallDate.filter.list = ["Text"]
        # manMadeSlop
        manMadeSlop = arcpy.Parameter(
            displayName="Man-made slope Layer (Year)",
            name="manMadeSlopLayer",
            datatype="GPFeatureLayer",
            parameterType="Required",
            direction="Input",
            multiValue=True
        )
        # Slope
        slope = arcpy.Parameter(
            displayName="Slope Layer (Year)",
            name="slopeLayer",
            datatype="GPRasterLayer",
            parameterType="Required",
            direction="Input",
            multiValue=True
        )
        # NDVI
        ndvi = arcpy.Parameter(
            displayName="NDVI Layer (Year)",
            name="ndviLayer",
            datatype="GPRasterLayer",
            parameterType="Required",
            direction="Input",
            multiValue=True
        )
        # Land use
        landUse = arcpy.Parameter(
            displayName="Land use Layer (Year)",
            name="landUseLayer",
            datatype="GPRasterLayer",
            parameterType="Required",
            direction="Input",
            multiValue=True
        )
        savePath = arcpy.Parameter(
            displayName="Save path",
            name="savePath",
            datatype="DEDiskConnection", #DEFolder
            parameterType="Required",
            direction="Output"
        )

        return [fishnet, landslide, landslideType, landslideDate, rainfall, rainfallClass, rainfallDate, manMadeSlop, slope, ndvi, landUse, savePath]

    def isLicensed(self):
        """Set whether the tool is licensed to execute."""
        return True

    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal
        validation is performed.  This method is called whenever a parameter
        has been changed."""
        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter. This method is called after internal validation."""
        return

    def execute(self, parameters, messages):
        """The source code of the tool."""
        fishnet = parameters[0].valueAsText
        landslide = parameters[1].valueAsText
        landslideType = parameters[2].valueAsText
        landslideDate = parameters[3].valueAsText
        rainfall = parameters[4].valueAsText
        rainfallClass = parameters[5].valueAsText
        rainfallDate = parameters[6].valueAsText
        manMade = parameters[7].valueAsText.split(";") # Man made slope
        slope = parameters[8].valueAsText.split(";")
        ndvi = parameters[9].valueAsText.split(";")
        landUse = parameters[10].valueAsText.split(";")
        savePath = parameters[11].valueAsText
        arcpy.env.extent = fishnet

        # Spatial Join for data join
        def spatialJoin(joinType: str, mainFeature: str, attachFeature: str, attachField: str, path: str, target: str) -> None:
            # Get fieldmappings
            fieldmappings = arcpy.FieldMappings()
            fieldmappings.addTable(mainFeature)
            addfiledmap = arcpy.FieldMappings()
            addfiledmap.addTable(attachFeature)
            classificationIndex = addfiledmap.findFieldMapIndex(attachField)
            classification = addfiledmap.getFieldMap(classificationIndex)
            fieldmappings.addFieldMap(classification)

            # Spatial join
            arcpy.analysis.SpatialJoin(mainFeature, attachFeature, path, "JOIN_ONE_TO_ONE", "KEEP_ALL", fieldmappings, joinType)
            arcpy.management.Delete(mainFeature)
            if attachFeature == landslide:
                expression = arcpy.AddFieldDelimiters(path, attachField)+" IS NOT NULL"
                arcpy.management.SelectLayerByAttribute(path, "NEW_SELECTION", expression)
            arcpy.management.CalculateField(path, target, "!{}!".format(attachField), "PYTHON3")
            arcpy.management.SelectLayerByAttribute(path, "CLEAR_SELECTION")
            arcpy.management.DeleteField(path, ["Join_Count", "TARGET_FID", attachField])

            return
        
        # Raster Join
        self.existPoints = []
        def rasterJoin(mainLayer: str, rasterLayer: str, path: str, target: str, year: str) -> None:
            ## Raster to points
            pathRaster = os.path.join("memory", target + year) # Save intermediate result in memory
            ### Check whether the raster points existing
            if pathRaster not in self.existPoints:
                arcpy.AddMessage("Using {} to generate points, saved in {}.".format(rasterLayer, pathRaster))
                self.existPoints.append(pathRaster) # Save the processed points to save calculation time
                arcpy.conversion.RasterToPoint(rasterLayer, pathRaster, "VALUE")
            else:
                arcpy.AddMessage("Using exisisting {} points data.".format(pathRaster))
            ## Add data into fishnet
            spatialJoin("CONTAINS", mainLayer, pathRaster, "grid_code", path, target)

            return
        
        # Creat Thiessen
        def creatThiessen(rainfall: str, rainfallDate: str, rainfallClass: str, day: str, path: str) -> None:
            expression = arcpy.AddFieldDelimiters(rainfall, rainfallDate)+"=\'" + day +'\''
            arcpy.management.SelectLayerByAttribute(rainfall, "NEW_SELECTION", expression)
            arcpy.analysis.CreateThiessenPolygons(rainfall, path, "ALL")
            arcpy.management.DeleteField(path, [rainfallClass], "KEEP_FIELDS")

            return

        # Find the right dataset
        def findRight(datasets: list[str], year: str) -> str:
            if len(datasets) == 1:
                return datasets[0].replace("'","")
            for dataset in datasets:
                if year in dataset:
                    return dataset.replace("'","") # Delete two sides ''
            arcpy.AddMessage("No suitable dataset in {}, cannot process!".format(datasets))
            exit()
            return

        # Day when landslides happend
        days = set()
        with arcpy.da.SearchCursor(landslide, [landslideDate]) as cursor:
            for row in cursor:
                days.add(row[0])
        totalDays = len(days)

        num = 1
        result = []
        existingThiessen = []

        for day in days:
            arcpy.AddMessage("Processing {} ({}/{})".format(day, num, totalDays))
            memoryName = randomName()

            # Copy a blank fish net
            copyPath = os.path.join("memory", "Copy" + memoryName) # Save intermediate result in memory
            copyData = arcpy.management.CopyFeatures(fishnet, copyPath)

            # Add land slide data
            arcpy.AddMessage("Adding landslide data...")
            ## Select day
            expression = arcpy.AddFieldDelimiters(landslide, landslideDate)+"=\'" + day +'\''
            arcpy.management.SelectLayerByAttribute(landslide, "NEW_SELECTION", expression)
            ## Add data into fishnet
            pathSlide = os.path.join("memory", "Slide" + memoryName) # Save intermediate result in memory
            spatialJoin("CONTAINS", copyData, landslide, landslideType, pathSlide, "landslide")
            arcpy.management.CalculateField(pathSlide, "date", day, "PYTHON3")

            # Add rainfall data
            arcpy.AddMessage("Adding rainfall data...")
            ## Creat Thiessen
            pathThiessen = os.path.join("Thiessen" + day) # Thiessen do not support memory process
            # Checking existing Thiessen
            if pathThiessen not in existingThiessen:
                arcpy.AddMessage("Creating new rainfall thiessen {}".format(pathThiessen))
                existingThiessen.append(pathThiessen)
                creatThiessen(rainfall, rainfallDate, rainfallClass, day, pathThiessen)
            ## Add data into fishnet
            pathRain = os.path.join("memory", "Rain" + memoryName) # Save intermediate result in memory
            spatialJoin("INTERSECT", pathSlide, pathThiessen, rainfallClass, pathRain, "rainfall")
            
            # Add rainfall data day before
            arcpy.AddMessage("Adding rainfall data from previous day...")
            ## Calculate day
            date = datetime.strptime(day, "%Y%m%d")
            previousDay = date - timedelta(days=1)
            dayBefore = previousDay.strftime("%Y%m%d")
            ## Creat Thiessen
            pathThiessen = os.path.join("Thiessen" + dayBefore) # Thiessen do not support memory process
            ### Checking existing Thiessen
            if pathThiessen not in existingThiessen:
                arcpy.AddMessage("Creating new rainfall thiessen {}".format(pathThiessen))
                existingThiessen.append(pathThiessen)
                creatThiessen(rainfall, rainfallDate, rainfallClass, dayBefore, pathThiessen)
            ## Spatial Join
            pathRainB = os.path.join("memory", "RainB" + memoryName) # Save intermediate result in memory
            spatialJoin("INTERSECT", pathRain, pathThiessen, rainfallClass, pathRainB, "rainfall_Daybefore")

            # Add man-made slope
            arcpy.AddMessage("Adding man-made slope data...")
            ## Find the dataset in right day
            year = day[0:4]
            manMadeLayer = findRight(manMade, year)
            arcpy.AddMessage("Using {}...".format(manMadeLayer))
            # manMadeLayer = manMadeLayer.split("\\")[-1]
            ## Add fieldmapping
            fieldmappings = arcpy.FieldMappings()
            fieldmappings.addTable(pathRainB)
            ## Spatial Join
            pathManMade = os.path.join("memory", "ManMade" + memoryName) # Save intermediate result in memory
            arcpy.analysis.SpatialJoin(pathRainB, manMadeLayer, pathManMade, "JOIN_ONE_TO_ONE", "KEEP_ALL", fieldmappings, "INTERSECT")
            codeblock = CODEBLOCK_1
            arcpy.management.CalculateField(pathManMade, "manMadeSlop", "calculate(!Join_Count!)", "PYTHON3", codeblock)
            arcpy.management.DeleteField(pathManMade, ["Join_Count", "TARGET_FID"])
            arcpy.management.Delete(pathRainB)

            # Add slope
            arcpy.AddMessage("Adding slope data...")
            ## Find the dataset in right day
            slopeLayer = findRight(slope, year)
            pathSlope = os.path.join("memory", "Slope" + memoryName) # Save intermediate result in memory
            rasterJoin(pathManMade, slopeLayer, pathSlope, "slope", year)

            # Add NDVI
            arcpy.AddMessage("Adding vegation data...")
            ## Find th dataset in right day
            ndviLayer = findRight(ndvi, year)
            pathNDVI = os.path.join("memory", "NDVI" + memoryName) # Save intermediate result in memory
            rasterJoin(pathSlope, ndviLayer, pathNDVI, "vegetation", year)

            # Add land use
            arcpy.AddMessage("Adding land use data...")
            ## Find the dataset in right day
            landUseLayer = findRight(landUse, year)
            pathLand = os.path.join("memory", randomName("r", 10)) # Save intermediate result in memory
            rasterJoin(pathNDVI, landUseLayer, pathLand, "landUse", year)
            
            result.append(pathLand)
            num += 1
        
        # Merge all layers
        arcpy.AddMessage("Saving Results...")
        arcpy.management.Merge(result, savePath)

        # Delete Null cell (edge or sea)
        arcpy.management.MakeFeatureLayer(savePath, "tempTableView")
        expression = arcpy.AddFieldDelimiters("tempTableView", "vegetation") + " IS NULL OR "+ \
            arcpy.AddFieldDelimiters("tempTableView", "landUse") + " IS NULL"
        arcpy.management.SelectLayerByAttribute("tempTableView", "NEW_SELECTION", expression)
        if int(arcpy.management.GetCount("tempTableView")[0]) > 0:
            arcpy.management.DeleteFeatures("tempTableView")
        # Change Null in landslide and rain fall into 0
        for i in ["landslide", "rainfall", "rainfall_Daybefore"]:
            expression = arcpy.AddFieldDelimiters("tempTableView", i)+" IS NULL"
            arcpy.management.SelectLayerByAttribute("tempTableView", "NEW_SELECTION", expression)
            arcpy.management.CalculateField("tempTableView", i, '0', "PYTHON3")
        arcpy.management.SelectLayerByAttribute("tempTableView", "CLEAR_SELECTION")
        arcpy.management.Delete(self.existPoints)
        arcpy.management.Delete(existingThiessen)

        return

    def postExecute(self, parameters):
        """This method takes place after outputs are processed and
        added to the display."""
        return