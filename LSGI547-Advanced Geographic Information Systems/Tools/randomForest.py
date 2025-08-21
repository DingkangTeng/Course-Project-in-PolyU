import arcpy, sys, os
import numpy as np

sys.path.append(".") # Set path to the roots
from Function.general import randomName

class randomForest:
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Landslides predict"
        self.description = "This tool are used to trainning models and predict landslides."

    def getParameterInfo(self):
        """Define the tool parameters."""
        trainNet = arcpy.Parameter(
            displayName="Training fishnet Layer",
            name="trainNet",
            datatype="GPFeatureLayer",
            parameterType="Required",
            direction="Input"
        )
        builtUp = arcpy.Parameter(
            displayName="Urban built-up area Layer",
            name="builtUp",
            datatype="GPFeatureLayer",
            parameterType="Required",
            direction="Input",
        )
        predictNet = arcpy.Parameter(
            displayName="Prediction fishnet Layer",
            name="predictNet",
            datatype="GPFeatureLayer",
            parameterType="Required",
            direction="Input",
        )
        predictDate = arcpy.Parameter(
            displayName="Prediction date period (YYYYMMDD)",
            name="predictDate",
            datatype="GPString",
            parameterType="Required",
            direction="Input",
        )
        predictDate.value = "2023"
        savePath = arcpy.Parameter(
            displayName="Save path",
            name="savePath",
            datatype="DEFolder",
            parameterType="Required",
            direction="Input"
        )

        return [trainNet, builtUp, predictNet, predictDate, savePath]

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
        trainNet = parameters[0].valueAsText
        bulitUp = parameters[1].valueAsText
        predictNet = parameters[2].valueAsText
        predictDate = parameters[3].valueAsText
        savePath = parameters[4].valueAsText

        # Copy to memory
        ## Copy training data
        arcpy.AddMessage("Initialize analysis, copy data into memory...")
        memoryName = randomName()
        trainPath = os.path.join("memory", "train" + memoryName)
        arcpy.management.CopyFeatures(trainNet, trainPath)
        ## Down sampling
        arcpy.management.MakeFeatureLayer(trainPath, "tempTableView")
        expression = arcpy.AddFieldDelimiters("tempTableView", "date") + " NOT LIKE \'"+ predictDate + "%\' AND " + \
            arcpy.AddFieldDelimiters("tempTableView", "landslide") + " = \'0\'"
        arcpy.management.SelectLayerByAttribute("tempTableView", "NEW_SELECTION", expression)
        if int(arcpy.management.GetCount("tempTableView")[0]) > 0:
            arcpy.management.DeleteFeatures("tempTableView")
        arcpy.management.SelectLayerByAttribute("tempTableView", "CLEAR_SELECTION")
        ## Copu predict data
        arcpy.AddMessage("Calculating x and y coordinate...")
        arcpy.management.CalculateGeometryAttributes(trainPath, [['x', "CENTROID_X"], ['y', "CENTROID_Y"]])
        arcpy.AddMessage("Initialize predict feature, copy data into memory...")
        predictPath = os.path.join("memory", "predict" + memoryName)
        arcpy.management.CopyFeatures(predictNet, predictPath)
        arcpy.AddMessage("Calculating x and y coordinate...")
        arcpy.management.CalculateGeometryAttributes(predictNet, [['x', "CENTROID_X"], ['y', "CENTROID_Y"]])

        # Predict and save results
        arcpy.AddMessage("Start model building and predicting...")
        predictedFeaturePath = os.path.join(savePath, "predict")
        arcpy.stats.Forest(
            prediction_type="PREDICT_FEATURES",
            in_features=trainPath,
            variable_predict="landslide",
            treat_variable_as_categorical="CATEGORICAL",
            explanatory_variables=[
                ["rainfall", "true"],
                ["rainfall_Daybefore", "true"],
                ["manMadeSlop", "true"],
                ["slope", "true"],
                ["vegetation", "true"],
                ["landUse", "true"],
                ['x', "false"],
                ['y', "false"]
            ],
           features_to_predict=predictPath,
           output_features=predictedFeaturePath,
           explanatory_variable_matching=[
                ["rainfall", "rainfall"],
                ["rainfall_Daybefore", "rainfall_Daybefore"],
                ["manMadeSlop", "manMadeSlop"],
                ["slope", "slope"],
                ["vegetation", "vegetation"],
                ["landUse", "landUse"],
                ['x', 'x'],
                ['y', 'y']
            ],
            output_importance_table=os.path.join(savePath, "importance"),
            number_of_trees=128,
            percentage_for_training=10,
            output_classification_table=os.path.join(savePath, "classification"),
            output_validation_table=os.path.join(savePath, "validationTable"),
            compensate_sparse_categories=True,
            number_validation_runs=5,
            calculate_uncertainty=True,
            output_trained_model=os.path.join(savePath, "model")
        )

        arcpy.management.Delete([trainPath])

        # Change built-up area into non-landslide
        arcpy.management.MakeFeatureLayer(predictedFeaturePath, "tempTableView")
        arcpy.management.SelectLayerByLocation("tempTableView", "INTERSECT", bulitUp)
        arcpy.management.CalculateField("tempTableView", "PREDICTED", '0', "PYTHON3")
        arcpy.management.SelectLayerByAttribute("tempTableView", "CLEAR_SELECTION")
        
        # Annotation in real prediction work
        self.valid(predictedFeaturePath, predictPath)

        arcpy.management.Delete([predictPath])

        return

    def postExecute(self, parameters):
        """This method takes place after outputs are processed and
        added to the display."""
        return
    
    # Valid the predicting results
    def valid(self, predictedFeaturePath, predictPath):
        prediction = arcpy.da.TableToNumPyArray(predictedFeaturePath, ["PREDICTED"]).tolist()
        trueResult = arcpy.da.TableToNumPyArray(predictPath, ["landslide"]).tolist()
        right = 0
        wrong1 = 0 # Predicted happened, but not happened
        wrong2 = 0 # Predicted happened, but wrong type
        wrong3 = 0 # Predicted not happened, but happened in reality (Fatal wrong prediction)
        for i in range(len(prediction)):
            if prediction[i] == trueResult[i]:
                right += 1
            elif prediction[i] != '0' and trueResult[i] == '0':
                wrong1 += 1
            elif prediction[i] != '0' and prediction[i] != trueResult[i]:
                wrong2 += 1
            elif prediction[i] == '0' and trueResult[i] != '0':
                wrong3 += 1
                
        total = right + wrong1 + wrong2 + wrong3
        realSlide = total - np.sum(trueResult == '0')
        arcpy.AddMessage("Landslide incidents times in reality: {}".format(realSlide))
        arcpy.AddMessage(
            "Correct prediction: {}/{},\n \
            Wrong Prediction: \n \
                False alarm: {}/{},\n \
                Wrong prodection type: {}/{},\n \
                Not predicted (fatal): {}/{}"
            .format(
                right, total,
                wrong1, total,
                wrong2, realSlide,
                wrong3, realSlide
            )
        )

        return