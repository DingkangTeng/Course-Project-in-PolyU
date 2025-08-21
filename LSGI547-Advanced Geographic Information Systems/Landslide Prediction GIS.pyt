# -*- coding: utf-8 -*-

import arcpy

from Tools.creatFishnet import creatFishnet as CF
from Tools.randomForest import randomForest as RF

# # For debug
# import os
# import numpy as np
# from datetime import datetime, timedelta

# from Function.general import randomName

# CODEBLOCK_1 = """
# def calculate(a):
#  if a == 0:
#   return 0
#  else:
#    return 1
# """

class Toolbox:
    def __init__(self):
        """Define the toolbox (the name of the toolbox is the name of the
        .pyt file)."""
        self.label = "Landslide Prediction GIS"
        self.alias = "Landslide Prediction GIS"

        # List of tool classes associated with this toolbox
        self.tools = [CF, RF]