"""
Applied Machine Learning Project_1: Using kNN

COPYRIGHT (C) 2020 John Fahringer (jrf5001@yahoo.com) and Naomi Burhoe ()
All rights reserved.

This program tries to classify a portion of our selected data set of
monarch butterflies using kNN.
"""

import pandas as pd
butterflies = pd.read_csv('data/Monarch_Butterfly_Habitat_Restoration__Polygon_Feature_Layer.csv')
butterflies.head()

X = butterflies[['']]
y = butterflies['']