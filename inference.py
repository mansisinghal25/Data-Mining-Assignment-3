'''DO NOT DELETE ANY PART OF CODE
We will run only the evaluation function.

Do not put anything outside of the functions, it will take time in evaluation.
You will have to create another code file to run the necessary code.
'''
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import f1_score
from itertools import permutations

def preprocessing(test_set):
    column_to_cat = {
        "Elevation":["elevation_low", "elevation_medium", "elevation_high", "elevation_ultra"],
        "Aspect":["aspect_low", "aspect_medium", "aspect_high", "aspect_ultra"],
        "Slope":["slope_low", "slope_medium", "slope_high", "slope_ultra"],
        "Hillshade_9am":["hillshade_9am_min", "hillshade_9am_max"],
        "Hillshade_Noon":["hillnoon_min", "hillnoon_max"],
        "Horizontal_Distance_To_Fire_Points":["low", "mid", "high"]
    }
    test_set['Elevation'] = OrdinalEncoder(categories = [column_to_cat['Elevation']]).fit_transform(test_set[['Elevation']])
    test_set['Aspect'] = OrdinalEncoder(categories = [column_to_cat['Aspect']]).fit_transform(test_set[['Aspect']])
    test_set['Slope'] = OrdinalEncoder(categories = [column_to_cat['Slope']]).fit_transform(test_set[['Slope']])
    test_set['Hillshade_9am'] = OrdinalEncoder(categories = [column_to_cat['Hillshade_9am']]).fit_transform(test_set[['Hillshade_9am']])
    test_set['Hillshade_Noon'] = OrdinalEncoder(categories = [column_to_cat['Hillshade_Noon']]).fit_transform(test_set[['Hillshade_Noon']])
    test_set['Horizontal_Distance_To_Fire_Points'] = OrdinalEncoder(categories = [column_to_cat['Horizontal_Distance_To_Fire_Points']]).fit_transform(test_set[['Horizontal_Distance_To_Fire_Points']])

    #reducing dimentionality
    test_set['Distance_To_Hydrology'] = np.sqrt(np.square(test_set['Horizontal_Distance_To_Hydrology']) + np.square(test_set['Vertical_Distance_To_Hydrology']))
    test_set = test_set.drop(columns=['Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology'])

    return test_set

def mapping_true_label(test_labels, ground_labels):
    ground_labels = ground_labels-1
    # finding the best mapping
    perm = permutations([0, 1, 2, 3, 4, 5, 6])
    reference_labels = []
    score = 0
    for i in list(perm):
        relabel = np.choose(test_labels,i).astype(np.int64)
        new_score = f1_score(ground_labels, relabel, average='weighted')
        if new_score > score:
            reference_labels = i
            score = new_score
    # new_labels = np.choose(test_labels,reference_labels).astype(np.int64)
    return reference_labels

def predict(test_set) :
    # find and load your best model
    # Do all preprocessings inside this function only.
    # predict on the test set provided
    '''
    'test_set' is a csv path "test.csv", You need to read the csv and predict using your model.   
    '''
    df = pd.read_csv("covtype_train.csv")
    X = df.drop(columns=['target'])
    y = df['target'].copy()
    X = preprocessing(X)
    kmeans = KMeans(n_clusters=7).fit(X)
    y_pred = kmeans.predict(X)  
    reference_labels = mapping_true_label(y_pred, y)                             
    #Predicting the labels on the test set
    test_df = pd.read_csv(test_set)
    test_df = preprocessing(test_set)
    prediction = kmeans.predict(test_df) #Cluster labels for the test set 
    prediction = np.choose(prediction,reference_labels).astype(np.int64) + 1
    '''
    prediction is a 1D 'list' of output labels. just a single python list.
    '''
    return prediction.tolist()