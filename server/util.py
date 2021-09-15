import json  
import pickle
import numpy as np


__locations = None 
__data_columns = None
__model = None


def estimated_price(location,sqft,bath,bhk):
    try:
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index = -1

    x = np.zeros(len(__data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk 
    if loc_index >= 0:
        x[loc_index] = 1
    
    return round(__model.predict([x])[0] ,2)    

def get_location_names():
    return __locations


def load_saved_artifacts():
    print('Loading Artifacts........')
    global __data_columns
    global __locations 
    global __model 
    
    with open('./artifacts/columns.json', 'r') as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[3:]
        
    with open('./artifacts/Real_Estate_Price_Model.pickle', 'rb') as f:
        __model = pickle.load(f)
    print('Completed')

if __name__ == "__main__":
    load_saved_artifacts()
    print(get_location_names())
    print(estimated_price('Kalhalli', 1000, 2,2))
    print(estimated_price('Ejipura', 1000, 2,2))
    print(estimated_price('Indira Nagar', 1000, 2,2))
    
    # app.run(debug=True)