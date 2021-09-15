import pandas as pd 
import numpy as np   
from matplotlib import pyplot as plt
# %matplotlib inline 
import matplotlib
from sklearn.model_selection import train_test_split, ShuffleSplit, cross_val_score
from sklearn.linear_model import LinearRegression

matplotlib.rcParams['figure.figsize'] = (20,10)


def is_float(x):
    try:
        float(x)
    except:
        return False
    return True     

def convert_sqft_to_num(x):
    tokens = x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0]) + float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None 


df1 = pd.read_csv('Bengaluru_House_Data.csv')

df1 = df1.drop(['area_type', 'society', 'balcony', 'availability'], axis= 'columns')
df1 = df1.dropna()
df1['bhk'] = df1['size'].apply(lambda x: int(x.split(' ')[0]))
df1['total_sqft'] = df1['total_sqft'].apply(convert_sqft_to_num)

df1['price_per_sqft'] = df1['price']*100000 / df1['total_sqft']

df1.location = df1.location.apply(lambda x: x.strip())
location_stats = df1.groupby('location')['location'].agg('count').sort_values(ascending=False)
location_stats_less_than_10 = location_stats[location_stats<=10]
df1.location = df1.location.apply(lambda x: 'other' if x in location_stats_less_than_10 else x)

df1 = df1[~(df1.total_sqft/ df1.bhk<300)]

def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft>(m-st)) & (subdf.price_per_sqft<=(m+st))]
        df_out = pd.concat([df_out, reduced_df], ignore_index=True)
    return df_out

df1 = remove_pps_outliers(df1)



def plot_scatter_chart(df, location):
    bhk2 = df[(df.location==location) & (df.bhk==2) ]
    bhk3 = df[(df.location==location) & (df.bhk==3) ]
    matplotlib.rcParams['figure.figsize'] = (15,10)
    plt.scatter(bhk2.total_sqft,bhk2.price_per_sqft,color='blue',label='2 BHK',s=50)
    plt.scatter(bhk3.total_sqft,bhk3.price_per_sqft,color='green',marker='+', label='3 BHK',s=50)
    plt.xlabel('Total Square Feet Area')
    plt.ylabel('Price Per Square Feet')
    plt.title(location)
    plt.legend()
    plt.show()


def remove_bhk_outlier(df):
    exclude_indices = np.array([])
    for location , location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std' : np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices, axis='index')
        
        
df1 = remove_bhk_outlier(df1)

# matplotlib.rcParams['figure.figsize'] = (10,6)
# plt.hist(df1.bath, rwidth=0.8)
# plt.xlabel('Price Per Square Feet')
# plt.ylabel('Count')
# plt.show()

df1 = df1[df1.bath<df1.bhk+2]
df1 = df1.drop(['size', 'price_per_sqft'], axis='columns')


dummies = pd.get_dummies(df1.location)
df1 = pd.concat([df1, dummies.drop('other', axis='columns')], axis='columns')
df1 = df1.drop('location', axis=1)




X = df1.drop('price', axis=1)
y = df1['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

clf = LinearRegression()
clf.fit(X_train, y_train)
# print(clf.score(X_test, y_test))

# cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
# print(cross_val_score(LinearRegression(), X, y, cv=cv))

def predict_price(location,sqft,bath,bhk):
    loc_index = np.where(X.columns==location)[0][0]
    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk 
    if loc_index >= 0:
        x[loc_index] = 1
    return clf.predict([x])[0]


# print(predict_price('Indira Nagar', 3000, 4, 4))

import pickle
with open('Real_Estate_Price_Model.pickle', 'wb') as f:
    pickle.dump(clf, f)
    

import json 
columns = {
    'data_columns': [col.lower() for col in X.columns]
}
with open('columns.json', 'w') as f:
    f.write(json.dumps(columns))