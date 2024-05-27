# ML/AI Career Path Final Project
# Predicting Song Popularity on Spotify using Machine Learning and AI

#import libraries, functions and objects

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MultiLabelBinarizer, FunctionTransformer
from category_encoders import BinaryEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from scipy.stats import uniform, randint

#inspect dataset, determine target variable

songs = pd.read_csv('spotify_top_songs_data.csv')
#print(songs.head())
#print(songs['popularity'].describe())
#print(songs['popularity'].median())

X = songs.drop(columns=['popularity', 'artist', 'song'])
y = songs['popularity']

#print(songs.info())

# Observed zero null values
# removed artist and song columns, as they would have too much variation to be relevant--any relevant titles or artists would be very successful outliers
# Examine data relationship with plots -- do the features have a linear (1-to-1) relationship?
# sort columns into numerical and categorical features

# print(X.dtypes)

num_cols = []
cat_cols = []
cat_cols_no_genre = []

for col in X.columns:
    if X[col].dtype == 'float64':
        num_cols.append(col)
    elif col == 'year' or col == 'duration_ms':
        num_cols.append(col)
    else:
        cat_cols.append(col)
        cat_cols_no_genre.append(col)

print(num_cols)
cat_cols_no_genre.pop()
print(cat_cols_no_genre)

# Numerical features are all scaled 0-1 except duration, year, and tempo
# Need a scaler for year, duration and tempo to measure relative values' effects

# Encode categorical variables? Check value counts
#for c in cat_cols:
#    print(X[c].value_counts())

# rename key labels for clarity according to dataset description
key_dict = {0: 'C',
            1: 'C#/Db',
            2: 'D',
            3: 'D#/Eb',
            4: 'E',
            5: 'F',
            6: 'F#/Gb',
            7: 'G',
            8: 'G#/Ab',
            9: 'A',
            10: 'A#/Bb',
            11: 'B',
            -1: 'none'}

X['key'] = X['key'].map(key_dict)

# Label-encode, one-hot-encode, or binary-encode the categorical variables?
# OHE categorical variables, multi-label binarize the genre column

# PROPERLY PREPARE GENRE COLUMN FOR MULTI-LABEL BINARIZER--STANDARDIZE GENRE LABELS

# function to manually edit each entry in Genre column as a string
'''
def genre_replacer_string(str):
    # space between words of the same genre
    str.replace('hip hop', 'hip-hop')
    str.replace('easy listening', 'easy-listening')
    #comma removal
    str.replace(',', '')
    # split string
    split_string = str.split()
    return split_string

# rewrite function to apply to lists

def genre_replacer_list(list):
    # space between words of the same genre
    for item in list:
        if type(item) == 'str':
            item.replace(',', '')
    return list
'''
#helper function to strip whitespace on edges

def strip_whitespace(list):
    for str in list:
        str.strip()
    return list

genre_dtypes = X['genre'].apply(type)
unique_genre_dtypes = genre_dtypes.unique()
print(unique_genre_dtypes)
genre_split = X['genre'].apply(lambda x: x.split(','))
genre_split_stripped = X['genre'].apply(strip_whitespace)
print(genre_split_stripped.head(10))
new_genre_dtypes = genre_split_stripped.apply(type)
new_unique_genre_dtypes = genre_dtypes.unique()
print(new_unique_genre_dtypes)

# Encode/scale data
# get genre labels from dataset

mlb = MultiLabelBinarizer()

#genre_x = pd.DataFrame(mlb.fit_transform(genre_split), columns=mlb.classes_)
#cat_x = pd.DataFrame(ohe.fit_transform(X[cat_cols.remove('genre')]))
#num_x = pd.DataFrame(scaler.fit_transform(X[num_cols]))

#print(genre_x.head())

X['genre'] = genre_split_stripped

#print(X['genre'].head())

#Perform train test split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


#print(X_train['genre'].head())


#Multilabel binarize the 'genre' column, and O.H.E. the other categorical columns

# Use FunctionTransformer as a helper for the pipeline

# Fit the training data to the MLB before fitting the pipeline to ensure equal number of columns
mlb.fit(X_train['genre'])

def fit_transform_mlb(data):
    return mlb.fit_transform(data)

def transform_mlb(data):
    return mlb.transform(data)

#category_categories = [X_train[cat].unique() for cat in cat_cols_no_genre]

mlb_transformer = FunctionTransformer(transform_mlb, validate=False)
#mlb_transform_transformer = FunctionTransformer(transform_mlb)

genre_vals = Pipeline([('mlb', mlb_transformer)])
#genre_vals_transform = Pipeline([('mlb_transform', mlb_transform_transformer)])
other_cat_vals = Pipeline([('ohe', OneHotEncoder(drop='first', sparse_output=False))])

print('MLB Train Output Shape:', genre_vals.fit_transform(X_train['genre']).shape)
print('MLB Test Output Shape:', genre_vals.transform(X_test['genre']).shape)
#genre_test_transform = genre_vals.transform(X_test['genre'])
#print(genre_test_transform.shape)

print('OHE Train Output Shape:', other_cat_vals.fit_transform(X_train[cat_cols_no_genre]).shape)
print('OHE Test Output Shape:', other_cat_vals.transform(X_test[cat_cols_no_genre]).shape)
#x_test_transform = other_cat_vals.transform(X_test[cat_cols_no_genre])
#print(x_test_transform.shape)

num_vals = Pipeline([('scale', StandardScaler())])
print('Scaler Train Output Shape:', num_vals.fit_transform(X_train[num_cols]).shape)
print('Scaler Test Output Shape:', num_vals.transform(X_test[num_cols]).shape)

preprocess = ColumnTransformer(transformers=[('genre_preprocess', genre_vals, 'genre'), ('other_cat_preprocess', other_cat_vals, cat_cols_no_genre), ('num_preprocess', num_vals, num_cols)])
#print('Preprocess Output Shape:', preprocess.fit_transform(X_train).shape)
#x_test_transform = preprocess.transform(X_test)
#print(x_test_transform)

# Define the ColumnTransformer for testing (only transform, no fit)
'''
preprocess_transform = ColumnTransformer(
    transformers=[
        ('genre_preprocess', genre_vals_transform, 'genre'),
        ('other_cat_preprocess', other_cat_vals, cat_cols_no_genre),
        ('num_preprocess', num_vals, num_cols)
    ]
)

X_train_transformed = preprocess.fit_transform(X_train)
X_test_transformed = preprocess_transform.transform(X_test)

# Check shapes
print('Training data shape after preprocessing:', X_train_transformed.shape)
print('Test data shape after preprocessing:', X_test_transformed.shape)
'''

# ERROR ENCOUNTERED: pipeline recognizing multi-genre labels as unique genres; try specifying the "Categories" parameter in the OneHotEncoder
# ERROR RESOLVED: used a multi-label-binarizer wrapped in a Function Transformer, modified the data appropriately by changing the genre strings into lists of genres

# ERROR ENCOUNTERED: column transformer had an issue concatenating the data during a transformation, recognizing the array at index 0 as having 1 row and the array at index 1 as having 1600 rows
# ERROR RESOLVED: the "genre" column label in the genre preprocess tuple was entered as: ['genre'], in a list. it should have been entered as 'genre', as a string.

#Build and test full pipelines

# Linear Regression
lin_reg_pipe = Pipeline([('preprocess', preprocess), ('lin_regr', LinearRegression())])
lin_reg_pipe.fit(X_train, y_train)

# Ensure the pipeline transforms the test data using the same fitted transformations
#X_train_transformed = lin_reg_pipe.named_steps['preprocess'].transform(X_train)
#X_test_transformed = lin_reg_pipe.named_steps['preprocess'].transform(X_test)

# Check the shapes of the transformed training and test data
#print('Transformed Training Data Shape:', X_train_transformed.shape)
#print('Transformed Test Data Shape:', X_test_transformed.shape)

train_score = lin_reg_pipe.score(X_train, y_train)
print('Linear Regression Train Score:', train_score)
test_score = lin_reg_pipe.score(X_test, y_test)
print('Linear Regression Test Score:', test_score)

# ERROR ENCOUNTERED: number of features in test data not matching expected amount as learned from the training data
# ERROR RESOLVED: fit the MultiLabelBinarizer on the training data separately, and included only the transform function in the pipeline to ensure consistent column numbers 

# K-Neighbors Regression
k_neighbors_pipe = Pipeline([('preprocess', preprocess), ('k_neighbors_regr', KNeighborsRegressor())])
k_neighbors_pipe.fit(X_train, y_train)
k_neighbors_train_score = k_neighbors_pipe.score(X_train, y_train)
k_neighbors_test_score = k_neighbors_pipe.score(X_test, y_test)
print('K-Neighbors Regression Train Score:', k_neighbors_train_score)
print('K-Neighbors Regression Test Score:', k_neighbors_test_score)

# Decision Tree Regression
dt_pipe = Pipeline([('preprocess', preprocess), ('dt_regr', DecisionTreeRegressor())])
dt_pipe.fit(X_train, y_train)
dt_train_score = dt_pipe.score(X_train, y_train)
dt_test_score = dt_pipe.score(X_test, y_test)
print('Decision Tree Regression Train Score:', dt_train_score)
print('Decision Tree Regression Test Score:', dt_test_score)

# Random Forest Regression
rfr_pipe = Pipeline([('preprocess', preprocess), ('rf_regr', RandomForestRegressor())])
rfr_pipe.fit(X_train, y_train)
rfr_train_score = rfr_pipe.score(X_train, y_train)
rfr_test_score = rfr_pipe.score(X_test, y_test)
print('Random Forest Regression Train Score:', rfr_train_score)
print('Random Forest Regression Test Score:', rfr_test_score)

# HYPERPARAMETER TUNING USING RANDOMIZED SEARCH CV

# Create single pipeline with the different ML models to prepare the Randomized Search CV? Or perform different RSearchCV on different pipelines?

# Test multiple different RandomSearchCVs on different pipelines
'''
lin_reg_params_rs = [{'lin_regr': [LinearRegression()], 'lin_regr__fit_intercept': [True, False]},
                   {'lin_regr': [Lasso()], 'lin_regr__alpha': uniform(loc=0, scale=100)},
                   {'lin_regr': [Ridge()], 'lin_regr__alpha': uniform(loc=0, scale=100)}]
k_neighbors_params_rs = {'k_neighbors_regr__n_neighbors': randint(low=1, high=10)}
dt_params_rs = {'dt_regr__max_depth': randint(low=1, high=10)}
rf_params_rs = {'rf_regr__max_depth': randint(low=1, high=10)}

# Linear Regression, Lasso and Ridge alternatives Randomized Search CV

rs_lin_reg = RandomizedSearchCV(estimator=lin_reg_pipe, param_distributions=lin_reg_params_rs, scoring='neg_mean_squared_error', cv=5)
rs_lin_reg.fit(X_train, y_train)
lin_reg_best_estimator = rs_lin_reg.best_estimator_.named_steps['lin_regr']
print('RSearchCV Linear Regression Best Estimator:', lin_reg_best_estimator)
print('RSearchCV Linear Regression Best Parameters:', lin_reg_best_estimator.get_params())


# K-Neighbors Regression Randomized Search CV

rs_kn_reg = RandomizedSearchCV(estimator=k_neighbors_pipe, param_distributions=k_neighbors_params_rs)
rs_kn_reg.fit(X_train, y_train)
kn_reg_best_estimator = rs_kn_reg.best_estimator_.named_steps['k_neighbors_regr']
print('RSearchCV K-Neighbors Regression Best Estimator:', kn_reg_best_estimator)
print('RSearchCV K-Neighbors Regression Best Parameters:', kn_reg_best_estimator.get_params())


# Decision Tree Regression Randomized Search CV
rs_dt_reg = RandomizedSearchCV(estimator=dt_pipe, param_distributions=dt_params_rs)
rs_dt_reg.fit(X_train, y_train)
dt_best_estimator = rs_dt_reg.best_estimator_.named_steps['dt_regr']
print('RSearchCV Decision Tree Regression Best Estimator:', dt_best_estimator)
print('RSearchCV Decision Tree Regression Best Parameters:', dt_best_estimator.get_params())


# Random Forest Regression Randomized Search CV
rs_rf_reg = RandomizedSearchCV(estimator=rfr_pipe, param_distributions=rf_params_rs)
rs_rf_reg.fit(X_train, y_train)
rf_best_estimator = rs_rf_reg.best_estimator_.named_steps['rf_regr']
print('RSearch CV Random Forest Regression Best Estimator:', rf_best_estimator)
print('RSearchCV Random Forest Regression Best Parameters:', rf_best_estimator.get_params())

# Randomized Search CV Train and Test Scores Together

print('RSearchCV Linear Regression Best Estimator Train Score:', rs_lin_reg.best_estimator_.score(X_train, y_train))
print('RSearchCV Linear Regression Best Estimator Test Score:', rs_lin_reg.best_estimator_.score(X_test, y_test))

print('RSearchCV K-Neighbors Regression Best Estimator Train Score:', rs_kn_reg.best_estimator_.score(X_train, y_train))
print('RSearchCV K-Neighbors Regression Best Estimator Test Score:', rs_kn_reg.best_estimator_.score(X_test, y_test))

print('RSearchCV Decision Tree Regression Best Estimator Train Score:', rs_dt_reg.best_estimator_.score(X_train, y_train))
print('RSearchCV Decision Tree Regression Best Estimator Test Score:', rs_dt_reg.best_estimator_.score(X_test, y_test))

print('RSearchCV Random Forest Regression Best Estimator Train Score:', rs_rf_reg.best_estimator_.score(X_train, y_train))
print('RSearchCV Random Forest Regression Best Estimator Test Score:', rs_rf_reg.best_estimator_.score(X_test, y_test))

# All scores are pretty terrible...try a different regressor? Support vector machine regression?
'''
svr_pipe = Pipeline([('preprocess', preprocess), ('svm_regr', SVR())])
svr_pipe.fit(X_train, y_train)
svr_train_score = svr_pipe.score(X_train, y_train)
svr_test_score = svr_pipe.score(X_test, y_test)
print('Support Vector Regression Train Score:', svr_train_score)
print('Support Vector Regression Test Score:', svr_test_score)

'''
# Hyperparameter turning with Randomized Search CV

svr_params_rs = {'svm_regr__kernel': ['linear', 'poly', 'rbf'],
              'svm_regr__gamma': uniform(loc=0, scale=5),
              'svm_regr__C': uniform(loc=0, scale=10)}

rs_svr = RandomizedSearchCV(estimator=svr_pipe, param_distributions=svr_params_rs)
rs_svr.fit(X_train, y_train)
svr_best_estimator = rs_svr.best_estimator_.named_steps['svm_regr']
print('RSearchCV Support Vector Regression Best Estimator:', svr_best_estimator)
print('RSearchCV Support Vector Regression Best Parameters:', svr_best_estimator.get_params())
print('RSearchCV Support Vector Regression Best Estimator Train Score:', rs_svr.best_estimator_.score(X_train, y_train))
print('RSearchCV Support Vector Regression Best Estimator Test Score:', rs_svr.best_estimator_.score(X_test, y_test))
'''
# Re-format data so that " classical" with extra space gets recognized properly by the multi-label binarizer

# Accuracies are still low...try grid search CV instead?

# HYPERPARAMETER TUNING USING GRID SEARCH

# Define parameter grids
'''
lin_reg_params_gs = [{'lin_regr': [LinearRegression()], 'lin_regr__fit_intercept': [True, False]},
                   {'lin_regr': [Lasso()], 'lin_regr__alpha': [0.01, 0.1, 1, 10]},
                   {'lin_regr': [Ridge()], 'lin_regr__alpha': [0.01, 0.1, 1, 10]}]
k_neighbors_params_gs = {'k_neighbors_regr__n_neighbors': list(range(1, 11))}
svr_params_gs = {'svm_regr__kernel': ['linear', 'poly', 'rbf'],
              'svm_regr__gamma': [0.01, 0.1, 1, 3, 5],
              'svm_regr__C': [0.01, 0.1, 1, 3, 5]}
dt_params_gs = {'dt_regr__max_depth': list(range(1, 11))}
rf_params_gs = {'rf_regr__max_depth': list(range(1, 10))}

# Linear Regression Grid Search

gs_lin_reg = GridSearchCV(estimator=lin_reg_pipe, param_grid=lin_reg_params_gs)
gs_lin_reg.fit(X_train, y_train)
print('GridSearch Linear Regression Best Estimator:', gs_lin_reg.best_estimator_.named_steps['lin_regr'])
print('GridSearch Linear Regression Best Parameters:', gs_lin_reg.best_estimator_.named_steps['lin_regr'].get_params())
print('GridSearch Linear Regression Best Score:', gs_lin_reg.best_score_)
print('GridSearch Linear Regression Train Score:', gs_lin_reg.score(X_train, y_train))
print('GridSearch Linear Regression Test Score:', gs_lin_reg.score(X_test, y_test))

# K-Neighbors Grid Search

gs_k_neighbors = GridSearchCV(estimator=k_neighbors_pipe, param_grid=k_neighbors_params_gs)
gs_k_neighbors.fit(X_train, y_train)
print('GridSearch K-Neighbors Regression Best Estimator:', gs_k_neighbors.best_estimator_.named_steps['k_neighbors_regr'])
print('GridSearch K-Neighbors Regression Best Parameters:', gs_k_neighbors.best_estimator_.named_steps['k_neighbors_regr'].get_params())
print('GridSearch K-Neighbors Regression Best Score:', gs_k_neighbors.best_score_)
print('GridSearch K-Neighbors Regression Train Score:', gs_k_neighbors.score(X_train, y_train))
print('GridSearch K-Neighbors Regression Test Score:', gs_k_neighbors.score(X_test, y_test))

# Support Vector Machine Grid Search

gs_svr = GridSearchCV(estimator=svr_pipe, param_grid=svr_params_gs)
gs_svr.fit(X_train, y_train)
print('GridSearch Support Vector Regression Best Estimator:', gs_svr.best_estimator_.named_steps['svm_regr'])
print('GridSearch Support Vector Regression Best Parameters:', gs_svr.best_estimator_.named_steps['svm_regr'].get_params())
print('GridSearch Support Vector Regression Best Score:', gs_svr.best_score_)
print('GridSearch Support Vector Regression Train Score:', gs_svr.score(X_train, y_train))
print('GridSearch Support Vector Regression Test Score:', gs_svr.score(X_test, y_test))

# Decision Tree Grid Search

gs_dt = GridSearchCV(estimator=dt_pipe, param_grid=dt_params_gs)
gs_dt.fit(X_train, y_train)
print('GridSearch Decision Tree Regression Best Estimator:', gs_dt.best_estimator_.named_steps['dt_regr'])
print('GridSearch Decision Tree Regression Best Parameters:', gs_dt.best_estimator_.named_steps['dt_regr'].get_params())
print('GridSearch Decision Tree Regression Best Score:', gs_dt.best_score_)
print('GridSearch Decision Tree Regression Train Score:', gs_dt.score(X_train, y_train))
print('GridSearch Decision Tree Regression Test Score:', gs_dt.score(X_test, y_test))

gs_rfr = GridSearchCV(estimator=rfr_pipe, param_grid=rf_params_gs)
gs_rfr.fit(X_train, y_train)
print('GridSearch Random Forest Regression Best Estimator:', gs_rfr.best_estimator_.named_steps['rf_regr'])
print('GridSearch Random Forest Regression Best Parameters:', gs_rfr.best_estimator_.named_steps['rf_regr'].get_params())
print('GridSearch Random Forest Regression Best Score:', gs_rfr.best_score_)
print('GridSearch Random Forest Regression Train Score:', gs_rfr.score(X_train, y_train))
print('GridSearch Random Forest Regression Test Score:', gs_rfr.score(X_test, y_test))
'''
# The accuracy is still abysmal
# Will reattempt by creating a new column in the data set to create a binary classification of "popular" or "not popular"
# "popular" represents songs of a popularity score above the median
# Use classification models instead of regression models