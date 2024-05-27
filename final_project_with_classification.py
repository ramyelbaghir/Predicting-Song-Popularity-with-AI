# ML/AI Career Path Final Project
# Predicting Song Popularity on Spotify using Machine Learning and AI

#import libraries, functions and objects

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MultiLabelBinarizer, FunctionTransformer
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SequentialFeatureSelector as SFS
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from scipy.stats import uniform, randint

#inspect dataset, create classification label, determine target variable

songs = pd.read_csv('spotify_top_songs_data.csv')
print(songs.head())
print(songs['popularity'].median())
popularity_median = songs['popularity'].median()
songs['popular'] = songs['popularity'].apply(lambda score: 1 if score >= popularity_median else 0)
print(songs.head())

X = songs.drop(columns=['popularity', 'artist', 'song', 'popular'])
y = songs['popular']

# prepare data: Scale currently unscaled numerical data, one-hot encode categorical data, multi-label binarize genre column

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

# prepare genre column for Multi Label Binarizer

## helper function to strip whitespace on edges

def strip_whitespace(list):
    for str in list:
        str.strip()
    if str == ' classical':
        str.replace(' classical', 'classical')
    return list

genre_dtypes = X['genre'].apply(type)
unique_genre_dtypes = genre_dtypes.unique()
print(unique_genre_dtypes)
genre_split = X['genre'].apply(lambda x: x.split(','))
genre_split_stripped = genre_split.apply(strip_whitespace)
print(genre_split_stripped.head(10))
new_genre_dtypes = genre_split_stripped.apply(type)
new_unique_genre_dtypes = genre_dtypes.unique()
print(new_unique_genre_dtypes)

X['genre'] = genre_split_stripped

# initialize Multilabel Binarizer and Function Transformer, fit and transform the genre column through these objects

mlb = MultiLabelBinarizer()


# perform Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Fit the training data to the MLB before fitting the pipeline to ensure equal number of columns
mlb.fit(X_train['genre'])

def fit_transform_mlb(data):
    return mlb.fit_transform(data)

def transform_mlb(data):
    return mlb.transform(data)

# Wrap inside Function Transformer

mlb_transformer = FunctionTransformer(transform_mlb, validate=False, feature_names_out='one-to-one')

# create preprocessing pipelines and consolidate them with a column transformer

genre_vals = Pipeline([('mlb', mlb_transformer)])
other_cat_vals = Pipeline([('ohe', OneHotEncoder(drop='first', sparse_output=False))])
num_vals = Pipeline([('scale', StandardScaler())])

preprocess = ColumnTransformer(transformers=[
    ('genre_preprocess', genre_vals, 'genre'),
    ('other_cat_preprocess', other_cat_vals, cat_cols_no_genre),
    ('num_preprocess', num_vals, num_cols)])

# troubleshooting unknown category error
'''
# Fit the ColumnTransformer on the training data
preprocess.fit(X_train)

# Transform both training and test data
X_train_transformed = preprocess.transform(X_train)
X_test_transformed = preprocess.transform(X_test)

# Define and fit the classifier on the transformed training data
classifier = LogisticRegression()
classifier.fit(X_train_transformed, y_train)

# Evaluate the classifier on the transformed training and test data
train_score = classifier.score(X_train_transformed, y_train)
test_score = classifier.score(X_test_transformed, y_test)
print('Train Score:', train_score)
print('Test Score:', test_score)
'''
# TRYING FEATURE SELECTION TO IMPROVE SCORES
# Note: Sequential Feature Selection created an Attribute Error that was not easily resolved, trying Recursive Feature Elimination instead
# Same error occured

# Build and test full pipelines

# Logistic Regression Classifier

log_reg_pipe = Pipeline([('preprocess', preprocess), ('log_reg', LogisticRegression(max_iter=1000))])
#X_train_preprocessed = preprocess.fit_transform(X_train)
#X_test_preprocessed = preprocess.transform(X_test)
log_reg_pipe.fit(X_train, y_train)
print('Logistic Regression Train Score:', log_reg_pipe.score(X_train, y_train))
print('Logistic Regression Test Score:', log_reg_pipe.score(X_test, y_test))
'''
# Logicistic Regression with Feature Selection
rfe = RFE(estimator=log_reg_pipe, n_features_to_select=5)
rfe.fit(X_train_preprocessed, y_train)

X_train_rfe = rfe.transform(X_train_preprocessed)
X_test_rfe = rfe.transform(X_test_preprocessed)
# re-fit and re-score pipeline with feature-selected training data
log_reg_pipe.fit(X_train_rfe, y_train)
print('Logistic Regression w Feature Selection Train Score:', log_reg_pipe.score(X_train_rfe, y_train))
print('Logistic Regression w Feature Selection Test Score:', log_reg_pipe.score(X_test_rfe, y_test))
'''

# ERROR ENCOUNTERED: category in test set unknown to the model, despite fitting and transforming the data beforehand
# ERROR RESOLVED: forgot to drop the newly created "popular" column when creating X, which led to other errors

# K Nearest Neighbors Classifier

knn_pipe = Pipeline([('preprocess', preprocess), ('knn_clf', KNeighborsClassifier())])
knn_pipe.fit(X_train, y_train)
print('KNeighbors Classifier Train Score:', knn_pipe.score(X_train, y_train))
print('KNeighbors Classifier Test Score:', knn_pipe.score(X_test, y_test))

# Decision Tree Classifier

dt_pipe = Pipeline([('preprocess', preprocess), ('dt_clf', DecisionTreeClassifier())])
dt_pipe.fit(X_train, y_train)
print('Decision Tree Classifier Train Score:', dt_pipe.score(X_train, y_train))
print('Decision Tree Classifier Test Score:', dt_pipe.score(X_test, y_test))

# Support Vector Classifier

svc_pipe = Pipeline([('preprocess', preprocess), ('svm_clf', SVC())])
svc_pipe.fit(X_train, y_train)
print('Support Vector Machine Classifier Train Score:', svc_pipe.score(X_train, y_train))
print('Support Vector Machine Classifier Test Score:', svc_pipe.score(X_test, y_test))

'''
# Naive Bayes Classifier
#nb_pipe = Pipeline([('preprocess', preprocess), ('NB_clf', MultinomialNB())])
#nb_pipe.fit(X_train, y_train)
#print('Naive Bayes Classifier Train Score:', nb_pipe.score(X_train, y_train))
#print('Naive Bayes Classifier Test Score:', nb_pipe.score(X_test, y_test))
'''
# ERROR ENCOUNTERED: Naive Bayes Classifier cannot handle negative values; try converting negative values in 'loudness' column to positive only
# ERROR IGNORED: Deciding not to use Naive Bayes classifier, as it is commonly recommended for primarily text data--Will proceed by excluding it

# Random Forest Classifier

rf_pipe = Pipeline([('preprocess', preprocess), ('rf_clf', RandomForestClassifier(max_depth=9))])
rf_pipe.fit(X_train, y_train)
print('Random Forest Classifier Train Score:', rf_pipe.score(X_train, y_train))
print('Random Forest Classifier Test Score:', rf_pipe.score(X_test, y_test))

# TRYING ADAPTIVE BOOST CLASSIFIER AND GRADIENT BOOSTED CLASSIFIER

# AdaBoost Classifier

adaboost_pipe = Pipeline([('preprocess', preprocess), ('adaboost_clf', AdaBoostClassifier())])
adaboost_pipe.fit(X_train, y_train)
print('AdaBoost Classifier Train Score:', adaboost_pipe.score(X_train, y_train))
print('AdaBoost Classifier Test Score:', adaboost_pipe.score(X_test, y_test))

# Gradient Boost Classifier
gradboost_pipe = Pipeline([('preprocess', preprocess), ('gradboost_clf', GradientBoostingClassifier())])
gradboost_pipe.fit(X_train, y_train)
print('GradientBoost Classifier Train Score:', gradboost_pipe.score(X_train, y_train))
print('GradientBoost Classifier Test Score:', gradboost_pipe.score(X_test, y_test))


# HYPERPARAMETER TUNING USING GRID SEARCH
print('***********')
print('HYPERPARAMETER TUNING USING GRID SEARCH')
print('***********')

# Define parameter grids for each model

lr_params = {'log_reg__penalty': ['l1', 'l2'],
             'log_reg__C': [0.01, 0.1, 1, 10, 100]}
knn_params = {'knn_clf__n_neighbors': [1, 5, 10, 15, 20, 30, 40, 50, 75]}
dt_params = {'dt_clf__max_depth': [1, 3, 5, 7, 10, 15]}
svc_params = {'svm_clf__C': [0.01, 0.1, 1, 10, 100]}
rf_params = {'rf_clf__max_depth': [7, 9, 12, 13]}
adaboost_params = {'adaboost_clf__n_estimators': [1, 3, 5, 7, 10, 15]}
gradboost_params = {'gradboost_clf__n_estimators': [10, 15, 20, 25, 30, 35, 40, 50]}

# Logistic Regression Grid Search

gs_lr = GridSearchCV(estimator=log_reg_pipe, param_grid=lr_params)
gs_lr.fit(X_train, y_train)
gs_lr_best_estimator = gs_lr.best_estimator_.named_steps['log_reg']
print('GS--Logistic Regression Best Estimator:', gs_lr_best_estimator)
print('GS--Logistic Regression Best Parameters:', gs_lr_best_estimator.get_params())
print('GS--Logistic Regression Best K-Fold Score:', gs_lr.best_score_)
print('GS--Logistic Regression Test Score:', gs_lr.best_estimator_.score(X_test, y_test))


# KNN Grid Search

gs_knn = GridSearchCV(estimator=knn_pipe, param_grid=knn_params)
gs_knn.fit(X_train, y_train)
gs_knn_best_estimator = gs_knn.best_estimator_.named_steps['knn_clf']
print('GS--KNN Classifier Best Estimator:', gs_knn_best_estimator)
print('GS--KNN Classifier Best Parameters:', gs_knn_best_estimator.get_params())
print('GS--KNN Classifier Best K-Fold Score:', gs_knn.best_score_)
print('GS--KNN Classifier Test Score:', gs_knn.best_estimator_.score(X_test, y_test))


# Decision Tree Grid Search

gs_dt = GridSearchCV(estimator=dt_pipe, param_grid=dt_params)
gs_dt.fit(X_train, y_train)
gs_dt_best_estimator = gs_dt.best_estimator_.named_steps['dt_clf']
print('GS--Decision Tree Classifier Best Estimator:', gs_dt_best_estimator)
print('GS--Decision Tree Classifier Best Parameters:', gs_dt_best_estimator.get_params())
print('GS--Decision Tree Classifier Best K-Fold Score:', gs_dt.best_score_)
print('GS--Decision Tree Classifier Test Score:', gs_dt.best_estimator_.score(X_test, y_test))


# Support Vector Machine Grid Search

gs_svc = GridSearchCV(estimator=svc_pipe, param_grid=svc_params)
gs_svc.fit(X_train, y_train)
gs_svc_best_estimator = gs_svc.best_estimator_.named_steps['svm_clf']
print('GS--Support Vector Machine Classifier Best Estimator:', gs_svc_best_estimator)
print('GS--Support Vector Machine Classifier Best Parameters:', gs_svc_best_estimator.get_params())
print('GS--Support Vector Machine Classifier Best K-Fold Score:', gs_svc.best_score_)
print('GS--Support Vector Machine Classifier Test Score:', gs_svc.best_estimator_.score(X_test, y_test))


# Random Forest Grid Search

gs_rf = GridSearchCV(estimator=rf_pipe, param_grid=rf_params)
gs_rf.fit(X_train, y_train)
gs_rf_best_estimator = gs_rf.best_estimator_.named_steps['rf_clf']
print('GS--Random Forest Classifier Best Estimator:', gs_rf_best_estimator)
print('GS--Random Forest Classifier Best Parameters:', gs_rf_best_estimator.get_params())
print('GS--Random Forest Classifier Best K-Fold Score:', gs_rf.best_score_)
print('GS--Random Forest Classifier Test Score:', gs_rf.best_estimator_.score(X_test, y_test))

'''
# AdaBoost Grid Search

gs_adaboost = GridSearchCV(estimator=adaboost_pipe, param_grid=adaboost_params)
gs_adaboost.fit(X_train, y_train)
gs_adaboost_best_estimator = gs_adaboost.best_estimator_.named_steps['adaboost_clf']
print('GS--AdaBoost Classifier Best Estimator:', gs_adaboost_best_estimator)
print('GS--AdaBoost Classifier Best Parameters:', gs_adaboost_best_estimator.get_params())
print('GS--AdaBoost Classifier Train Score:', gs_adaboost.score(X_train, y_train))
print('GS--AdaBoost Classifier Test Score:', gs_adaboost.score(X_test, y_test))
print('GS--AdaBoost Classifier Best Score:', gs_adaboost.best_score_)

# Gradient Boost Grid Search

gs_gradboost = GridSearchCV(estimator=gradboost_pipe, param_grid=gradboost_params)
gs_gradboost.fit(X_train, y_train)
gs_gradboost_best_estimator = gs_gradboost.best_estimator_.named_steps['gradboost_clf']
print('GS--Gradient Boost Classifier Best Estimator:', gs_gradboost_best_estimator)
print('GS--Gradient Boost Classifier Best Parameters:', gs_gradboost_best_estimator.get_params())
print('GS--Gradient Boost Classifier Train Score:', gs_gradboost.score(X_train, y_train))
print('GS--Gradient Boost Classifier Test Score:', gs_gradboost.score(X_test, y_test))
print('GS--Gradient Boost Classifier Best Score:', gs_gradboost.best_score_)
'''
# Final report: despite some fitting errors with the logistic regression model, all of these classifiers are predicting a song as "popular" or "not popular" with about 64% accuracy
# These scores just marginally beat out the predictions of the models in their default implementation before hyperparameter tuning
# Attempt a RandomizedSearchCV to see if we can improve the test data

# HYPERPARAMETER TUNING USING RANDOM SEARCH CV
print('***********')
print('HYPERPARAMETER TUNING USING RANDOM SEARCH')
print('***********')

# Define hyperparameter distributions for each model

lr_dist = {'log_reg__penalty': ['l1', 'l2'],
             'log_reg__C': uniform(loc=0, scale=100)}
knn_dist = {'knn_clf__n_neighbors': randint(low=1, high=100)}
dt_dist = {'dt_clf__max_depth': randint(low=1, high=30)}
svc_dist = {'svm_clf__C': uniform(loc=0, scale=100)}
rf_dist = {'rf_clf__max_depth': randint(low=7, high=13)}
adaboost_dist = {'adaboost_clf__n_estimators': randint(5, 10)}
gradboost_dist = {'gradboost_clf__n_estimators': randint(20, 30)}
'''
# Logistic Regression Random Search

rs_lr = RandomizedSearchCV(estimator=log_reg_pipe, param_distributions=lr_dist)
rs_lr.fit(X_train, y_train)
rs_lr_best_estimator = rs_lr.best_estimator_.named_steps['log_reg']
print('RS--Logistic Regression Best Estimator:', rs_lr_best_estimator)
print('RS--Logistic Regression Best Parameters:', rs_lr_best_estimator.get_params())
print('RS--Logistic Regression Train Score:', rs_lr.score(X_train, y_train))
print('RS--Logistic Regression Test Score:', rs_lr.score(X_test, y_test))
print('RS--Logistic Regression Best Score:', rs_lr.best_score_)

# KNN Random Search

rs_knn = RandomizedSearchCV(estimator=knn_pipe, param_distributions=knn_dist)
rs_knn.fit(X_train, y_train)
rs_knn_best_estimator = rs_knn.best_estimator_.named_steps['knn_clf']
print('RS--KNN Classifier Best Estimator:', rs_knn_best_estimator)
print('RS--KNN Classifier Best Parameters:', rs_knn_best_estimator.get_params())
print('RS--KNN Classifier Train Score:', rs_knn.score(X_train, y_train))
print('RS--KNN Classifier Test Score:', rs_knn.score(X_test, y_test))
print('RS--KNN Classifier Best Score:', rs_knn.best_score_)

# Decision Tree Random Search

rs_dt = RandomizedSearchCV(estimator=dt_pipe, param_distributions=dt_dist)
rs_dt.fit(X_train, y_train)
rs_dt_best_estimator = rs_dt.best_estimator_.named_steps['dt_clf']
print('RS--Decision Tree Classifier Best Estimator:', rs_dt_best_estimator)
print('RS--Decision Tree Classifier Best Parameters:', rs_dt_best_estimator.get_params())
print('RS--Decision Tree Classifier Train Score:', rs_dt.score(X_train, y_train))
print('RS--Decision Tree Classifier Test Score:', rs_dt.score(X_test, y_test))
print('RS--Decision Tree Classifier Best Score:', rs_dt.best_score_)

# Support Vector Machine Random Search

rs_svc = RandomizedSearchCV(estimator=svc_pipe, param_distributions=svc_dist)
rs_svc.fit(X_train, y_train)
rs_svc_best_estimator = rs_svc.best_estimator_.named_steps['svm_clf']
print('RS--Support Vector Machine Classifier Best Estimator:', rs_svc_best_estimator)
print('RS--Support Vector Machine Classifier Best Parameters:', rs_svc_best_estimator.get_params())
print('RS--Support Vector Machine Classifier Train Score:', rs_svc.score(X_train, y_train))
print('RS--Support Vector Machine Classifier Test Score:', rs_svc.score(X_test, y_test))
print('RS--Support Vector Machine Classifier Best Score:', rs_svc.best_score_)
'''
# Random Forest Random Search
rs_rf = RandomizedSearchCV(estimator=rf_pipe, param_distributions=rf_dist)
rs_rf.fit(X_train, y_train)
rs_rf_best_estimator = rs_rf.best_estimator_.named_steps['rf_clf']
print('RS--Random Forest Classifier Best Estimator:', rs_rf_best_estimator)
print('RS--Random Forest Classifier Best Parameters:', rs_rf_best_estimator.get_params())
print('RS--Random Forest Classifier Best Score:', rs_rf.best_score_)
print('RS--Random Forest Classifier Test Score:', rs_rf.best_estimator_.score(X_test, y_test))

'''
# AdaBoost Random Search

rs_adaboost = RandomizedSearchCV(estimator=adaboost_pipe, param_distributions=adaboost_dist)
rs_adaboost.fit(X_train, y_train)
rs_adaboost_best_estimator = rs_adaboost.best_estimator_.named_steps['adaboost_clf']
print('RS--AdaBoost Classifier Best Estimator:', rs_adaboost_best_estimator)
print('RS--AdaBoost Classifier Best Parameters:', rs_adaboost_best_estimator.get_params())
print('RS--AdaBoost Classifier Train Score:', rs_adaboost.score(X_train, y_train))
print('RS--AdaBoost Classifier Test Score:', rs_adaboost.score(X_test, y_test))
print('RS--AdaBoost Classifier Best Score:', rs_adaboost.best_score_)

# Gradient Boost Random Search

rs_gradboost = RandomizedSearchCV(estimator=gradboost_pipe, param_distributions=gradboost_dist)
rs_gradboost.fit(X_train, y_train)
rs_gradboost_best_estimator = rs_gradboost.best_estimator_.named_steps['gradboost_clf']
print('RS--Gradient Boost Classifier Best Estimator:', rs_gradboost_best_estimator)
print('RS--Gradient Boost Classifier Best Parameters:', rs_gradboost_best_estimator.get_params())
print('RS--Gradient Boost Classifier Train Score:', rs_gradboost.score(X_train, y_train))
print('RS--Gradient Boost Classifier Test Score:', rs_gradboost.score(X_test, y_test))
print('RS--Gradient Boost Classifier Best Score:', rs_gradboost.best_score_)
'''
# Found marginal improvements in accuracy by using a random search, highest accuracy available is the Random Forest Classifier with a max depth of about 25-30--accuracy is about 65%
# Try AdaBoost Classifier and GradientBoost Classifier?
# After implementing AdaBoost and GradientBoost, resulted in very similar accuracy of about 64%.
# Try feature selection/elimination to improve accuracy?

# ERROR ENCOUNTERED: Sequential Feature Selector unable to handle data in numpy array format, which is how it is transformed through the pipeline preprocess
# ERROR IGNORED: Despite many attempts to handle the data differently and trying both ML-Extend's SFS and SKLearn's SFS, the error persisted.
# Try recursive feature elimination instead?
# SAME ERROR ENCOUNTERED AND IGNORED

rf_feature_ranks = rs_rf_best_estimator.feature_importances_

# helper function to get feature names

preprocessor = rf_pipe.named_steps['preprocess']

# Retrieve the feature names for each transformer in the ColumnTransformer
def get_feature_names(preprocessor):
    feature_names = []
    for name, transformer, columns in preprocessor.transformers_:
        if name == 'genre_preprocess':
            feature_names.extend(mlb.classes_)
        elif name == 'other_cat_preprocess':
            feature_names.extend(transformer.named_steps['ohe'].get_feature_names_out(columns))
        elif name == 'num_preprocess':
            feature_names.extend(columns)
    return feature_names

rf_feature_names = get_feature_names(preprocessor)

results = pd.DataFrame({'Feature Names': rf_feature_names, 'Importances': rf_feature_ranks})

results = results.sort_values(by='Importances', ascending=False)

print(results)
# Final thoughts: Best estimator is the Random Forest with a max depth of 9