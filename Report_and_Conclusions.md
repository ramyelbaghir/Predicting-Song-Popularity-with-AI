Predicting Song Popularity Using AI
=================================

# Project Goals
1. Determine what musical attributes are measured in the dataset
2. Use machine learning techniques and models to predict song popularity
3. Extract useful insights for musicians hoping to make music with more popular appeal

# Process

## Finding and Inspecting Data, Scoping
Inspecting the dataset and its description written by the publishers on Kaggle.com, it seemed that the data was collected to perhaps predict other features, since the "popularity" column had very little explanation to its entries. The songs were given a numerical score that wasn't clearly described by the authors--simply described with "The higher the value the more popular the song is." However, because the scores are numerical, I began with preparing the data for regression models.
## Regression

### Obstacles
The largest obstacle with preprocessing the data came with the "genre" column. The songs are labeled as multiple genres, and a simple OneHotEncoder would treat each unique genre combination as its own genre. I.e., "pop", "pop, rock", "pop, hip hop", and "pop, Dance/Electronic, rock" would all be considered separate genres.

Eventually, I came across the MultiLabelBinarizer object in sci-kit learn. However, this presented its own series of problems when attempting to implement it into the pipeline.
The final solution involved a series of helper functions and wrapping the MultiLabelBinarizer in a FunctionTransformer, another sklearn object, and fitting the training data to this transformer ahead of time before fitting the remainder of the pipeline.

### Results

After successfully preprocessing the data, the regression model pipelines were all more or less successfully implemented. They included a linear regression, a k-nearest-neighbors regression, a decision tree regression, a random forest regression, and a support vector machine regression.

However, the accuracy scores measured were abysmal, even on the training data. The highest test scores were around 2%, even after hyperparameter tuning using a GridSearchCV and RandomizedSearchCV.

So I tried a different approach: create a new column that categorizes the popularity score, and attempt the same process with classification models on this new column.

## Classification

### Obstacles

After creating the new column and dropping the old ones, the process was similar. A few errors were encountered here and there, but nothing major with the implementation.
Some more problems emerged when Sequential Feature Selection and Recursive Feature Elimination were applied to improve accuracy--online research and individual inspection did not reveal the source of the incompatibility. However, the overall accuracy scores of the models saw siginificant improvements across the board with the classification models, so ultimately I decided to ignore this obstacle and do without the feature selection/elimination.

### Results

The classification model pipeline scores saw huge improvements compared to the regression scores--roughly 62% accuracy across the board. These models included a logistic regression, k-nearest-neighbors classifier, decision tree, support vector machine classifier, random forest classifier, ada-boost classifier, and gradient boost classifier. I also attempted to include a Naive Bayes Classifier, but after encountering errors and doing further research I decided to exclude it--Naive Bayes Classifiers seem to be mostly used for text data, which is not related to this project.

After hyperparameter tuning, the best model found was a Ranodm Forest Classifier with a max depth of 9. This resulted in about a 65% accuracy score on test data. Success!

# Conclusion

## Biggest Predictors of Popularity

After extracting and evaluating the best-performing model (the random forest) and its feature importances, it was revealed that the feature with the most predictive power was the year released, accounting for about 11% of the predictive power across all features to a song being considered "popular" or not.

The next most predictive features in order of importance, accounting for roughly 4%-8% of predictive power each, are:
energy (8%)
valence (7.5%)
danceability (7.2%)
loudness (7.1%)
speechiness (6.9%)
tempo (6.9%)
acousticness (6.8%)
duration (6.8%)
liveness (6.7%)
instrumentalness (4.6%)

The remaining features' predictive power are all around 1.5% or less, suggesting that they have little influence over a song's popularity either way. These features include genre, a song's key center, its mode (major or minor), and whether it is labeled as 'explicit.'

## Limitations in Dataset

1. "Mode" is binary with only major and minor; doesn't account for other modes that can be characteristic of particular genres or artists (e.g., hip hop can often be in Phrygian modes, funk music can often be in Dorian modes, folk music can often be in Mixolydian modes, etc).

2. "Key" is structured on a 12-tone equal temperament scale; doesn't account for key changes, music that is more polytonal/atonal, or microtonal.

3. "Year" -- The dataset is titled "Top Hits Spotify from 2000-2019", however there are two large caveats to consider:
- Spotify was founded in 2006, so the source from where the data was gathered in years 2000 to 2006 is unclear.
- The role of novelty with respect to release year creates another discrepancy in the data. The earliest released songs in the dataset are from 1998. Songs' popularity in the dataset released before 2000 (the oldest data according to the title of the study) and before 2006 (Spotify's founding) will not be affected by novelty like their newer counterparts in the dataset released between 2006 and 2019. In simpler terms, listeners in 2015 are not listening to NSYNC because they're a new and exciting band. But listening to Migos in 2015 is a very different situation, as they were a much newer and more recent group in that year. These factors of novelty and longevity undoubtedly play non-negligible roles in a track's popularity, however novelty is not explicitly accounted for in the data. The popularity is measured as a total across all years, not within any given year measured.

4. Finally, with regards to popularity: as mentioned before, "popularity" is given a numerical score in the dataset that isn't clearly described by the authors--simply described with "The higher the value the more popular the song is." Additionally, this dataset is from Spotify's Top Hits, which is made up of songs that have already passed a certain threshold of popularity and appeal. As such, these results are less applicable to aspiring artists looking to break into the music industry, but rather more applicable to established artists and their producers looking to make the next big hit of their careers.

## Final Thoughts

Given this dataset and analysis, it would seem that aspiring musicians should highly consider their song's energy/activity, valence, danceability, and the other more predictive features listed above. This is not to say that one tempo, valence, or "liveness" score is superior to another. Rather, a musician should take these attributes seriously and ensure that they all work well together to create a harmonious whole--they seem to be more foundational to a given song's popularity than modality, genre, key center, or others.

Thank you for reading and I hope you gained something valuable out of this project report.