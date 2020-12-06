 <div style=><img src="https://camo.githubusercontent.com/52d2ff8778b60261533a7dba8dd989c6893a519b/68747470733a2f2f692e696d6775722e636f6d2f315167724e4e772e706e67"/></div>

# Diamonds prices ML prediction
# Winning the Inclass Kaggle Competition


 <div style="text-align:center"><img src="img/banner.jpg" height=200 /></div>

 The scope of the project is to **optimize** the prediction of diamonds prices as part of this [Inclass Kaggle competition](https://www.kaggle.com/c/diamonds-datamad1020-rev/), being 'optimize' understood as to improve as much as possible the model metric, not to find a sweet balance between great metric and explanability. 

 My final position was **Top1**. 
 
 Different models have been tried: **LinearRegressor**, **KNNRegressor**, **RandomForest**, **GradientBoosting**, and **stacking** of previous type of models with a meta-model of **LassoRegressor**  and **RandomForest**.


 ## Introduction

The rules for the competition is as follows:

* Submissions are limited to 5 times a day.
* The test set is divided into a public part (with which the public leaderboard is calculated, accessible during the competition) and another private part, with which the final positions are calculated, after the end of the competition.
* Train a minimum of 4 different models.
* Metric is MSE.

These are the features of the data:

* id: only for test & sample submission files, id for prediction sample identification
* price: price in USD
* carat: weight of the diamond
* cut: quality of the cut (Fair, Good, Very Good, Premium, Ideal)
* color: diamond colour
* clarity: a measurement of how clear the diamond is
* x: length in mm
* y: width in mm
* z: depth in mm
* depth: total depth percentage = z / mean(x, y)
* table: width of top of diamond relative to widest point 

## Description

* After a first exploration of the data, the categorical columns (i.e. cut, color, clarity) were identified as ordinal categorical data. For further information about color and clarity categories, please refer to [here](https://www.brilliance.com/education/diamonds/color) and [here](https://4cs.gia.edu/en-us/diamond-clarity/), respectively.

* Depth column is removed as it is just a mathematical combination of other variables.

* Data given in the competition for training is split into train data and test data with a test size of 0.8 .

* Tuning of hyperparameteres is done by crossvalidation with CV = 5.


## Models

#### Linear Regression

Color and clarity ordinal categories are values with two different approaches each based on the information shown in the references included above. These are the options:

    * Clarity Option 1 
        Proportional relation bewteen categories
```python
{'I3': 1 , 'I2': 2, 'I1': 3, 'SI2': 4, 'SI1': 5, 'VS2': 6, 'VS1': 7, 'VVS2': 8, 'VVS1': 9, 'IF': 10 }

```
    * Clarity Option 2 
        Higher gap between major categories
```python
{'I3': 1 , 'I2': 1.5, 'I1': 2, 'SI2': 4, 'SI1': 4.5, 'VS2': 6.5, 'VS1': 7, 'VVS2': 9, 'VVS1': 9.5, 'IF': 11.5 }

```

    * Color Option 1 
        Proportional relation bewteen categories
```python
{'D': 10 , 'E': 9, 'F': 8, 'G': 7, 'H': 6, 'I': 5, 'J': 4, 'K': 3, 'L': 2, 'M': 1}

```
    * Clarity Option 2 
        Higher gap between major categories
```python
{'D': 10 , 'E': 9.5, 'F': 9, 'G': 7, 'H': 6.5, 'I': 6, 'J': 5.5, 'K': 3.5, 'L': 3, 'M': 2.5}

```

Combination of those 4 is checked. No major differences bewteen those, being the best the combination of both Options 1.

No hyperparameter tuning. Based on the comparison of the metric between train and test split, seems the model is not overfitting and that's why regularization with Lasso or Ridge is not implemented.

### KNN Regression

Same approach related to ordinal categories as in Linear regression.
No significant differences between combinations, being this time the best combination the one with Color option 1 and Clarity option 2.

A **pipeline** is carried out to incldue a **standarization** step by `StandardScaler` previous model fitting.

`n_neighbors` hyperparameter is tuned via crossvalidation. Best hyperparameter value found is 6 neighbors. 

### Random Forest

The hyperparameters tuned by crossvalidation are `n_estimators`, `max_features`, `max_depth` and `min_samples_leaf`.

The approach, which will be followed similarly for GradientBoosting is to first via **GridSearch**, define a broad range of hyperparameters in order to find best model and later, fine-tuning.


### GradientBoosting

The hyperparameters tuned by crossvalidation are `n_estimators`, `learning_rate`, `max_features`, `max_depth` and `min_samples_leaf`.

Initially, `n_estimators` is fixed in order to get a feeling of the other hyperparameters and later on, `n_estimators` is optimized during fine-tuning.


### Stacking

Base models are previous models which have learnt only the train split. Two meta models are tried: 
* Lasso Regressor
* RandomForest

## Results

I won the competition submitting the following 3 candidates:

* GradientBoosting: 

    * `n_estimators`: 30000
    * `learning_rate`: 0.0032
    * `max_features`: 0.4
    * `max_depth`: 8
    * `min_samples_leaf`: 10

    Public score: 0.00721

    Private Score: 0.00719

* GradientBoosting: 

    * `n_estimators`: 22000
    * `learning_rate`: 0.0032
    * `max_features`: 0.4
    * `max_depth`: 8
    * `min_samples_leaf`: 10

    Public score: 0.00723

    Private Score: 0.00723

* Stacking with LassoRegressor as meta-model: 

    * `alpha`: 7.4437e-07

    As base models:
        * LinearRegressor
        * KNN(n_neighbors = 6)
        * RandomForest(n_estimators= 300, max_features=0.6, max_depth= 150, min_samples_leaf=1)
        * GradientBoosting(hypyerparameters as the one above)

    Public score: 0.00726

    Private Score: 0.00723


My final score was 0.00719. The following competitor (rank#2) got a final score of 0.00734.

It is worth to mention that I obtained a slightly better result with a model I did  not submit which is the following:

* GradientBoosting: 

    * `n_estimators`: 18000
    * `learning_rate`: 0.0033
    * `max_features`: 0.4
    * `max_depth`: 9
    * `min_samples_leaf`: 10

    Public score: 0.00724

    Private Score: 0.00717




## Repo Structure
* `predict.ipynb` : working notebook.
* `src`:
    * `conditioning.py`: auxiliar functions for modularization.
* `img`: folder containing pictures for repo. 


## Further developments
* Need to modularize further my code (I prioritized my time finding the best model).
    
* I foresee there is room for improvement with the stacking approach. Not enough time to fine-tune it and try other configurations.


## Technologies and Environment

Python3 on Ubuntu 20.04

* __[sklearn](https://scikit-learn.org/stable/index.html)__






















 
