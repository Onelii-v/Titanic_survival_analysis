# Titanic Survival Prediction
 
A machine learning project that predicts whether a passenger survived the Titanic disaster, using the classic Kaggle Titanic dataset and a Logistic Regression model.

## Overview
 
This is a project I worked on to practice the basics of data science: cleaning messy data, creating new features, and training a model to predict survival (`Survived`: 0 = did not survive, 1 = survived)

## Approach
 
**1. Exploratory Data Analysis**
- Checked for missing values across all columns.
- Found that ~80% of the `Cabin` column was missing, so instead of dropping the information entirely, created a new binary feature, `cabin_flag` (1 = cabin info available, 0 = not), then dropped the original `Cabin` column.
- Visualized survival rate by cabin info availability and by age (via bar plots and box plots).
**2. Handling Missing Values**
- Compared three strategies for imputing missing `Age` values — mean, median, and mode — using box plots against the `Survived` outcome to see which preserved the original distribution best.
- Filled missing `Age` values using the median.
- Filled the two missing `Embarked` values with the mode (most frequent port).
**3. Feature Engineering & Encoding**
- Scaled `Age` and `Fare` using `StandardScaler`.
- Converted `Sex` into a binary numeric column (male = 0, female = 1).
- One-hot encoded `Embarked` into `Embarked_C`, `Embarked_Q`, and `Embarked_S`.
- Dropped non-predictive or redundant columns (`PassengerId`, `Name`, `Ticket`, raw `Age`/`Fare`, and the intermediate mean/median/mode age columns used for comparison).
**4. Modeling**
- Split the data into training and test sets (80/20 split, `random_state=42`).
- Trained a **Logistic Regression** model on the processed features.
- Evaluated performance using a classification report.

## Tech Stack
 
- Python
- pandas
- seaborn / matplotlib
- scikit-learn
