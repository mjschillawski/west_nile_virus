# Predicting West Nile Virus in the City of Chicago
Repo for GA-DSI Project 4: West Nile Virus in Chicago (Kaggle Competition)

Arash Ghafouri, Hector Martinez, Ken Yale, Puneet Kollipara, Michael Schillawski

Data Science Immersive, General Assembly, 9 March 2018

## Table of Contents

- [Repository Contents](#repository-contents) - Description of this repository's contents
- [Data Description](#data-description) - Description of dataset
- [Project Overview](#project-overview) - Summary of the project's goals
- [Analysis Explanation](#analysis-explanation) - Explanation of the project's methods and analysis
- [Project Concepts](#project-concepts) - Skills and concepts demonstrated

## Repository Contents

| FILENAME |  DESCRIPTION |
|:---------:|:-----------:|
| [README](./README.md) | Project description |
| [Cleaned Data](https://drive.google.com/open?id=1ufkmcXhP1cH5SZWU_jbFxkzuapyRgkjJ) | Cleaned data |
| [Project Notes](https://docs.google.com/document/d/1Yibk3n7HQaYOhnJOWlWfAm0LW7pbpHwGGwymSviIfRk/edit?usp=sharing) | Notes |
| [Presentation Deck](https://docs.google.com/presentation/d/1tx8yh1hB7GuOJhIJrUsmANXGqyYYvIOYoYWtXjavvCE/edit) | Slide deck |
| [Assets](https://github.com/mjschillawski/west_nile_virus/tree/master/assets) | Supporting materials, including data and graphs |
| | |
| [.gitignore](./.gitignore) | gitignore file |

## Data Description

Weather observations from Chicago O'Hare and Chicago Midway airports for 2007-2014. Mosquito abatement spraying records for 2011-2013, noting where and when sprays took place.

Lastly, mosquito trap observations on a weekly basis, noting where the trap was located, the mosquito species, how many mosquitoes were present, and whether West Nile Virus was present in the sample.

More information on the datasets is available here(https://www.kaggle.com/c/predict-west-nile-virus/data).

## Project Overview

The goal of this project is to predict, for a day, location, and mosquito species, whether West Nile Virus will be present in the trap observation. Ultimately, we aimed to maximize the area under the receiver-operator characteristic curve. 

The data was spread across several datasets, so one of the key steps was finding and setting the relationships between the data. We weighted the weather data based on the distance from the weather station to the trap observation. We created a qualitative spatiotemporal relationship between the mosquito traps and the spray abatements, identifying how far away the spraying occurred, both in time elapsed between the spraying and observation and the distance between them. This was reflected in flags for less than 1 week, less than 1 month, less than 1 quarter and within a half mile, within a mile, and within 5 miles.

We took several approaches to modeling this problem and pitted them head-to-head to determine the best model. This included k-nearest neighbors, random forests, extra trees, support vector machines, gradient boosting, and ADAboosting. 

## Analysis Explanation



## Project Concepts

Data munging; gradient boosting; gridsearch; cross-validation; haversine distance; amazon web services; exploratory data analysis; k-nearest neighbors; unbalanced classes; random forest classifier; extra trees classifier; adaboost; feature engineering; auc-roc.

