# AQI-Prediction
> ## by Shivam Chauhan

## Installation
These notebooks runs on anaconda environment and the requirements of this environment can be installed using the file “requiremnt.txt”

## Introduction
This project was a part of final submission of my Predictive Analytics course at Northeastern University in Spring 2020. In this project I have applied various Machine Learning Regressor model to predict the Air Quality Index of a city. I have extended this project in 2021 to extract live data and predict AQI of a city. Using Apache Airflow DAG we will schedule our application to extract data daily, perform required transformation and load it to the database. Then these models can be applied or re-trained to predict AQI. 

## Files in this repository
>- [`create_db.py`](https://github.com/Chauhanshi/AQI-Prediction/blob/master/create_db.py): Created a database and table to store daily weather data
>- [`open-weather_ETL.ipynb`](https://github.com/Chauhanshi/AQI-Prediction/blob/master/open-weather%20ETL.ipynb): Extrating weather data data using Rapid API and prepare it to store in the database. 
>- [`All_models.py`](https://github.com/Chauhanshi/AQI-Prediction/blob/master/All%20models.py): Includes Loading the data, Training the data for different Machine Learning models like Decision Tree, Random Forest, XgBoost, KNN, and ANN.  
