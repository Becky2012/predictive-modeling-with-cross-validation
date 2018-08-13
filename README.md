Purpose: The purpose of the program is to build multiple machine learning models for Census adult income data. Thus, the classification goal is to predict the most important variables or 10 out of 15 factors contributing to income greater than 50k.

The machine learning models selected to use in this project include: Decision Tree, Random Forest and Logistic Regression.

Data are imported as pandas dataframe, filtered by intact records, encoded on categorical variables using onehot scheme, normalized using min-max scaler, and divided into training and testing datasets. The program evaluates the performance of all machine learning models used in this project by calculating the accuracy of cross validation scores. Then the program split off the data into train data and test data to build the machine learning models and  predicts the top10 predictors and their weights(parameters) for each  models. All cross validation scores and the results of prediction are presented in top_vars.xls as one of output files. Also, the importance of variables are sorted by descending order and visualized using bar plots. The bar plots is presented in a file named as "plot.pdf".

Program name: The machine learning models are run by cv.py

Data source: Detailed data description and data could be downloaded from the website https://archive.ics.uci.edu/ml/datasets/adult
