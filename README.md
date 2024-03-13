# TurinTech
**Objective**: Practice understanding of PyCharm IDE, Github, and basic Python libraries for data analysis and machine learning.

## Iris Dataset Machine Learning Project Summary

This project demonstrates a complete machine learning workflow using the famous Iris dataset. The Iris dataset includes data on sepal length, sepal width, petal length, and petal width of 150 iris flowers, categorized into three species: Iris-setosa, Iris-versicolor, and Iris-virginica. The goal is to build models that can accurately classify the species of iris flowers based on these features.

### Key Steps in the Project:

1. **Data Loading**: The Iris dataset is loaded from a public URL into a pandas DataFrame for manipulation and analysis.

2. **Data Exploration**: Initial data exploration is conducted to understand the dataset's structure, including checking the dimensions, the first few rows, statistical summaries, and the distribution of instances across the three classes.

3. **Data Visualization**: Various plots, including box plots, histograms, and scatter matrices, are generated to visualize the dataset's distribution and relationships between features.

4. **Preprocessing**: The dataset is split into feature variables (`X`) and the target variable (`y`), followed by splitting into training and validation sets to prepare for model training and evaluation.

5. **Model Selection**: Several machine learning models are evaluated, including Logistic Regression, Linear Discriminant Analysis, K-Nearest Neighbors, Decision Tree, Gaussian Naive Bayes, and Support Vector Machine. Stratified k-fold cross-validation is used to assess each model's performance objectively.

6. **Model Evaluation**: The models are compared based on their cross-validation accuracy scores, visualized through box plots to identify the top-performing model(s).

7. **Final Model Training and Prediction**: The Support Vector Machine (SVM) model, identified as the most appropriate based on the evaluation, is trained on the training set and used to make predictions on the validation set.

8. **Results Analysis**: The predictions of the final model are evaluated using accuracy score, confusion matrix, and a classification report, providing insights into the model's performance in classifying the iris species.

### Conclusion:

The project illustrates an end-to-end machine learning workflow, from data loading and exploration through model training, evaluation, and final predictions. The SVM model emerged as the most suitable classifier for this dataset, demonstrating high accuracy in distinguishing between the three species of iris flowers.

## Reference

1. Brownlee, J. (n.d.). Your First Machine Learning Project in Python Step-By-Step. Python Machine Learning. https://machinelearningmastery.com/machine-learning-in-python-step-by-step/
