# Classifying Asteroids

In this project, our goal is to predict whether or not an asteroid is hazardous using machine learning techniques. In this README.md file I will provide an overview of our project. I will also attach the dataset, the code we used, and the full written report.
Our asteroid dataset comes from Kaggle.com, and it was recorded by the Jet Propultion Laboratory of the California Institute of Technology (JPL).
Since potentially hazardous asteroids (PHA) can be classified by 'yes' or 'no', we decided to use logistic regression in our analysis.

After preprocessing our data, our first step was to eliminate unnecessary columns. We determined which columns to drop based on a correlation matrix. If two variables have a strong correlation we will drop one of those columns.

![Unknown](https://github.com/JasonBauer26/Asteroid_Classification/assets/145518855/f1c59bca-d000-4195-b0a3-b081792e2630)

Below is a list of the columns we decided to drop.

<img width="682" alt="Screenshot 2023-12-21 at 6 03 34 PM" src="https://github.com/JasonBauer26/Asteroid_Classification/assets/145518855/7db0b307-d6b6-48e1-b60f-3816be6d58f4">

Our last step before the analysis is to convert the 'pha' column values from 'Y' and 'N' to '1' and '0'.

Below is a plot of hazardous vs non-hazardous frequencies in the dataset. We thought this could be valuable since a large discrepancy between these two could bring up issues in our model.

![Unknown-2](https://github.com/JasonBauer26/Asteroid_Classification/assets/145518855/af05b132-6be3-4ae3-9814-ea00902b4958)

As we can see, there is a large discrepancy between hazardous and non-hazardous asteroids. 

We begin our analysis by selecting features through random forest classification. The image below shows which of the variables were selected by the random forest classfication.

<img width="639" alt="Screenshot 2023-12-21 at 6 22 00 PM" src="https://github.com/JasonBauer26/Asteroid_Classification/assets/145518855/8e16d938-21f7-4de5-bf02-46c34368478f">

The next step is to train and fit our logistic regression model. The confusion matrix below highlights the results from fitting the model.

![Unknown-3](https://github.com/JasonBauer26/Asteroid_Classification/assets/145518855/ee281cfc-990d-45ea-8931-380630aa39b1)

We can see that the model seemingly performed perfectly. While this seems like a favorable result, it is most likely due to the discrepancy in the number of hazardous cases vs the number of non-hazardous cases.

At this point, we decided to try Random Forest Classification with GridSearchCV and XGBoost to compare results with our Logistic Regression model.

<img width="639" alt="Screenshot 2023-12-21 at 6 34 45 PM" src="https://github.com/JasonBauer26/Asteroid_Classification/assets/145518855/7197601a-3a33-4bca-a7ec-100a40513140">

We can see from the image that Random Forest Classifier and XGBoost performed much better than our initial Logistic Regression. Our next steps would be to use methods specializing in unbalanced data.
One of these methods is called SMOTE (Synthetic Minority Oversampling Technique). This method increases the number of minority instances in the dataset using the existing minority cases.
We think this method, combined with XGBoost, could perform even better than XGBoost alone. The next iteration of the project will focus on SMOTE with XGBoost.

