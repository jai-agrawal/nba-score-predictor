# Predicting the Score of an NBA Game with Machine Learning
One may elect for many different options whilst contemplating how to predict the score of an NBA game, i.e., the Home/Away team scoring x points. It's important 
to distinguish these predictions as either being real-time/live or pre-game. One could thus use machine learning techniques such as Linear Regression and Gradient
Boosting Regression on specific data and labels in order to further predict certain labels (in this case - the final score of the home team) using certain data (in 
this case - scores from the end of the 3rd quarter). 

## Data Used
The data used in this project was previously scraped from stats.nba.com and then formatted for simplicity into a .csv file. More speciifically it conforms to the dates, final, and 3rd-Quarter scores of all matchups that occurred during the 2015-2016 NBA season. The same data was used in the nba-outcome-predictor 

## Working of the Code
As shown in main.py, the overall flow involves itself with the insertion of the data as a pandas DataFrame, the further splitting of said data into data (x) and 
labels (y). Further, the data is then split into training data and testing data (80-20). Then, the data is scaled using StandardScaler(). Since the label is the 
home team score, the best fit prediction will conform to home-team scores as well. Thus, the Linear Regression model is initialised and fitted. One can then use 
.predict() to find predicted scores for the test dataset. However, since a score is a very specific value and could hardly be accurately predicted, one could then 
find a range values rather than a single one. For this, we give confidence intervals to our prediction using a Gradient Boosting Regressor. As shown in 
confidenceIntervals.py's 'find_ci()' function, we find a range of predicted values fo the home-team's final score. The separate range values are then rounded off, 
converted into tuples and then returned to the main code. Finally, all the findings are added to a 'predictions' dataframe. A sample of the final DataFrame looks like: 

<img width="418" alt="Screenshot 2021-03-26 at 4 49 35 PM" src="https://user-images.githubusercontent.com/77375209/112624177-49694e80-8e53-11eb-9951-45f6a50843d2.png">

All of the machine learning algorithms used in this code have been implemented using sklearn. 

## Evaluation 
The test data is scored 0.7 approximately using .score(). Prediction of other seasons using the same model yielded scores of around 0.65-0.73. One could use more 
than one season to fit the data, which could yield higher scores. However, one should beware of overfitting the data.
