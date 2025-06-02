# Calorie Prediction Kaggle Competition

This project was for the [Predict Calorie Expenditure Kaggle Competition](https://www.kaggle.com/competitions/playground-series-s5e5) Where the objective was to most accurately predict calorie expenditure for a person given their personal information and activity data. The compeition ran from May 1st to May 31st, 2025.

This competition was evaluated using Root Mean Squared Logarithmic Error (RMSLE) as the metric, which measures the average magnitude of the errors between predicted and actual calorie expenditure values. The lower the RMSLE, the better the model's performance. There was a public leaderboard, showing cursory evaluations of submitted models and a private leaderboard, which was revealed at the end of the competition to determine final rankings.

There were 9,345 participants in the competition. I finished 755/9,345 with an RMSLE of 0.05883; a 0.00042 difference from the winning RMSLE of 0.05841.

# Data
The dataset for this competition was synthetically generated using a deep learning model trained on a Calories Burnt Prediction dataset from a previous Kaggle competition. We were provided with training and test datasets, along with a sample submission file to guide the format of our predictions. The distribution of the data in the synthetic dataset was similar to the original dataset, but not identical.

**Files included:**
- `train.csv`: Training data with the target column `Calories`.
- `test.csv`: Test data for which calorie predictions are required.
- `sample_submission.csv`: Example submission format.

**Columns:**
- `id`
- `Sex`
- `Age`
- `Height`
- `Weight`
- `Duration`
- `Heart_Rate`
- `Body_Temp`
- `Calories`

# Solutions

It took me 7 versions and various sub-versions to reach my final score. The final model was a CatBoosting Regressor with hyperparameter tuning and extensive feature selection analysis. This section will briefly describe each version leading up to this one and the improvements made along the way.

## v1.0: Neural Network
To start most machine learing projects, I build a simple neural network using PyTorch. Very little effort gets put into this model. I use all features, no feature selection, best-practice hyperparameters, and no hyperparameter tuning. I use this to compare future models against, but I also had the advantage of knowing some teams had already achieved public RMLSEs of 0.06.

It achieved an RMSLE of 0.09670.

## v2.0: XGBoost
After v1.0, I felt a NN may be inefficient for the problem at hand, so I switched to XGBoost, a popular gradient boosting technique and again did very minimal work on the rest of the design.

It achieved an RMSLE of 0.09496.

### v2.1: XGBoost with Hyperparameter Tuning
I then used Optuna to perform hyperparameter tuning on the XGBoost model. I used a small number of trials, but still managed to improve the model to a small degree.

It achieved an RMSLE of 0.09369.

## ensemble-v1.0: XGBoost, CatBoost, and LightGBM
At this point I made a developmental mistake. I became too focused on the high leaderboard scores and thinking of ways to maximize model design, rather than focusing on the best principles of machine learning, which is to focus early efforst on feature selection and engineering.

I created an ensemble of XGBoost, CatBoost, and LightGBM models. I used the same hyperparameters as the XGBoost model in v2.1, but I did not perform hyperparameter tuning on the CatBoost and LightGBM models. In a stroke of luck, this model showed significant improvement over the previous models, due primarly to the fact I learned later that CatBoost handles this problem particularly well.

It achieved an RMSLE of 0.05897.

### ensemble-v2.0-v5.3
These model versions all focused on various improvements to model design. Some of the many designs tested were:
- Isotonic Regression on the output of the ensemble model to improve the final predictions.
- Linear Regression on the output of the ensemble model to improve the final predictions.
- Ensemble model weighting to maximize the best performing models.
- Adding Optuna hyperparameter tuning to all models.
- Creating derived features to enhance the dataset.

All these enhancements only provided an 0.00014 improvement over the previous ensemble model, which I was pleased to see, but it was not worth the effort, since there was still a 0.002 gap to close in the leaderboards. I also discovered in these iterations that a solo CatBoost model outperformed my ensemble, so I pivoted to a CatBoost only model and began more rigorous feature selection and engineering.

## CatBoost v1
After moving to a solo CatBoost model, I performed deep feature analysis using various feature selection techniques, including SHAP value analysis, feature importance, correlation analysis, and by-hand testing. After the new features were selected, I reperformed hyperparameter tuning.

It achieved an RMSLE of 0.05879, which was the final result I achieved prior to the submission deadline.

# Conclusion
These competitions are always a mix of fun and learning for me, because it's nice to tackle a "classical" machine learning problem with a defined target and defined dataset from time to time. That makes it a nice bonus place so closely to the winning team's score, but at the same time I can take a lesson from this, mainly to remind myself to not get distracted by small improvments when bigger leaps need to be made.
