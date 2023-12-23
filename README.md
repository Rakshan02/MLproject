## Home Credit Risk Prediction | Machine Learning project

Home Credit Group is an international consumer finance provider which was founded in 1997 and has operations in 9 countries. Our responsible lending model empowers underserved customers with little or no credit history by enabling them to borrow easily and safely, both online and offline. 

Home Credit offers easy, simple and fast loans for a range of Home Appliances, Mobile Phones, Laptops, Two Wheeler's , and varied personal needs.

## Exploratory Data Analysis

Data Exploration is an open-ended process where we calculate statistics and make figures to find trends, anomalies, patterns, or relationships within the data. The goal of Data Exploration is to learn what our data can tell us. It generally starts out with a high level overview, then narrows in to specific areas as we find intriguing areas of the data. The findings may be interesting in their own right, or they can be used to inform our modeling choices, such as by helping us decide which features to use.

We use Label Encoding for any categorical variables with only 2 categories and One-Hot Encoding for any categorical variables with more than 2 categories.

## Supervised Learning Techniques

Algorithms used:
Logistic regression
Random Forest
Extreme Gradient Boost

We have created base models without balancing the target class and then We have used SMOTE and Random OverSampler methods to balance the data. XGBoost model balanced using Random OverSampler method gave better accuracy and ROC AUC score.

## Built With

• pandas - Pandas is used for data cleaning and analysis. Its features were used for exploring, cleaning, transforming and visualizing the data. Also Pandas is an open-source python package built on top of Numpy.

• numpy - Numpy adds support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays.

• Matplotlib - Matplotlib work like MATLAB that helps in data visualization to makes some change to a figure: e.g., creates a figure, creates a plotting area in a figure, plots some lines in a plotting area, decorates the plot with labels, etc.

• seaborn - Seaborn is also used for data visualization and it is based on matplotlib. It provides a high-level interface for attractive and informative statistical graphics.

• Sklearn - Scikit-learn is a free machine learning library for the Python. It features various classification algorithms including 'LogisticRegression', 'RandomForsetClassifier', 'DecisionTreeClassifier',  'KNeighborsClassifier','AdaBoostClassifier','GradientBoostingClassifier', 'XGBClassifier which were used in this project. It also helps in calculating the metrics such as classification report, accuracy score, f1 score, roc auc score and confusion matrix.

• Sweetviz is an open-source Python library that generates beautiful, high-density visualizations to kickstart EDA (Exploratory Data Analysis) with just two lines of code. Output is a fully self-contained HTML application.
 




In conclusion, the Random Forest Classifier is recommended for production use due to its balanced performance in identifying loan default cases, taking into account the challenges faced during data analysis and model evaluation. Further model fine-tuning and interpretability techniques should be considered for enhancing its effectiveness in real-world lending scenarios.

Double-click (or enter) to edit Final Conclusion Report Performance Analysis of Machine Learning Models for Loan Repayment Prediction

In this comprehensive analysis, we evaluated multiple machine learning models on a loan repayment dataset to identify the most suitable model for production use. Our evaluation considered various performance metrics, including accuracy, precision, recall, F1-score, and area under the ROC curve (AUC-ROC). Additionally, we addressed key challenges in the dataset to ensure robust model performance. Model Performance Metrics

We examined the performance of seven different machine learning models:

Logistic Regression
Random Forest Classifier
Decision Tree Classifier
K-Nearest Neighbors Classifier
AdaBoost Classifier
Gradient Boosting Classifier
XGBoost Classifier

Here are the key performance metrics for each model: Model Accuracy Precision Recall F1-Score AUC-ROC

Logistic Regression 0.9173 0.6765 0.0187 0.0364 0.7293 

Random Forest Classifier 0.9167 0.6552 0.0357 0.0679 0.7651 

Decision Tree Classifier 0.8477 0.2489 0.1898 0.2166 0.5960 

K-Nearest Neighbors Classifier 0.9124 0.4783 0.0612 0.1085 0.6255 

AdaBoost Classifier 0.9166 0.6481 0.0372 0.0703 0.7667 

Gradient Boosting Classifier 0.9188 0.7148 0.0251 0.0486 0.7620 

XGBoost Classifier 0.9193 0.7235 0.0221 0.0429 0.7671 

Model Selection

To determine the best model for production use, we considered the specific requirements of the lending application. The lending industry places a high value on recall to correctly identify loan applicants who may default.

While the XGBoost Classifier achieved the highest accuracy and precision, it had a relatively low recall and F1-Score. On the other hand, the Random Forest Classifier performed well with a high accuracy, and it exhibited a better balance between recall and precision, making it suitable for our application.

Therefore, we recommend the Random Forest Classifier for production use due to its balanced performance in identifying loan default cases. Challenges Faced and Techniques Used

During our analysis, we encountered several challenges in the dataset and employed techniques to address them:

    Imbalanced Dataset

    Challenge: The dataset had a significant class imbalance, with a much higher number of non-default cases compared to default cases. Technique: We applied the Synthetic Minority Over-sampling Technique (SMOTE) to oversample the minority class (default cases) in the training dataset. This helped mitigate the class imbalance issue and improved model performance.

    Missing Data

    Challenge: The dataset contained missing values in various features, which could adversely affect model training. Technique: We used a custom DataFrameImputer transformer to impute missing values in both numerical and categorical features. Numerical features were imputed with medians, while categorical features were imputed with the most frequent values.

    Feature Engineering

    Challenge: The dataset contained a large number of features, some of which might not be relevant for predicting loan repayment. Technique: We performed feature selection and importance analysis using techniques such as RandomForestRegressor. This allowed us to identify and focus on the most influential features for model training and interpretability.

    Model Interpretability

    Challenge: Understanding and interpreting complex machine learning models is crucial for decision-making in the lending domain. Technique: We employed feature importance analysis to gain insights into the factors contributing to loan defaults. Additionally, SHAP (SHapley Additive exPlanations) values and model-specific interpretability tools can be used to enhance model interpretability.

Conclusion

In conclusion, the Random Forest Classifier has been recommended for production use due to its balanced performance in identifying loan default cases, taking into account the specific challenges faced during data analysis and model evaluation. Further model fine-tuning and interpretability techniques should be considered for enhancing its effectiveness in real-world lending scenarios.

This comprehensive analysis provides valuable insights for making informed decisions in the lending industry, contributing to more accurate loan applicant assessments and reducing the risk of defaults.
