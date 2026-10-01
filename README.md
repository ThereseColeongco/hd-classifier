# Heart Disease Classifier
Cardiovascular diseases (CVDs) are the leading cause of death worldwide. According to the [WHO](https://www.who.int/health-topics/cardiovascular-diseases), "More than four out of five CVD deaths are due to heart attacks and strokes, and one third of these deaths occur prematurely in people under 70 years of age." 

This project uses XGBoost classification and the combined 920-sample [UCI heart disease dataset](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data?resource=download) on Kaggle to predict presence/absence of heart disease. This dataset includes data from clinics in Cleveland, Hungary, Switzerland, and the VA Long Beach.

As between 0.22% and 52.83% of data is missing for certain features, rows may have to be dropped and/or data imputed. This and exploratory data analysis must be done before any training.

I chose to use XGBoost classification because a nearly identical UCI heart disease dataset featuring data from the same combination of clinics showed highest accuracy and precision with this type of model at baseline. (Logistic regression came close but its upper bound for precision is slightly less.)