Sentiment Analysis for Book Character Labelling 📚

A machine learning project to classify fictional book characters as "Good" or "Bad" based on their textual descriptions.

Team ID: 19 | Project ID: 19

            Name	                 SRN
      B TEJA DEEP SAI KRISHNA	PES2UG23CS135
      BOBBA KOUSHIK           PES2UG23CS133



🎯 Problem Statement & Objective
This project tackles the text classification challenge of determining the moral alignment of fictional characters. Given a short description of a book character, the goal is to build a model that can accurately label them as either "Good" or "Bad". The primary objective was to compare five different classification algorithms to find the most effective model for this sentiment analysis task.

💾 Dataset Details
The dataset was custom-collected for this project.

Source: Character descriptions were manually gathered from online literary resources like LitCharts, SparkNotes, GradeSaver, and CliffsNotes.

Size: The final processed dataset used for binary classification contains 1,274 samples.

Features: The primary feature is the Description of the character.

Target Variable: The Verdict (Good/Bad). The dataset is moderately imbalanced, with 786 "Good" samples and 488 "Bad" samples.


⚙️ Project Pipeline & Methodology
The project followed a systematic workflow from data collection to model evaluation.

Data Collection & Cleaning:

An initial dataset of 1,442 records was analyzed.

The target variable, Verdict, was simplified from seven classes to a binary classification task ('Good' or 'Bad'). Ambiguous entries were removed.

Text Preprocessing:

A custom function was used to clean the character Description text.

This involved converting text to lowercase, removing URLs, punctuation, and numbers.

Standard English stopwords were filtered out using the NLTK library.

Feature Engineering:

The cleaned text was converted into numerical vectors suitable for machine learning.

TF-IDF and Count Vectorization techniques were used with an n-gram range of (1, 2), resulting in 2,391 features.

Model Training & Evaluation:

The data was split into 80% for training and 20% for testing.

Five different classification models were trained and evaluated to find the best performer.


🤖 Models Compared
The following five machine learning models were trained and compared:

Logistic Regression

Multinomial Naive Bayes

Support Vector Machine (SVM) with a Linear Kernel

Random Forest Classifier

Gradient Boosting Classifier
📊 Results & Evaluation
The models were evaluated on Accuracy, Precision, Recall, F1-Score, and AUC. The Multinomial Naive Bayes model emerged as the clear winner. 🏆

Model Performance Comparison



      Model	                Accuracy   AUC Score	F1-Score (Bad)	F1-Score (Good)
      Logistic Regression	  0.7176	  0.8550	       0.48	          0.81
      Naive Bayes	          0.8118	  0.8601	       0.74	          0.85
      SVM	                  0.8000	  0.8321	       0.72	          0.84
      Random Forest      	  0.7333	  0.7982	      0.71	          0.75
      Gradient Boosting  	  0.7373	  0.7715	      0.58	          0.81  



The Multinomial Naive Bayes model demonstrated the best ability to correctly identify both "Good" and "Bad" characters, achieving a balanced and high-performing result.

✅ Conclusion
This project successfully developed an effective pipeline for classifying book characters based on their descriptions. By comparing five distinct models, the Multinomial Naive Bayes classifier was identified as the most suitable, achieving a test accuracy of 81.18%.

File Structure

├── Machine_Learning_Project_Team_19_Project_ID_19 (1).ipynb

├── Character.csv

├── logistic_regression_model.joblib

├── naive_bayes_model.joblib

├── svm_model.joblib

├── random_forest_model.joblib

├── gradient_boosting_model.joblib

├── tfidf_vectorizer.joblib

├── count_vectorizer.joblib

├── label_encoder.joblib

├── streamlit_app.py

├── style.css

├── book_animation.html

└── README.md
