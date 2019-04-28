# 20 Newsgroups Text Dataset Binary and Multi-Class Classification

For Binary Classification, Linear SVM, Logistic Regression and Gaussian Naive Bayes were performed

For Multi-Class Classification, Logistic Regression (both One vs Rest and One vs One) and Gaussian Naive Bayes were performed

Dimensionality Reduction was compared between Latent Semantic Indexing (LSI) aka truncated SVD and Non-negative Matrix Factorization (NMF)

Logistic Regression with certain parameters and LSI was found to be best for binary classification.

Logistic Regression (One vs Rest) with certain parameters was found to be best for multi-class classification.

To run the entire code, simply run 
```
python3 -i project1.py
```

Since Q7 Grid Search takes a long time to run, 
we have stored our results in a pickle file
To view the result in the pickle file, run the following command
```
python3 -i Q7_load_results.py
```

This was Project 1 for EE219 Large-Scale Data Mining: Models and Algorithms
