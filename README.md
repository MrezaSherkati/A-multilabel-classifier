# A-multilabel-classifier
This project was about designing a multilabel classifier on a dataset with high number of features and samples.
# Dataset charactrisitics
We were provided with:  
– 25,000 Samples, each with up to 19 classes (multiclass; non-exclusive)  
  • Binary class data in labels_train.csv (25K rows x 19 columns)  
– R1_train.csv through R6_train.csv files containing six different feature sets  
  • One sample per row. Each feature set has 25K rows.  
  • Feature sets 1-3 have 768 features each. Feature sets 4-6 have 1024 features each.  
• Blind test set had 25K samples.  
# Purpose of the project and performance metric
The purpose of the project was to design a classifier which trains on one of the featuresets and then gets tested on a blind testset. The performance metric which was used for assessing the models was micro average F1-score.
# Experiment design
The experiment design phase serves as the foundation for our pattern classification endeavor. In order to make the most out of our dataset, we used 5 fold cross validation for estimating the performance of our model on unseen future data and in each fold, we adopt an 80-20 split in the training subset for training and validation, respectively. We used the KFold() method from Scikit Learn library for creating the different folds for our cross validation setting. Class balancing considerations are also explored to mitigate biases and ensure the robustness of our classifier.
# Preprocessing steps on dataset
The following preprocessing steps is performed on the featureset to enable our model to perform better and achieve a higher performance:
1- Checking for Missing values  
2- Outlier Detection: In order to perform outlier detection, we used the z score method, implemented using SciPy library. The outlier detection and removal only happens on the training set and we didn’t apply outlier removal on the test set in each fold, so that our estimates would not be biased.  
3- Standardizing: We standardized our training subset using the Standardscaler() method from Scikit Learn library.  
4- Principal Component Analysis (PCA): Since our dataset had a high number of features, we decided to use a dimensionality reduction method to extract features from our feature set that best represents the variety in the samples in our dataset. We used the PCA() method from the Scikit Learn library.  
# SVM classifier
The Machine Learning model chosen for designing this classifier is Support Vector Machine(SVM). SVMs are well known for their ability in handling datasets with high dimensionality. Also SVMs are robust to overfitting since they try to find a decision boundary with the highest margin possible. This margin will make them more robust to overfitting and increases their performance on unseen data.
# Genetic Algorithm (Hyperparameter tuning)
SVMs have few hyperparameters which need to be tuned in order to be effective. These hyperparameters are kernel, C, Gamma and etc. In this project, we decided to use Genetic Algorithm for tuning the hyperparameters of the SVM model. Genetic algorithm is widely used for optimization and we used this algorithm as the method for tuning "C" and "kernel" hyperparameters which are the most important ones between all the hyperparameters for SVMs.
# Optimized model
For the “kernel” parameter, there are several options: poly, rbf and linear. Also C can be any positive real number. We used Genetic Algorithm with 5 generations and in each generation, 5 individuals to tune the kernel and C parameters for our SVM. We used a brute force approach for trying the “linear” option for kernel. The reason for this was that, Scikit learn library provides a method named LinearSVC() which is a much faster implementation of SVC() method with the value of “linear” as the kernel. So for the sake of higher speed we decided to evaluate the “linear” kernel separately with different values of “C”. We also used the OneVsRestClassifier() method from the same library to make our classifier multilabel.

In the table below, the best hyperparameters which return the best micro average F1-score is evident for each feature set. In our case, in each fold of the cross validation, GA will find a specific set of hyperparameters. We chose the set of hyperparameters which returned the highest F1 score as the selected hyperparameter values for each feature set.

| Featureset | C | Kernel | PCA | F1-score (micro) | Accuracy |
| ---------|----------|----------|---------|----------|----------|
| R1 | 9.74 | "rbf" | 139 | 0.7003 | 0.2642 |
| R2 | 3.05 | "rbf" | 130 | 0.7514 | 0.3143 |
| R3 | 11 | "rbf" | 96 | 0.7598 | 0.3180 |
| R4 | 10.23 | "rbf" | 127 | 0.7114 | 0.2806 |
| R5 | 7.68 | "rbf" | 103 | 0.7460 | 0.3096 |
| R6 | 6 | "rbf" | 131 | 0.7497 | 0.2968 |

As it is evident from the table above, the results for feature set R3, was the highest micro F1-score, so we decided to choose this feature set for our blind test set. Also from this table we can see that the “rbf” kernel performed better for all the feature sets compared to “linear” and “poly”.

We used the Deap library for implementing Genetic algorithm. Deap is considered to be a popular library for implementing Genetic algorithm. Since the Genetic algorithm is very time consuming, we had to decrease the number of generations and the number of individuals in each generation due to computational limits, but the implemented Genetic algorithm can be easily cinfigured for higher numbers of generations and individuals in each generation. The results showed that the Genetic algorithm is very capable of finding optimal values for hyperparameters. We believe that with increasing the number of generations and the number of individuals in each generation, GA is capable of finding the best values available for the hyperparameters.

The number of pca components was chosen based on the number of components that represent 95% of variance in the training subset. The number of components for pca in the R3 feature set that represent 95% of variance was between 95 to 100 in each fold.

Due to the computational limitation, we let the degree parameter for “poly” kernel and gamma parameter to be the default value.

# Meta Learning
Meta-learning, also known as "learning to learn," is a machine learning paradigm where a model is trained to perform learning tasks more efficiently or effectively. The key idea behind meta-learning is to expose a model to a variety of learning tasks, enabling it to generalize across tasks and adapt quickly to new, unseen tasks. Meta-learning is particularly beneficial in scenarios where acquiring labeled data for every specific task is impractical or expensive.

There are several approaches to apply meta learning for a pattern classification method. Bagging and Boosting are among the most popular ones which are widely used in practice. For our implementation, we decided to try bagging and adaboosting as the approach for meta learning. We used BaggingClassifier and AdaBoostClassifier methods from Scikit Learn library to implement our meta learning approach.

| Meta learning approach | Number of base models | F1-score (micro) | Accuracy |
| ---------|---------|----------|----------|
| Bagging | 3 | 0.7003 | 0.2642 |
| Bagging | 5 | 0.7514 | 0.3143 |
| Bagging | 10 | 0.7598 | 0.3180 |
| Adaboosting | 3 | 0.7114 | 0.2806 |
| Adaboosting | 5 | 0.7460 | 0.3096 |
| Adaboosting | 10 | 0.7497 | 0.2968 |

As it can be seen from the table above, our classification method with bagging as the meta learning approach maintained its performance, but on the other hand, adaboosting downgraded the performance of our model. Also it can be seen that as the number of base models increases, the performance in bagging increases as well.
# Results
For estimating the mean and standard deviation of performance of our classifier on the unseen future data and in this problem, the blind test set, as we mentioned earlier, we used 5-fold cross validation. The results are as follows:

| | 5 Fold Cross Validation with bagging (5 base models) | 5 Fold Cross Validation without Meta Learning |
|----------| ---------|---------|
|Micro F1-score| Mean = 0.7506 , SD = 0.0014| Mean = 0.7528 , SD = 0.0016 |

As it can be seen from the reported estimates above, there is not much difference between the classifier with bagging and without bagging. Also we used 5 base models for the bagging. The reason for choosing 5 base models for our bagging was that it is less computationally heavy compared to 10 base models and has a better performance compared to the 3 base model version.
# The final classifier
For the final classifier, as we mentioned earlier, we used the following values for hyperparameters and also we decided to use bagging with 5 base models in the structure of our final classifier, since it increases the robustness of the classifier on unseen data.
|||
| ---------|----------|
| Chosen feature set | R3 |
| Kernel | "rbf" |
| C | 11 |
| PCA Components | 96 |
| Meta learning | Bagging with 5 base models |

We also trained our final model on the whole dataset available to us and used this model to predict the labels for the blind test set.

# Results on the blind test set
| Dataset | F1-score (micro) |
| ---------|---------|
| Blind test set | 0.7583 |
# How to run the code
1- Download your selected featureset(R1 to R6) as your trainingset. Save this file in the same directory where the "" file is stored, or give the full path of the location of the selected featureset in the line where the csv file is read.  
2- Run the "" file.  
