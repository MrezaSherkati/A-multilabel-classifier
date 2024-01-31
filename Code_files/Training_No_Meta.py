import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import f1_score, accuracy_score, make_scorer
import random
from sklearn.multiclass import OneVsRestClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from deap import base, creator, tools, algorithms

# Reading the CSV files
X = pd.read_csv("R3_train.csv") # Change this line for reading another feature set
Y = pd.read_csv("labels_train.csv")
print('Done!')

# Check to see if we have any missing data in our features
missing_values = X.isnull().sum()
print("Missing values in our dataset: ", missing_values)

# Defining the specific type of our score metric
scorer = make_scorer(f1_score, average='micro')

# Defining the different folds for our k fold cross validation
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# Defining the dictionaries for saving our results
f1_list = [] # Saving micro F1-scores
accuracy_list = [] # Saving Accuracy Scores

# Loop through the folds
for train_index, test_index in kf.split(X, Y):

    # Assigning data to our train and test subsets in each fold
    X_train_before, X_test = X.iloc[train_index], X.iloc[test_index]
    Y_train_before, Y_test = Y.iloc[train_index], Y.iloc[test_index]

    # Splitting our training subset into training and validation subsets
    X_train, X_val, Y_train, Y_val = train_test_split(X_train_before, Y_train_before, test_size=0.2, random_state=42)

    # Perform outlier detection for the train subset(Z-score)
    z_scores = zscore(X_train)

    # Set a threshold for Z-Scores (4 standard deviations)
    threshold = 4
    outliers = (abs(z_scores) > threshold).any(axis=1)

    # Remove outliers from the training subset
    X_train_no_outliers = X_train[~outliers]
    Y_train_no_outliers = Y_train[~outliers]

    # Standardize the features for the training subset
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train_no_outliers)
    X_val_std = scaler.transform(X_val)
    ##################################################################################
    # Doing the experiment with PCA to see how many number of components is good
    pca = PCA()
    X_train_pca = pca.fit_transform(X_train_std)

    # Choosing a threshold for cumulative variance
    cumulative_variance_threshold = 0.95
    num_components_cumulative = np.argmax(np.cumsum(pca.explained_variance_ratio_) >= cumulative_variance_threshold) + 1
    print(f'Number of components for {cumulative_variance_threshold * 100}% cumulative variance: {num_components_cumulative}')

    # Applying the chosen number for pca transformation
    pca = PCA(n_components=num_components_cumulative)
    X_train_pca = pca.fit_transform(X_train_std)
    X_val_pca = pca.transform(X_val_std)
    ##################################################################################
    # Testing the linear SVC separately because it takes less time compared to when we put in the kernel options of Genetic Algorithm
    param_grid = {'estimator__C': [0.1, 1, 10, 100]}

    # Creating a LinearSVC model
    svc = LinearSVC()
    multilabel_classifier_brute = OneVsRestClassifier(svc, n_jobs=-1)

    # Creating a GridSearchCV object
    grid_search = GridSearchCV(estimator=multilabel_classifier_brute, param_grid=param_grid, cv=5, scoring=scorer)

    # Fitting the GridSearchCV object to the data
    grid_search.fit(X_train_pca, Y_train_no_outliers)

    # Getting the best parameter from the search
    best_C = grid_search.best_params_['estimator__C']
    print(f"Best C for linear case is: {best_C}")

    # Using the best C to train a LinearSVC model
    best_svc_linear = LinearSVC(C=best_C)
    multilabel_classifier_linear = OneVsRestClassifier(best_svc_linear, n_jobs=-1)
    multilabel_classifier_linear.fit(X_train_pca, Y_train_no_outliers)

    # Making predictions on the validation subset
    y_pred = multilabel_classifier_linear.predict(X_val_pca)

    # Evaluating the model
    f_score_linear = f1_score(Y_val, y_pred, average="micro")
    accuracy_linear = accuracy_score(Y_val, y_pred)
    print(f"F1 score for R3 with micro averaging is (Linear case with C = {best_C}): {f_score_linear: .4f}")
    print(f"Accuracy for R3 is (Linear case with C = {best_C}): {accuracy_linear: .4f}")

    # Define the hyperparameter search space
    hyperparameter_ranges = {
        'C': [0.1, 1, 10],
        # rbf =1 and poly = 2
        'kernel': [1, 2],
    }

    # Define individual and population
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.choice, hyperparameter_ranges['C'])
    toolbox.register("attr_kernel", random.choice, hyperparameter_ranges['kernel'])

    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.attr_float, toolbox.attr_kernel), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)


    # Define the fitness function
    def evaluate(individual):
        C = max(0.1, individual[0])
        if (individual[1] < 1.5):
            kernel = 'rbf'
        else:
            kernel = 'poly'
        clf = SVC(C=C, kernel=kernel)

        # Make SVM a Multilabel classifier
        multilabel_classifier = OneVsRestClassifier(clf, n_jobs=-1)
        multilabel_classifier.fit(X_train_pca, Y_train_no_outliers)
        predictions = multilabel_classifier.predict(X_val_pca)
        f_score = f1_score(Y_val, predictions, average="micro")
        return float(f_score),


    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Create the population
    population = toolbox.population(n=5)

    # Run the genetic algorithm
    algorithms.eaMuPlusLambda(population, toolbox, mu=5, lambda_=20, cxpb=0.7, mutpb=0.2, ngen=5, stats=None,
                              halloffame=None)

    # Get the best individual
    best_individual = tools.selBest(population, k=1)[0]
    # best_hyperparameters = [best_individual[0], best_individual[1]]
    print(f"Best Hyperparameters found are: C = {best_individual[0]} and kernel ={best_individual[1]}")

    # Train your model with the best hyperparameters
    C = max(0.1, best_individual[0])
    if (best_individual[1] < 1.5):
        kernel = 'rbf'
    else:
        kernel = 'poly'
    best_model = SVC(C=C, kernel=kernel)

    # Make SVM a Multilabel classifier
    multilabel_classifier_gen = OneVsRestClassifier(best_model, n_jobs=-1)
    multilabel_classifier_gen.fit(X_train_pca, Y_train_no_outliers)
    predictions_gen = multilabel_classifier_gen.predict(X_val_pca)
    f_score_gen = f1_score(Y_val, predictions_gen, average="micro")
    print(f"F1 score for R3 with micro averaging using Genetic is: {f_score_gen: .4f}")
    ##################################################################################
    # Choose between linear model and the model from genetic algorithm
    if f_score_gen > f_score_linear:
        # Training on the whole training subset available in each fold of CV
        print("Model from Genetic is better")
        z_scores = zscore(X_train_before)

        # Set a threshold for Z-Scores (4 standard deviations)
        threshold = 4
        outliers = (abs(z_scores) > threshold).any(axis=1)

        # Remove outliers from the training subset
        X_train_no_outliers = X_train_before[~outliers]
        Y_train_no_outliers = Y_train_before[~outliers]

        # Standardize the features for the training subset
        scaler = StandardScaler()
        X_train_std = scaler.fit_transform(X_train_no_outliers)
        X_test_std = scaler.transform(X_test)

        # PCA
        pca = PCA(n_components=num_components_cumulative)
        X_train_pca = pca.fit_transform(X_train_std)
        X_test_pca = pca.transform(X_test_std)

        # Defining the model
        best_model = SVC(C=C, kernel=kernel)
        multilabel_classifier_best = OneVsRestClassifier(best_model, n_jobs=-1)
        multilabel_classifier_best.fit(X_train_pca, Y_train_no_outliers)
        predictions_best = multilabel_classifier_best.predict(X_test_pca)
        f_score_best = f1_score(Y_test, predictions_best, average="micro")
        accuracy_best = accuracy_score(Y_test, predictions_best)
        print(f"F1 score for R3 with micro averaging is (Genetic is the chosen one): {f_score_best: .4f}")
        print(f"Accuracy for R3 with micro averaging is (Genetic is the chosen one): {accuracy_best: .4f}")
    else:
        # Training on the whole training subset available in each fold of CV
        print("Linear model is better")
        z_scores = zscore(X_train_before)

        # Set a threshold for Z-Scores (4 standard deviations)
        threshold = 4
        outliers = (abs(z_scores) > threshold).any(axis=1)

        # Remove outliers from the training subset
        X_train_no_outliers = X_train_before[~outliers]
        Y_train_no_outliers = Y_train_before[~outliers]

        # Standardize the features for the training subset
        scaler = StandardScaler()
        X_train_std = scaler.fit_transform(X_train_no_outliers)
        X_test_std = scaler.transform(X_test)

        # PCA
        pca = PCA(n_components=num_components_cumulative)
        X_train_pca = pca.fit_transform(X_train_std)
        X_test_pca = pca.transform(X_test_std)

        # Defining the model
        best_model = LinearSVC(C=best_C)
        multilabel_classifier_best = OneVsRestClassifier(best_model, n_jobs=-1)
        multilabel_classifier_best.fit(X_train_pca, Y_train_no_outliers)
        predictions_best = multilabel_classifier_best.predict(X_test_pca)
        f_score_best = f1_score(Y_test, predictions_best, average="micro")
        accuracy_best = accuracy_score(Y_test, predictions_best)
        print(f"F1 score for R3 with micro averaging is (Linear is the chosen one): {f_score_best: .4f}")
        print(f"Accuracy for R3 with micro averaging is (Linear is the chosen one): {accuracy_best: .4f}")

    # Appending the results to the dictionary
    f1_list.append(f_score_best)
    accuracy_list.append(accuracy_best)
    print("\n-------------------End of this fold--------------\n")

print(f"The mean of F1-score for this no meta learning model 5 fold CV is: {np.mean(f1_list):.4f}")
print(f"The standard deviation of F1-score for this not meta learning model with 5 fold CV is: {np.std(f1_list):.4f}")

print(f"The mean of accuracy for this no meta learning model with 5 fold CV is: {np.mean(accuracy_list):.4f}")
print(
    f"The standard deviation of accuracy for this no meta learning model with 5 fold CV is: {np.std(accuracy_list):.4f}")