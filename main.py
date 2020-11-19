# coding: utf-8
import pickle
from collections import OrderedDict
import graphviz
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedShuffleSplit, cross_val_predict, RandomizedSearchCV, validation_curve, GridSearchCV
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report, confusion_matrix, explained_variance_score, accuracy_score
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, VotingClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn import svm
import data_tool
import matplotlib.pyplot as plt


def train_model_and_store_count_hotkeys(path_model: str):
    # Get the dataframe
    data = data_tool.train_csv_to_data_matrix("starcraft-2-player-prediction-challenge-2020/TRAIN.CSV")
    df = data_tool.get_counts_hotkeys(data)

    X = df.drop('id', axis=1)
    y = df['id']
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    classifier = tree.DecisionTreeClassifier()
    classifier.fit(X, y)

    """
    y_pred = classifier.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    """
    with open(path_model, 'wb') as stored_model:
        pickle.dump(classifier, stored_model, pickle.HIGHEST_PROTOCOL)


def train_model_and_store_all_count_actions(path_model: str):
    # Get the dataframe
    data = data_tool.train_csv_to_data_matrix("starcraft-2-player-prediction-challenge-2020/TRAIN.CSV")

    df = data_tool.get_counts_all_actions(data, train=True)

    X = df.drop('id', axis=1)
    y = df['id']
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    classifier = tree.DecisionTreeClassifier()
    classifier.fit(X, y)

    with open(path_model, 'wb') as stored_model:
        pickle.dump(classifier, stored_model, pickle.HIGHEST_PROTOCOL)


def train_model_and_store_all_count_actions_timed(path_model: str):
    # Get the dataframe
    data = data_tool.train_csv_to_data_matrix("starcraft-2-player-prediction-challenge-2020/TRAIN.CSV")

    df = data_tool.get_all_counts_actions_timed(data, train=True)
    print("Df terminated; Training Model ...")
    X = df.drop('id', axis=1)
    y = df['id']
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    classifier = tree.DecisionTreeClassifier()
    classifier.fit(X, y)

    with open(path_model, 'wb') as stored_model:
        pickle.dump(classifier, stored_model, pickle.HIGHEST_PROTOCOL)


def train_and_store_model_all_count_linear_regression(path_model: str):
    # Get the dataframe
    data = data_tool.train_csv_to_data_matrix("starcraft-2-player-prediction-challenge-2020/TRAIN.CSV")

    df = data_tool.get_counts_all_actions(data, train=True)

    X = df.drop('id', axis=1)
    y = df['id']
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    classifier = LogisticRegression()
    classifier.fit(X, y)

    with open(path_model, 'wb') as stored_model:
        pickle.dump(classifier, stored_model, pickle.HIGHEST_PROTOCOL)


def train_and_store_model_all_count_linear_regression_no_pandas(path_file_train: str, path_model: str):

    print("Read csv ...")
    data = data_tool.read_csv(path_file_train)
    print("Get Features ...")
    features = data_tool.get_counts_all_actions_no_pandas(data, train=True)
    # features = data_tool.get_counts_timed_no_pandas(data, train=True)
    print("Train model...")
    y = []
    for elem in features:
        y.append(elem.pop(0))

    classifier = RandomForestClassifier()
    # classifier = LogisticRegression(max_iter=5000, solver="newton-cg")
    classifier.fit(features, y)

    with open(path_model, 'wb') as stored_model:
        pickle.dump(classifier, stored_model, pickle.HIGHEST_PROTOCOL)


def train_and_store_model_freq_forest(path_file_train: str, path_model: str):

    print("Read csv ...")
    data = data_tool.read_csv(path_file_train)
    print("Get Features ...")
    features = data_tool.get_freq_all_actions_no_pandas(data, train=True)
    print("Train model...")
    y = []
    for elem in features:
        y.append(elem.pop(0))

    """
    classifier = RandomForestClassifier()
    classifier.fit(features, y)
    """
    # classifier = RandomForestClassifier()
    classifier = ExtraTreesClassifier(n_estimators=2000)
    classifier.fit(features, y)
    # scores = cross_val_score(classifier, features, y, cv=5)
    scores = StratifiedShuffleSplit(n_splits=5, test_size=0.2)
    print(scores)
    with open(path_model, 'wb') as stored_model:
        pickle.dump(classifier, stored_model, pickle.HIGHEST_PROTOCOL)


def train_and_store_model_all_count_linear_regression_tf_idf(path_model: str):
    # Get the dataframe
    data = data_tool.train_csv_to_data_matrix("starcraft-2-player-prediction-challenge-2020/TRAIN.CSV")

    df = data_tool.get_counts_all_actions(data, train=True)

    X = df.drop('id', axis=1)

    tfidf_transformer = TfidfTransformer(use_idf=True)
    X_tf_idf = tfidf_transformer.fit_transform(X)

    y = df['id']
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    classifier = LogisticRegression()
    classifier.fit(X_tf_idf, y)

    with open(path_model, 'wb') as stored_model:
        pickle.dump(classifier, stored_model, pickle.HIGHEST_PROTOCOL)


def get_trained_model(path_model):
    with open(path_model, 'rb') as stored_model:
        classifier = pickle.load(stored_model)
        return classifier


def get_png_of_decision_tree(model, name):
    dot_data = tree.export_graphviz(model, out_file=None)
    graph = graphviz.Source(dot_data)
    graph.render(name)
    # Then terminal : dot -Tpng "name" -o "name".png



def evaluate_model(path_file_train):
    print("Read csv ...")
    data = data_tool.read_csv(path_file_train)
    print("Get Features ...")
    X = data_tool.get_features(data, train=True)
    # X = data_tool.get_freq_all_actions_no_pandas(data, train=True)
    # X = data_tool.less_dimensions_counts(data, train=True)
    # X = data_tool.get_counts_timed_no_pandas(data, train=True)
    print("Testing model...")
    y = []
    for elem in X:
        y.append(elem.pop(0))

    """
    #   One classifier for each race
    Protoss = 0
    Zerg = 1
    Terran = 2

    terran_games = []
    protoss_games = []
    zerg_games = []

    y_terran_games = []
    y_protoss_games = []
    y_zerg_games = []

    for index, values in enumerate(X):
        if values[0] == Zerg:
            zerg_games.append(values)
            y_zerg_games.append(y[index])
        elif values[0] == Protoss:
            protoss_games.append(values)
            y_protoss_games.append(y[index])
        elif values[0] == Terran:
            terran_games.append(values)
            y_terran_games.append(y[index])

    classifier_terran = RandomForestClassifier()
    classifier_zerg = RandomForestClassifier()
    classifier_protoss = RandomForestClassifier

    # Terran score
    scores = cross_val_score(classifier_terran, terran_games, y_terran_games, cv=5)
    avg = 0
    for s in scores:
        avg += s
    print(scores)
    print(f"Average Terran : {avg/len(scores)}")
    # Zerg score
    scores = cross_val_score(classifier_zerg, zerg_games, y_zerg_games, cv=5)
    avg = 0
    for s in scores:
        avg += s
    print(scores)
    print(f"Average Zerg : {avg/len(scores)}")
    # Protoss score
    scores = cross_val_score(classifier_terran, terran_games, y_terran_games, cv=5)
    avg = 0
    for s in scores:
        avg += s
    print(scores)
    print(f"Average Protoss : {avg/len(scores)}")
    """
    """
    ids = data_tool.get_numbers_ids_reference(path_file_train)
    for index, player in enumerate(y):
        y[index] = ids[player]
    """
    # print(y)
    # classifier = RandomForestRegressor()

    # Create the random grid
    n_estimators = [10, 100, 300, 500, 700]
    max_features = ['auto', 'sqrt']
    bootstrap = [True, False]
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]
    criterion = ['gini', 'entropy'] 
    random_grid = {
        'n_estimators': n_estimators,
        'max_features': max_features,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'bootstrap': bootstrap,
        'criterion': criterion,
    }
    # Find the best parameters
    classifier = RandomForestClassifier(
        n_estimators= 300,
        min_samples_split= 2,
        min_samples_leaf= 1,
        max_features= 'sqrt',
        max_depth= 50,
        criterion= 'entropy',
        bootstrap= False
    )
    classifier_random = RandomizedSearchCV(estimator=classifier, param_distributions=random_grid, n_iter=50, cv=3, verbose=2, random_state=42, n_jobs=-1)
    classifier_random.fit(X, y)
    print(classifier_random.best_params_)
    # Test
    best_classifier = classifier_random.best_estimator_
    print("Read csv test data...")
    data = data_tool.read_csv("starcraft-2-player-prediction-challenge-2020/TEST.CSV")
    print("Get Features test data ...")
    features = data_tool.get_features(data, train=False)

    print("Evaluate results...")
    with open("Results/test.csv", 'w') as sub_file:
        sub_file.write("RowId,prediction\n")
        y_pred = best_classifier.predict(features)
        print("Store results...")
        for i, res in enumerate(y_pred):
            sub_file.write(f"{i+1},{res}\n") 
    # classifier = GradientBoostingClassifier()
    # classifier = RandomForestClassifier(bootstrap=False)
    """
    scores = cross_val_score(classifier, X, y, cv=5)
    avg = 0
    for s in scores:
        avg += s
    print(scores)
    print(f"Average : {avg/len(scores)}")
    """
    """
    X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.20)

    # classifier = LogisticRegressionCV(multi_class='multinomial')
    classifier = RandomForestClassifier(n_estimators=611)
    # classifier = GaussianNB()
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    # print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    # print(explained_variance_score(y_test, y_pred))
    """


def make_and_save_predictions(X_test, classifier, namefile: str):
    with open(namefile, 'w') as sub_file:
        sub_file.write("RowId,prediction\n")

        y_pred = classifier.predict(X_test)
        for i, res in enumerate(y_pred):
            sub_file.write(f"{i+1},{res}\n")


def make_and_save_predictions_no_pandas(path_to_test: str, path_classifier: str, namefile: str):
    print("Read csv test data...")
    data = data_tool.read_csv(path_to_test)
    print("Get Features test data ...")
    features = data_tool.get_freq_all_actions_no_pandas(data, train=False)
    # features = data_tool.get_counts_all_actions_no_pandas(data, train=False)
    print("Get classifier ...")
    classifier = get_trained_model(path_classifier)

    print("Evaluate results...")
    with open(namefile, 'w') as sub_file:
        sub_file.write("RowId,prediction\n")

        y_pred = classifier.predict(features)
        print("Store results...")
        for i, res in enumerate(y_pred):
            sub_file.write(f"{i+1},{res}\n")


def train_and_make_predictions(path_file_train: str, path_file_test: str, namefile: str):
    print("Read csv ...")
    data = data_tool.read_csv(path_file_train)
    print("Get Features ...")
    X = data_tool.get_features(data, train=True)
    print("Train model...")
    y = []
    for elem in X:
        y.append(elem.pop(0))

    # classifier = RandomForestClassifier()
    # classifier = RandomForestClassifier(class_weight="balanced")

    classifier = RandomForestClassifier(
        n_estimators= 300,
        min_samples_split= 2,
        min_samples_leaf= 1,
        max_features= 'sqrt',
        max_depth= 50,
        criterion= 'entropy',
        bootstrap= True,
        oob_score=True
    )
    classifier.fit(X, y)

    print("Read csv test data...")
    data = data_tool.read_csv(path_file_test)
    print("Get Features test data ...")
    features = data_tool.get_features(data, train=False)

    print("Evaluate results...")
    with open(namefile, 'w') as sub_file:
        sub_file.write("RowId,prediction\n")
        y_pred = classifier.predict(features)
        print("Store results...")
        for i, res in enumerate(y_pred):
            sub_file.write(f"{i+1},{res}\n")


def train_and_make_predictions_three_classifiers(path_file_train: str, path_file_test: str, namefile: str):
    print("Read csv ...")
    data = data_tool.read_csv(path_file_train)
    print("Get Features ...")
    X = data_tool.get_features(data, train=True)
    print("Train model...")
    y = []
    for elem in X:
        y.append(elem.pop(0))

    # One classifier for each race
    Protoss = 0
    Zerg = 1
    Terran = 2

    terran_games = []
    protoss_games = []
    zerg_games = []

    y_terran_games = []
    y_protoss_games = []
    y_zerg_games = []

    for index, values in enumerate(X):
        if values[0] == Zerg:
            zerg_games.append(values)
            y_zerg_games.append(y[index])
        elif values[0] == Protoss:
            protoss_games.append(values)
            y_protoss_games.append(y[index])
        elif values[0] == Terran:
            terran_games.append(values)
            y_terran_games.append(y[index])

    classifier_terran = RandomForestClassifier()
    classifier_zerg = RandomForestClassifier()
    classifier_protoss = RandomForestClassifier()

    classifier_terran.fit(terran_games, y_terran_games) 
    classifier_protoss.fit(protoss_games, y_protoss_games)
    classifier_zerg.fit(zerg_games, y_zerg_games)

    # classifier.fit(X, y)

    print("Read csv test data...")
    data = data_tool.read_csv(path_file_test)
    print("Get Features test data ...")
    features = data_tool.get_features(data, train=False)

    print("Evaluate results...")
    y_pred = []
    with open(namefile, 'w') as sub_file:
        sub_file.write("RowId,prediction\n")
        for game in features:
            print(game)
            if game[0] == Terran:
                pred = classifier_terran.predict([game])
                print(pred)
            elif game[0] == Zerg:
                pred = classifier_zerg.predict([game])
                print(pred)
            elif game[0] == Protoss:
                pred = classifier_protoss.predict([game])
                print(pred)
            
            y_pred.append(pred[0])
        # y_pred = classifier.predict(features)
        print("Store results...")
        for i, res in enumerate(y_pred):
            sub_file.write(f"{i+1},{res}\n")


def random_forest_experiements(path_file_train: str):
    print("Read csv ...")
    data = data_tool.read_csv(path_file_train)
    print("Get Features ...")
    X = data_tool.get_features(data, train=True)
    print("Testing model...")
    y = []
    for elem in X:
        y.append(elem.pop(0))

    for i in [0, 1, 2]:
        results_n_estimators_x = []
        results_n_estimators_y = []
        # 20 by 20 from 10 to 100 
        for i in range(10, 110, 10):
            classifier = RandomForestClassifier(n_estimators=i)
            score = cross_val_score(classifier, X, y, cv=5)
            print(f"Estimators = {i} : Average {score.mean()}")
            results_n_estimators_x.append(i)
            results_n_estimators_y.append(score.mean())
        
        # 100 by 100 from 100 to 1500
        for i in range(200, 1100, 100):
            classifier = RandomForestClassifier(n_estimators=i)
            score = cross_val_score(classifier, X, y, cv=5)
            print(f"Estimators = {i} : Average {score.mean()}")
            results_n_estimators_x.append(i)
            results_n_estimators_y.append(score.mean())

        # TODO Faire une fonction linéaire, pas un bar plot
        plt.plot(results_n_estimators_x, results_n_estimators_y)

    plt.show()


def evaluate_model_2(path_file_train: str):
    print("Read csv ...")
    data = data_tool.read_csv(path_file_train)
    print("Get Features ...")
    X = data_tool.get_features(data, train=True)

    print("Testing model...")
    y = []
    for elem in X:
        y.append(elem.pop(0))

    classifier = RandomForestClassifier()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=40) 


    # fit the model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    # make predictions
    y_pred = model.predict(X_test)
    # evaluate predictions
    acc = accuracy_score(y_test, y_pred)
    print('Accuracy standard params: %.3f' % acc)

    """
    # GridSearch
    param_grid = {
        'bootstrap': [True],
        'max_depth': [30, 40, 50, 60],
        'max_features': [20, 30, 40],
        'min_samples_leaf': [1, 2, 3, 4],
        'min_samples_split': [8, 10, 12],
        'n_estimators': [500, 600, 800, 1000]
    }
    rf = RandomForestClassifier()
    grid_search = GridSearchCV(
        estimator = rf,
        param_grid = param_grid,
        cv = 3,
        n_jobs = -1,
        verbose = 2
    )

    # => Result: {'bootstrap': True, 'max_depth': 60, 'max_features': 20, 'min_samples_leaf': 1, 'min_samples_split': 8, 'n_estimators': 600}
    grid_search.fit(X_train, y_train)
    print(grid_search.best_params_)

    best_grid = grid_search.best_estimator_
    """
    model2 = RandomForestClassifier(
        bootstrap=True,
        max_depth = 60,
        max_features = 20,
        min_samples_leaf = 1,
        min_samples_split = 8,
        n_estimators = 600
    )
    model2.fit(X_train, y_train)
    # make predictions
    y_pred = model2.predict(X_test)
    # evaluate predictions
    acc = accuracy_score(y_test, y_pred)
    print('Accuracy best params : %.3f' % acc)


    # En utilisant la méthode ci-dessous
    # n_estimators = 1000
    # max_depth = 40
    # max_features = 40
    """
    num_est = range(1, 60, 5)
    train_scoreNum, test_scoreNum = validation_curve(
                                RandomForestClassifier(n_estimators=500, max_depth=40, min_samples_leaf=1),
                                X = X_train, y = y_train, 
                                param_name = 'max_features', 
                                param_range = num_est, cv = 3)

    scores = classification_report(y_test, y_pred)
    print(scores)
    
    # Plot mean accuracy scores for training and test sets
    plt.plot(num_est, test_scoreNum, label="Cross-validation score", color="blue")

    # Create plot
    plt.title("Validation Curve With Random Forest")
    plt.xlabel("Number Of Trees")
    plt.ylabel("Accuracy Score")
    plt.tight_layout()
    plt.legend(loc="best")
    plt.show()
    """


if __name__ == "__main__":
    """
    train_and_make_predictions(
        "starcraft-2-player-prediction-challenge-2020/TRAIN.CSV",
        "starcraft-2-player-prediction-challenge-2020/TEST.CSV",
        "Results/30_SUBMISSION.CSV"
    )
    train_and_make_predictions_three_classifiers(
        "starcraft-2-player-prediction-challenge-2020/TRAIN.CSV",
        "starcraft-2-player-prediction-challenge-2020/TEST.CSV",
        "Results/27_SUBMISSION.CSV"
    )
    """
    # evaluate_model_2("starcraft-2-player-prediction-challenge-2020/TRAIN.CSV")
    # data_tool.get_informations_data("starcraft-2-player-prediction-challenge-2020/TRAIN.CSV")
    random_forest_experiements("starcraft-2-player-prediction-challenge-2020/TRAIN.CSV")
    print("Done")

