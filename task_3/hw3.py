# # Submitters:
# # Tal Rodgold & Binyamin Mor


from enum import Enum
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plotlib
from plotly.graph_objects import Figure


class Classifiers(Enum):
    """
    Enum class for all the classifiers from the sklearn library.
    """
    RandomForestClassifier = RandomForestClassifier
    DecisionTreeClassifier = DecisionTreeClassifier
    KNeighborsClassifier = KNeighborsClassifier
    AdaBoostClassifier = AdaBoostClassifier
    LogisticRegression = LogisticRegression
    GaussianNB = GaussianNB
    SVC = SVC


class TITANIC(Enum):
    """
    Enum class for all the strings related to the titanic.csv file
    """
    FilePath = r"/Users/talr/PycharmProjects/temporary/task_3/titanic_preprocessed_HW3.csv"
    Survived = 'Survived'
    DecisionFunction = 'decision_function'
    Scores = ['accuracy', 'f1', 'recall', 'precision', 'roc_auc']
    FP = 'False Positive'
    TP = 'True Positive'
    ROC = 'ROC'


def compute_evaluation_scores(y_test: pd.Series, y_pred: pd.Series) -> list:
    """
    This function will compute a list of evaluation scores for given prediction.
    :param y_test: The actual labels.
    :param y_pred: The predicted labels.
    :return: A list of evaluation scores.
    """
    evaluation_scores = [
        accuracy_score(y_test, y_pred),
        f1_score(y_test, y_pred),
        recall_score(y_test, y_pred),
        precision_score(y_test, y_pred),
        roc_auc_score(y_test, y_pred)
    ]
    return evaluation_scores


def train_and_evaluate_classifier(classifier, x_train: pd.DataFrame, x_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> tuple:
    """
    This function trains the classifier, predicts on test data, and computes evaluation scores.
    :param classifier: The classifier object.
    :param x_train: The training features.
    :param x_test: The testing features.
    :param y_train: The training labels.
    :param y_test: The testing labels.
    :return: A list of evaluation scores.
    """
    model = classifier.value()
    model.random_state = 1
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    evaluation_scores = compute_evaluation_scores(y_test, prediction)
    return model, evaluation_scores


def plotly_evaluation_scores(score_classifiers: dict):
    """
    This function Plots evaluation scores for each classifier.
    :param score_classifiers:  A dictionary containing evaluation scores for each classifier.
    """
    for index, score_name in enumerate(TITANIC.Scores.value):
        figure = Figure()
        for classifier_name, score in score_classifiers.items():
            name = f"{score_name} - {classifier_name}"
            score_value = score[index]
            figure.add_bar(name=name, x=[classifier_name], y=[score_value], text=str(round(score_value, 3)))
        figure.update(layout_title_text=score_name)
        figure.write_image(f"{score_name}.png", format="png", engine="kaleido")


def plotly_roc(decision_function_values: dict, y_test: pd.Series):
    """
    This function plots combined ROC curves for all classifiers.
    :param decision_function_values:  A dictionary containing decision function values for each classifier.
    :param y_test: The testing labels.
    """
    for cla_name, proof in decision_function_values.items():
        fpr, tpr, _ = roc_curve(y_test, proof)
        roc_auc = auc(fpr, tpr)
        plotlib.plot(fpr, tpr, label=f'{cla_name}, AUC = {roc_auc:.2f}')

    plotlib.plot([0, 1], [0, 1], lw=2)
    plotlib.legend(loc='lower right')
    plotlib.xlabel(TITANIC.FP.value)
    plotlib.ylabel(TITANIC.TP.value)
    plotlib.savefig(f"{TITANIC.ROC.value}.png")
    plotlib.show()


def main():
    # open and read data from file
    df = pd.read_csv(TITANIC.FilePath.value)

    # split data to training and testing sets
    target = df[TITANIC.Survived.value]
    data = df.drop(TITANIC.Survived.value, axis=1)
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.1, random_state=1)

    # set dicts
    df_values_dict = dict()
    score_classifiers_dict = dict()

    # iterate all classifiers
    for classifier in Classifiers:
        model, evaluation_scores = train_and_evaluate_classifier(classifier, x_train, x_test, y_train, y_test)
        score_classifiers_dict[classifier.name] = evaluation_scores
        if hasattr(model, TITANIC.DecisionFunction.value):
            proof = model.decision_function(x_test)
        else:
            proof = model.predict_proba(x_test)[:, 1]
        df_values_dict[classifier.name] = proof
        print(f"{classifier.name}: {evaluation_scores}")

    # creat graphs
    plotly_evaluation_scores(score_classifiers_dict)
    plotly_roc(df_values_dict, y_test)


if __name__ == "__main__":
    main()
