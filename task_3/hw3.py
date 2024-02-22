# Submitters:
# Tal Rodgold & Binyamin Mor

import pandas as pd
from enum import Enum
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


class TITANIC(Enum):
    """
    Enum class for all the strings related to the titanic.csv file
    """
    Survived = 'Survived'
    Classifiers = [RandomForestClassifier,
                   AdaBoostClassifier,
                   LogisticRegression,
                   KNeighborsClassifier,
                   DecisionTreeClassifier,
                   GaussianNB,
                   SVC]


def get_data(file_path: str) -> (pd.DataFrame, pd.DataFrame):
    """
    This function gets a file path to the cve file and returns the target and train data
    :param file_path: path to file
    :return: data set after spliting to target and train
    """
    df = pd.read_csv(file_path)    # read and open csv file into a DataFrame
    target = df[TITANIC.Survived.value]
    train = df.drop(TITANIC.Survived.value, axis=1)
    return target, train

def train_classifier():
    #TODO: fill
def print_classifier():
    #TODO: fill
def evaluate_classifier():
    #TODO: fill


def main():
    # File paths for input and output
    input_file_path = 'titanic_preprocessed HW3.csv'
    target, train = get_data(input_file_path)
    x_train, x_test, y_train, y_test = train_test_split(train, target, test_size=0.1, random_state=1)

    for cla in TITANIC.Classifiers.value:
        train_classifier()
        print_classifier()
        evaluate_classifier()

if __name__ == '__main__':
    main()