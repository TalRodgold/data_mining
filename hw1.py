# Submitters:
# Tal Rodgold & Binyamin Mor

from enum import Enum
import pandas as pd
from scipy.stats import pearsonr


class TRAIN(Enum):
    """
    Enum class for all the strings related to the train.csv file
    """
    PassengerId = 'PassengerId'
    Survived = 'Survived'
    Pclass = 'Pclass'
    Name = 'Name'
    Sex = 'Sex'
    Age = 'Age'
    SibSp = 'SibSp'
    Parch = 'Parch'
    Ticket = 'Ticket'
    Fare = 'Fare'
    Cabin = 'Cabin'
    Embarked = 'Embarked'


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function receives a data frame and verify that the input is valid
    :param df: The data set before validation
    :return: The data set after validation
    """
    # Verify PassengerId is an Integer ------------------------------------------------------------------------------
    for passengerid in df[TRAIN.PassengerId.value]:
        if type(passengerid) != int:
            mask = df[TRAIN.PassengerId.value] == passengerid
            df = df[~mask]  # Remove rows with the specified value in the specified column

    # Verify Age is a Float -----------------------------------------------------------------------------------------
    for age in df[TRAIN.Age.value]:
        if type(age) != float:
            mask = df[TRAIN.Age.value] == age
            df = df[~mask]  # Remove rows with the specified value in the specified column

    # Verify Survived is a 1 or 0 ------------------------------------------------------------------------------------
    for survive in df[TRAIN.Survived.value]:
        if survive not in [0, 1]:
            mask = df[TRAIN.Survived.value] == survive
            df = df[~mask]  # Remove rows with the specified value in the specified column

    # Verify Sex is a male or female ---------------------------------------------------------------------------------
    for sex in df[TRAIN.Sex.value]:
        if sex not in ['male', 'female']:
            mask = df[TRAIN.Sex.value] == sex
            df = df[~mask]  # Remove rows with the specified value in the specified column

    # Verify Pclass is a 1 or 2 or 3  --------------------------------------------------------------------------------
    for pclass in df[TRAIN.Pclass.value]:
        if pclass not in [1, 2, 3]:
            mask = df[TRAIN.Pclass.value] == pclass
            df = df[~mask]  # Remove rows with the specified value in the specified column

    # Verify Parch is an int  ----------------------------------------------------------------------------------------
    for parch in df[TRAIN.Parch.value]:
        if type(parch) != int:
            mask = df[TRAIN.Parch.value] == parch
            df = df[~mask]  # Remove rows with the specified value in the specified column

    # Verify Fare is a Float -----------------------------------------------------------------------------------------
    for fare in df[TRAIN.Fare.value]:
        if type(fare) != float:
            mask = df[TRAIN.Fare.value] == fare
            df = df[~mask]  # Remove rows with the specified value in the specified column

    # Verify Embarked is a C or S or Q -------------------------------------------------------------------------------
    for embarked in df[TRAIN.Embarked.value]:
        if embarked not in ['C', 'S', 'Q']:
            mask = df[TRAIN.Embarked.value] == embarked
            df = df[~mask]  # Remove rows with the specified value in the specified column

    return df


def nan_converter(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function receives a data frame and changes all the nan values to 0
    :param df: The data set with nan
    :return: The data set without nan
    """
    # Replace all NaN values with 0 in the entire DataFrame
    df.fillna(0, inplace=True)
    return df


def dummy_converter(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function converts specific columns to dummy values
    :param df: The data set before setting dummy values
    :return: The data set after setting dummy values
    """
    # dummy column Embarked: C = 1, S = 2, Q = 3 ---------------------------------------------------------------------
    df.loc[df[TRAIN.Embarked.value] == 'C', TRAIN.Embarked.value] = 1
    df.loc[df[TRAIN.Embarked.value] == 'S', TRAIN.Embarked.value] = 2
    df.loc[df[TRAIN.Embarked.value] == 'Q', TRAIN.Embarked.value] = 3

    # dummy column Sex: male = 1, female = 2  ------------------------------------------------------------------------
    df.loc[df[TRAIN.Sex.value] == 'male', TRAIN.Sex.value] = 1
    df.loc[df[TRAIN.Sex.value] == 'female', TRAIN.Sex.value] = 2

    return df


def calculate_correlation(column1: pd.Series, column2: pd.Series) -> float:
    """
    Calculate the Pearson correlation coefficient between two columns.
    :param column1: representing the values of the first variable.
    :param column2: representing the values of the second variable.
    :return: The absolut Pearson correlation coefficient between the two columns.
    """

    correlation_coefficient, _ = pearsonr(column1, column2)
    return abs(correlation_coefficient)


# Function to read CSV, perform data manipulation, and save to TXT
def manipulate_and_save(input_csv_path: str, output_txt_path: str):
    """
    This function opens a csv file, manipulate the data in it and then save it in a txt file
    :param input_csv_path: path to input file
    :param output_txt_path: path to output file
    :return: NONE
    """

    df = pd.read_csv(input_csv_path)    # read and open csv file into a DataFrame
    df = validate_data(df)  # validate data from file
    df = nan_converter(df)  # take care of nans
    df = dummy_converter(df)    # convert necessary columns to dummy

    # calculate correlation and delete necessary columns
    relevant_list = [TRAIN.Pclass.value, TRAIN.Sex.value, TRAIN.Age.value, TRAIN.SibSp.value,
                   TRAIN.Parch.value, TRAIN.Fare.value, TRAIN.Embarked.value]
    for column in relevant_list:
        value = calculate_correlation(df[TRAIN.Survived.value], df[column])
        if value < 0.015:
            df = df.drop(column, axis=1)
            relevant_list.remove(column)

    # calculate correlation between all columns
    for column1 in relevant_list:
        for column2 in relevant_list:
            if column1 != column2:
                value = calculate_correlation(df[column1], df[column2])
                if value > 0.98:
                    df = df.drop(column2, axis=1)
                    relevant_list.remove(column2)

    # save manipulated data to new file
    df.to_csv(output_txt_path, sep='\t', index=False)


def main():
    # File paths for input and output
    input_file_path = 'train.csv'
    output_file_path = 'hw1.txt'

    # Call the function to open csv file, manipulate the data and then save it in a txt file
    manipulate_and_save(input_file_path, output_file_path)


if __name__ == '__main__':
    main()

