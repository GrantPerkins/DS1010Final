"""
EDA.py by Grant Perkins, for the DS 1010 group 2 final project. 2022.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
import seaborn as sns


def make_summary_table(df):
    """
    Calculate summary statistics for any DF
    :param df: input df
    :return: summary statistics written to file. Nothing returned.
    """
    df.describe().round(3).to_csv("tables/summary_stats_normal.csv")


def make_corr_table(df, title):
    """
    Creates a correlation matrix for a given DF. does not ignore catergorical variables.
    :param df: input df
    :param title: title for the output plot
    :return: correlation matrix written to file
    """
    plt.subplots(layout="constrained")
    sns.heatmap(df.corr(), cmap="RdYlGn", annot=True, fmt='.1g', vmin=0, vmax=1)
    plt.title(title)
    plt.show()

    df.corr().round(3).to_csv("tables/corr.csv")


def normalize_data(df):
    """
    Normalize each column for easier prediction
    :param df: input df
    :return: standardized df
    """
    df = (df - df.mean()) / df.std()
    df.to_csv("tables/normalized.csv")
    return df


def filter_data(df):
    """
    Remove rows with impossible values. Namely, if BP, BMI, or skin thickness is 0, then that human would be dead.
    :param df: input df
    :return: filtered df
    """
    df = df[(df["Blood Pressure"] != 0) & (df["BMI"] != 0) & (df["Skin Thickness"] != 0)]
    return df


def svm(df):
    """
    Train an SVM classifier on the input df. creates feature importance plot and classification report.
    :param df: input df
    :return: feature importance plot and classification report
    """
    svc = SVC(kernel="linear")
    features = ["Pregnancies", "Glucose", "Blood Pressure", "Skin Thickness", "Insulin", "BMI", "Age"]
    X = df[features].to_numpy()
    y = df[["Outcome"]].to_numpy().astype(np.uint8)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)
    features = [i.replace(' ', '\n') for i in features]
    svc.fit(X_train, y_train)
    importances = svc.coef_[0]
    imp, names = zip(*sorted(zip(importances, features)))
    plt.barh(range(len(names)), imp, align='center')
    plt.yticks(range(len(names)), names)
    plt.gca().tick_params(labelsize=8)
    plt.title("SVM Classifier Feature Importances")
    plt.show()

    y_pred = svc.predict(X_test)
    report = pd.DataFrame(metrics.classification_report(y_test, y_pred, output_dict=True))
    report.round(3).to_csv("tables/svm_report.csv")


def label_distribution(df):
    """
    Creates a bar plot showing distribution of 1's and 0's
    :param df: input df
    :return: plot of distributions
    """
    from collections import Counter
    x = ['Yes (has diabetes)' if i else 'No' for i in df["Outcome"].to_numpy()]
    c = Counter(x)
    bars = plt.bar(c.keys(), c.values())
    plt.gca().bar_label(bars)
    plt.title("Distribution of Dataset Labels")
    plt.ylabel("Frequency")
    plt.show()


if __name__ == "__main__":
    normalized_filtered_df = pd.read_csv("tables/normalized_filtered_diabetes_df.csv")
    make_summary_table(normalized_filtered_df)
    label_distribution(normalized_filtered_df)
    make_corr_table(normalized_filtered_df, "Correlation Table")
    svm(normalized_filtered_df)
    # make_corr_table(normalized_filtered_df, "Variable Correlation Matrix")
    # diabetes_df = pd.read_csv("tables/filtered_diabetes_df.csv")
    # make_summary_table(diabetes_df)
    # label_distribution(diabetes_df)
