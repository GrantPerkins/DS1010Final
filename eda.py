import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
import seaborn as sns


def make_summary_table(df):
    df.describe().round(3).to_csv("tables/summary_stats_normal.csv")


def make_corr_table(df, title):
    # plt.figure(figsize=(10, 10))
    # plt.matshow(df.corr(), cmap="RdYlGn")
    # plt.suptitle(title)
    # formatted_columns = [i.replace(" ", "\n") for i in df.columns]
    # fontsize = 5.5
    # plt.xticks(range(len(df.columns)), formatted_columns, fontsize=fontsize)
    # plt.yticks(range(len(df.columns)), formatted_columns, fontsize=fontsize)
    # cb = plt.colorbar()
    # cb.ax.tick_params(labelsize=fontsize)
    # plt.show()
    # # plt.savefig("plots/corr.png")
    # # plt.close()
    plt.subplots(layout="constrained")
    sns.heatmap(df.corr(), cmap="RdYlGn", annot=True, fmt='.1g', vmin=0, vmax=1)
    plt.title(title)
    # plt.gca().tick_params(labelsize=6)
    # plt.margins(x=100, y=100)
    plt.show()

    df.corr().round(3).to_csv("tables/corr.csv")


def normalize_data(df):
    df = (df - df.mean()) / df.std()
    df.to_csv("tables/normalized.csv")
    return df


def filter_data(df):
    df = df[(df["Blood Pressure"] != 0) & (df["BMI"] != 0) & (df["Skin Thickness"] != 0)]
    return df


def svm(df):
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
    from collections import Counter
    x= ['Yes (has diabetes)' if i else 'No' for i in df["Outcome"].to_numpy()]
    c = Counter(x)
    bars = plt.bar(c.keys(), c.values())
    plt.gca().bar_label(bars)
    plt.title("Distribution of Dataset Labels")
    plt.ylabel("Frequency")
    plt.show()

def box_plots(df):
    cols = ["Pregnancies", "Glucose", "Blood Pressure", "Skin Thickness", "Insulin", "BMI", "Age"]
    fig, ax = plt.subplots()

if __name__ == "__main__":
    # normalized_filtered_df = pd.read_csv("tables/normalized_filtered_diabetes_df.csv")
    # # make_summary_table(normalized_filtered_df)
    # # svm(normalized_filtered_df)
    # make_corr_table(normalized_filtered_df, "Variable Correlation Matrix")
    diabetes_df = pd.read_csv("tables/filtered_diabetes_df.csv")
    # make_summary_table(diabetes_df)
    label_distribution(diabetes_df)
