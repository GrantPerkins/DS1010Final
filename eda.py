import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def make_summary_table(df):
    df.describe().round(3).to_csv("tables/summary_stats.csv")


def make_corr_table(df):
    plt.figure(figsize=(10, 10))
    plt.matshow(df.corr(), cmap="RdYlGn")
    formatted_columns = [i.replace(" ", "\n") for i in df.columns]
    fontsize = 5.5
    plt.xticks(range(len(df.columns)), formatted_columns, fontsize=fontsize)
    plt.yticks(range(len(df.columns)), formatted_columns, fontsize=fontsize)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=fontsize)
    plt.show()
    # plt.savefig("plots/corr.png")
    # plt.close()
    df.corr().to_csv("tables/corr.csv")


if __name__ == "__main__":
    diabetes_df = pd.read_csv("diabetes.csv")
    # make_summary_table(diabetes_df)
    make_corr_table(diabetes_df)
