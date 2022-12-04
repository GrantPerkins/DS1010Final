import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def make_summary_table(df):
    df.describe().round(3).to_csv("tables/summary_stats.csv")


if __name__ == "__main__":
    diabetes_df = pd.read_csv("diabetes.csv")
    make_summary_table(diabetes_df)
