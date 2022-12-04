import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("diabetes.csv")
df.describe().round(3).to_csv("tables/summary_stats.csv")
