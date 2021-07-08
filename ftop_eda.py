import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.expand_frame_repr', False) #to show all the columns in the console

#infos about the dataset: https://www.kaggle.com/deepankurk/flight-take-off-data-jfk-airport
df = pd.read_csv('dataset/M1_final.csv')
print(df.head())
print(df.shape)
print(df.columns)

print(df.info()) #I notice that the column "Wind" has 2 null samples

df.describe(include=object)

df['Wind'].value_counts()