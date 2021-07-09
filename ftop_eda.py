import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

pd.set_option('display.expand_frame_repr', False) #to show all the columns in the console
sns.set(style="white")

#infos about the dataset: https://www.kaggle.com/deepankurk/flight-take-off-data-jfk-airport
df = pd.read_csv('dataset/M1_final.csv')
print(df.head())
print(df.shape)
print(df.columns)

print(df.info()) #I notice that the column "Wind" has 2 null samples
df.isnull().sum() #double-check if some values is null
print('Wind missing values --> ', np.round(df['Wind'].isnull().mean(), 4),'%')

#I see that missing values are just 0.0001% so i decide to remove that instances
df.dropna(axis = 0, inplace = True)

df.describe(include=object)

#some queries on target variable
print(df['TAXI_OUT'].mean())
print(df['TAXI_OUT'])

airlines = df['OP_UNIQUE_CARRIER'].unique()
len(airlines)
counts = df['OP_UNIQUE_CARRIER'].value_counts()
counts.plot(kind = 'barh')
plt.show()
#VEDO CHE IL NUMERO DI FLIGHTS CHE ABBIAMO NON è BILANCIATISSIMO RISPETTO A QUALI COMPAGNIE CONSIDERIAMO

df['CRS_DEP_M'] = pd.to_datetime(df.CRS_DEP_M, unit = 'm').dt.strftime('%H:%M')
df['DEP_TIME_M'] = pd.to_datetime(df.DEP_TIME_M, unit = 'm').dt.strftime('%H:%M')
df['CRS_ELAPSED_TIME'] = pd.to_datetime(df.CRS_ELAPSED_TIME, unit = 'm').dt.strftime('%H:%M')
df['CRS_ARR_M'] = pd.to_datetime(df.CRS_ARR_M, unit = 'm').dt.strftime('%H:%M')
print(df.head())

df.drop('TAIL_NUM', axis = 1, inplace = True)
df.head()

df['YEAR'] = np.where(df['MONTH'] == 1, 2020, 2019) #we have only january, december and november so if month == january then year = 2020
df['date'] = pd.to_datetime(df.YEAR*10000+df.MONTH*100+df.DAY_OF_MONTH, format='%Y%m%d')
df.head()

y = df[['date', 'TAXI_OUT']].groupby('date').mean()
x = df.date.unique()

#a graph to see the target trend over time (with mean foe each day)
plt.plot_date(x = x, y = y, fmt = 'k--x')
plt.gcf().autofmt_xdate()
plt.show()
#non si vede alcun tipo di linearità


df['Dew Point'] = df['Dew Point'].str.strip()

df['Dew Point'] = df['Dew Point'].astype(int)
df.info()
print(df.dtypes)


#dummyzzazione con label encoder
dummy_df = df.copy()
dummy_encoder = LabelEncoder()
categorical = (dummy_df.dtypes == 'object')
categorical_labels = list(categorical[categorical].index)
del categorical_labels[2:6]
print(categorical_labels)
for column in categorical_labels:
    dummy_df[column] = dummy_encoder.fit_transform(df[column])
dummy_df.head()

sns.set()
#grafico in cui metto taxi out / dep_delay
y = df['TAXI_OUT'].unique()
x = df[['DEP_DELAY', 'TAXI_OUT']].groupby('TAXI_OUT').mean()
plt.figure(figsize=(10,10))
plt.scatter(x, y)
plt.show()
#non si vede alcun tipo di linearità


#pairplot
#fig = plt.figure()
#sns.pairplot(dummy_df)
#plt.show()

#quanto cambia la distribuzione considerando i giorni della settimana?

df.groupby('DAY_OF_WEEK')['TAXI_OUT'].mean().plot.bar()
plt.show() #all day of week are at 20 of TAXI OUT MEAN VALUE

df.groupby('MONTH')['TAXI_OUT'].mean().plot.bar()
plt.show() #all 3 months are at 20 of TAXI OUT MEAN VALUE

df.shape