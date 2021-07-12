import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

pd.set_option('display.expand_frame_repr',
              False)  # to show all the columns in the console
sns.set(style="white")

# infos about the dataset: https://www.kaggle.com/deepankurk/flight-take-off-data-jfk-airport
df = pd.read_csv('dataset/M1_final.csv')
print(df.head())
print(df.shape)
print(df.columns)

print(df.info())  # I notice that the column "Wind" has 2 null samples
df.isnull().sum()  # double-check if some values is null
print('Wind missing values --> ', np.round(df['Wind'].isnull().mean(), 4), '%')

# I see that missing values are just 0.0001% so i decide to remove that instances
df.dropna(axis=0, inplace=True)

df.describe(include=object)

# some queries on target variable
print(df['TAXI_OUT'].mean())
print(df['TAXI_OUT'].value_counts())

airlines = df['OP_UNIQUE_CARRIER'].unique()
len(airlines)
counts = df['OP_UNIQUE_CARRIER'].value_counts()
counts.plot(kind='barh')
plt.show()
# VEDO CHE IL NUMERO DI FLIGHTS CHE ABBIAMO NON è BILANCIATISSIMO RISPETTO A QUALI COMPAGNIE CONSIDERIAMO

df['CRS_DEP_M'] = pd.to_datetime(df.CRS_DEP_M,
                                 unit='m').dt.strftime('%H:%M')
df['DEP_TIME_M'] = pd.to_datetime(df.DEP_TIME_M,
                                  unit='m').dt.strftime('%H:%M')
df['CRS_ELAPSED_TIME'] = pd.to_datetime(df.CRS_ELAPSED_TIME,
                                        unit='m').dt.strftime('%H:%M')
df['CRS_ARR_M'] = pd.to_datetime(df.CRS_ARR_M,
                                 unit='m').dt.strftime('%H:%M')
print(df.head())

df.drop('TAIL_NUM',
        axis=1,
        inplace=True)
df.head()

df['YEAR'] = np.where(df['MONTH'] == 1,
                      2020,
                      2019)  # we have only january, december and november so if month == january then year = 2020
df['date'] = pd.to_datetime(df.YEAR * 10000 + df.MONTH * 100 + df.DAY_OF_MONTH,
                            format='%Y%m%d')
df.head()

y = df[['date', 'TAXI_OUT']].groupby('date').mean()
x = df.date.unique()

# a graph to see the target trend over time (with mean foe each day)
plt.plot_date(x=x,
              y=y,
              fmt='k--x')
plt.gcf().autofmt_xdate()
plt.show()
# non si vede alcun tipo di linearità


df['Dew Point'] = df['Dew Point'].str.strip()

df['Dew Point'] = df['Dew Point'].astype(int)
df.info()
print(df.dtypes)

# dummyzzazione con label encoder
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
# grafico in cui metto taxi out / dep_delay
y = df['TAXI_OUT'].unique()
x = df[['DEP_DELAY', 'TAXI_OUT']].groupby('TAXI_OUT').mean()
plt.figure(figsize=(10, 10))
plt.scatter(x, y)
plt.show()
# non si vede alcun tipo di linearità


# pairplot
# fig = plt.figure()
# sns.pairplot(dummy_df)
# plt.show()

# quanto cambia la distribuzione considerando i giorni della settimana?

df.groupby('DAY_OF_WEEK')['TAXI_OUT'].mean().plot.bar()
plt.show()  # all day of week are at 20 of TAXI OUT MEAN VALUE

df.groupby('MONTH')['TAXI_OUT'].mean().plot.bar()
plt.show()  # all 3 months are at 20 of TAXI OUT MEAN VALUE

df.groupby('DAY_OF_MONTH')['TAXI_OUT'].mean().plot.bar()
plt.show()

df.groupby('OP_UNIQUE_CARRIER')['TAXI_OUT'].mean().plot.bar()
plt.show()
# Notiamo da questo grafico che la compagnia che tiene il TAXI_OUT medio più alto è AS.

features = ['DISTANCE',
            'sch_dep',
            'sch_arr',
            'TAXI_OUT']

df[features].hist()
df[features].plot(kind='density',
                  subplots=True,
                  layout=(2, 2),
                  sharex=False)
plt.show()

sns.boxplot(x='TAXI_OUT',
            data=df)

sns.violinplot(data=df['TAXI_OUT'])
plt.show()
# notiamo che sulla variabil target i valori sono ben distribuiti e ci sono pochissimi outliers


# MATRICE DI CORRELAZIONE

correlation_matrix = dummy_df.corr()
sns.heatmap(correlation_matrix,
            cmap='RdYlGn')
plt.show()
# practically we don't have any strong correlation except wind gust/wind speed and temperature/dew point


# which Features shows perfect correlation with "TAXI_OUT"?
plt.figure(figsize=(8, 12))
heatmap = sns.heatmap(dummy_df.corr()[['TAXI_OUT']].sort_values(by='TAXI_OUT',
                                                                ascending=False),
                      vmin=-1,
                      vmax=1,
                      annot=True,
                      cmap='BrBG')

heatmap.set_title('Features Correlating with TAXI_OUT',
                  fontdict={'fontsize': 18},
                  pad=16);
plt.show()
# ZERO CORRELATIONS WITH TARGET VARIABLE

# save heatmap as .png file
plt.savefig('heatmap.png',
            dpi=300,
            bbox_inches='tight')

nums = ['DISTANCE',
        'Temperature',
        'Dew Point',
        'Humidity',
        'Wind Speed',
        'Wind Gust',
        'Pressure',
        'sch_dep',
        'sch_arr',
        'TAXI_OUT']

sns.set(style='ticks',
        color_codes=True)

sns.pairplot(df[nums])
plt.show()

# SCATTER AND DENSITY PLOT
num = df.select_dtypes(include=[np.number])
colnames = list(num)
print(colnames)
colnames.remove('MONTH')
colnames.remove('DAY_OF_MONTH')
colnames.remove('DAY_OF_WEEK')
if len(colnames) > 10:
    colisnames = colnames[:10]
num = num[colnames]
ax = pd.plotting.scatter_matrix(num,
                                alpha=0.75,
                                figsize=[20, 20],
                                diagonal='kde')
corrs = num.corr().values
for i, j in zip(*plt.np.triu_indices_from(ax, k=1)):
    ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j],
                      (0.8, 0.2),
                      xycoords='axes fraction',
                      ha='center',
                      va='center',
                      size=10)

plt.suptitle('scatter and density plot')
plt.show()

ax = sns.catplot(y="TAXI_OUT",
                 kind="count",
                 data=df,
                 height=2.6,
                 aspect=2.5,
                 orient='h')
plt.show()

df.shape

# Feature Importance (it will be clearer later...) #####################################################################
params = {'random_state': 0,
          'n_jobs': 4,
          'n_estimators': 5000,
          'max_depth': 8}
# One-hot encode
# df = pd.get_dummies(df)
# Drop redundant columns (for features with two unique values)

dummy_df.fillna(method='ffill', inplace=True)

drop = ['TAXI_OUT',
        'CRS_DEP_M',
        'DEP_TIME_M',
        'CRS_ELAPSED_TIME',
        'CRS_ARR_M',
        'date',
        'YEAR']

x, y = dummy_df.drop(drop,
                     axis=1), \
       dummy_df['TAXI_OUT']

# Fit RandomForest Regressor
clf = RandomForestRegressor(**params)
clf = clf.fit(x, y)
x.dtypes
# Plot features importances
imp = pd.Series(data=clf.feature_importances_, index=x.columns).sort_values(ascending=False)
plt.figure(figsize=(10, 12))
plt.title("Feature importance")
ax = sns.barplot(y=imp.index,
                 x=imp.values,
                 palette="Blues_d",
                 orient='h')
plt.show()
