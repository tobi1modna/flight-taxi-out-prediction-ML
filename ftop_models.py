import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.svm import SVR
from sklearn import metrics
from yellowbrick.regressor import ResidualsPlot

pd.set_option('display.expand_frame_repr',
              False)  # to show all the columns in the console
sns.set(style="white")

df = pd.read_csv('dataset/M1_final.csv')

df.dropna(axis=0, inplace=True)
df.drop('TAIL_NUM',
        axis=1,
        inplace=True)
df['Dew Point'] = df['Dew Point'].str.strip()
df['Dew Point'] = df['Dew Point'].astype(int)

##DISTANCE E ELAPSED TIME SONO ESTREMAMENTE COLLINEARI (0.99 COME COEFFICIENTE DI CORRELAZIONE)
##QUINDI DECIDO DI CANCELLARE LA FEATURE 'DISTANCE'
#DECIDO ANCHE DI CANCELLARE TAIL NUM PERCHE NON HA ALCUN SIGNIFICATO
df.drop(['DISTANCE'], axis = 1,
        inplace = True)


#########################           ENCODING CATEGORICAL FEATURES WITH LABEL ENCODER              ########################

labeled_df = df.copy()
label_encoder = LabelEncoder()
categorical = (labeled_df.dtypes == 'object')
print(categorical)
categorical_labels = list(categorical[categorical].index)
print(categorical_labels)
for column in categorical_labels:
    labeled_df[column] = label_encoder.fit_transform(df[column])


def feature_importance():

    params = {'random_state': 0,
              'n_jobs': 4,
              'n_estimators': 5000,
              'max_depth': 8}

    # Feature Importance WITH LABEL ENCODER
    labeled_df.fillna(method='ffill', inplace=True)

    drop = ['TAXI_OUT']

    x, y = labeled_df.drop(drop,
                           axis=1), \
           labeled_df['TAXI_OUT']

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


    #provo nel labeled a vedere quanto tempo ci mette una RANDOM FOREST
    rf = RandomForestRegressor()
    x = labeled_df.drop('TAXI_OUT', axis=1)
    y = labeled_df['TAXI_OUT']
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=0)
    rf.fit(x_train, y_train)
    y_pred = rf.predict(x_val)
    print(y_pred)
    print(y_val)

    print('RMSE: ', metrics.mean_squared_error(y_val, y_pred, squared=False))
    #con labeled 30 secondi circa RMSE:  5.7076
    #con dummy ci mette lo stesso tempo 1 minuto circa



    #provo nel labeled a vedere quanto tempo ci mette una SVM
    svr = SVR()
    x = labeled_df.drop('TAXI_OUT', axis=1)
    y = labeled_df['TAXI_OUT']
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=0)
    svr.fit(x_train, y_train)
    y_pred = svr.predict(x_val)
    print(y_pred)
    print(y_val)

    print('RMSE: ', metrics.mean_squared_error(y_val, y_pred, squared=False))
    #con labeled 1 minuto e 15  RMSE:  6.8467
    #con dummy 2 minuti e 50 secondi RMSE: 6.8585



##################################################################################################################
#                                CONTROLLO SE CE' LINEARITA' NEI DATI                                            #
##################################################################################################################








##################################################################################################################
#                                              PRE-PROCESSING                                                   #
##################################################################################################################


num = df.copy()
num = num.select_dtypes(include=[np.number])


###DEPARTURE DELAY HO FATTO LO SCALING PER TOGLIERE I VALORI NEGATIVI POI HO FATTO ESPONENZIALE E LOGARITMO
x = num['DEP_DELAY'].values.reshape(28818, 1)
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x.astype('float64'))
num['new'] = x_scaled
x_logged = np.log(x_scaled + 1)
xx = (x_logged)**(1/5)
sns.distplot(xx)
plt.show()
sns.boxplot(x)
plt.show()
sns.boxplot(xx)
plt.show()
#LA RE-INSERISCO NEL DATAFRAME
xx = xx.reshape(28818,)
num['DEP_DELAY'] = xx.tolist()#il problema dell'errore che dava qui l'ho risolto inizializzando num con una copia profonda di df


#########################################################################
#                        INIZIO CON I MODELLI                          #
#########################################################################


