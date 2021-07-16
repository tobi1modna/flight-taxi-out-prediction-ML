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


# MATRICE DI CORRELAZIONE

plt.figure(figsize = (20,17))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix,
            cmap='RdYlGn',
            annot = True)
plt.show()

#########################    ENCODING CATEGORICAL FEATURES WITH GET-DUMMIES (One Hot Encoding)    ########################

dummy_df = pd.get_dummies(df)


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
    # Feature Importance WITH DUMMY
    params = {'random_state': 0,
              'n_jobs': 4,
              'n_estimators': 5000,
              'max_depth': 8}

    dummy_df.fillna(method='ffill', inplace=True)

    drop = ['TAXI_OUT']

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


    #provo nel labeled a vedere quanto tempo ci mette una KNN
    knn = KNeighborsRegressor()
    x = dummy_df.drop('TAXI_OUT', axis=1)
    y = dummy_df['TAXI_OUT']
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=0)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_val)
    print(y_pred)
    print(y_val)

    print('RMSE: ', metrics.mean_squared_error(y_val, y_pred, squared=False))
    #con labeled ha fatto prestissimo RMSE 6.877
    #con dummy ci mette lo stesso tempo RMSE 6.874

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
    x = dummy_df.drop('TAXI_OUT', axis=1)
    y = dummy_df['TAXI_OUT']
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

def linear_regression():
    #eseguo una linear regression
    lr = LinearRegression()
    x = scaled_df.drop('TAXI_OUT', axis=1)
    y = labeled_df['TAXI_OUT']
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=0)


    #print('RMSE: ', metrics.mean_squared_error(y_val, y_pred, squared=False))
    #linear regression LABELED RMSE: 6.5458
    #RESIDUAL PLOT

    residualPlot = ResidualsPlot(lr)

    residualPlot.fit(x_train, y_train)
    residualPlot.score(x_val, y_val)
    residualPlot.show()

    # print('RMSE: ', metrics.mean_squared_error(y_val, y_pred, squared=False))
    # linear regression LABELED RMSE: 6.5458
    # RESIDUAL PLOT

    # residualPlot = ResidualsPlot(lr)

    # residualPlot.fit(x_train, y_train)
    # residualPlot.score(x_val, y_val)
    # residualPlot.show()


##################################################################################################################
#                                              PRE-PROCESSING                                                   #
##################################################################################################################


#provo a vedere trasformando qualche features

#x['DISTANCE'].head()
#x['DISTANCE'] = np.log2(x['DISTANCE'])
#x['Temperature'] = np.log2(x['Temperature'])

#df.hist(column = 'DISTANCE')
#plt.show()

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


### SCH ARRIVALS ###
##non ho trovato alcun miglioramento sostanziale




#REGRESSIONE LINEARE
#facciamo una copia del dataset completamente scalato con il minmax scaling

x = labeled_df.drop('TAXI_OUT', axis=1)
x['DEP_DELAY'] = num['DEP_DELAY']
y = num[['TAXI_OUT']]

numerical = (df.dtypes != 'object')
print(numerical)
numerical_labels = list(numerical[numerical].index)
numerical_labels.remove('TAXI_OUT')

x_train, x_t, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

scaler_ppro = MinMaxScaler()
scaler_ppro.fit(x_train[numerical_labels].astype('float64'))
x_scalati_train = pd.DataFrame(scaler_ppro.transform(x_train[numerical_labels].astype('float64')), columns = numerical_labels)
x_scalati_test = pd.DataFrame(scaler_ppro.transform(x_t[numerical_labels].astype('float64')), columns = numerical_labels)

#resetto gli indici delle istanze perchè dopo lo splitting del dataset sono stati spostati tutti gli elemeti in ordine casuale
#e facendo i join delle tabelle non riesco a far coincidere gli indici.
x_train.reset_index(drop = True, inplace = True)
x_t.reset_index(drop = True, inplace = True)
y_train.reset_index(drop = True, inplace = True)
y_test.reset_index(drop = True, inplace = True)
x_training = pd.concat([x_train[categorical_labels], x_scalati_train], axis = 1)
x_test = pd.concat([x_t[categorical_labels], x_scalati_test], axis = 1)

lr = LinearRegression()
#print('RMSE: ', metrics.mean_squared_error(y_val, y_pred, squared=False))
#linear regression LABELED RMSE: 6.5458
#RESIDUAL PLOT

residualPlot = ResidualsPlot(lr)

residualPlot.fit(x_training, y_train)
residualPlot.score(x_test, y_test)
residualPlot.show()




#######       SENZA IL PREPROCESSING      #######
x = labeled_df.drop('TAXI_OUT', axis=1)
x['DEP_DELAY'] = num['DEP_DELAY']
y = num[['TAXI_OUT']]

numerical = (df.dtypes != 'object')
print(numerical)
numerical_labels = list(numerical[numerical].index)
numerical_labels.remove('TAXI_OUT')

x_train, x_t, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


lr = LinearRegression()
#print('RMSE: ', metrics.mean_squared_error(y_val, y_pred, squared=False))
#linear regression LABELED RMSE: 6.5458
#RESIDUAL PLOT

residualPlot = ResidualsPlot(lr)

residualPlot.fit(x_train, y_train)
residualPlot.score(x_t, y_test)
residualPlot.show()







sns.pairplot(x_scalati_train, diag_kind = 'kde')

plt.suptitle('scatter and density plot')
plt.show()















#########################################################################
#               INIZIO A FARE ROBA CON I MODELLI                        #
#########################################################################


