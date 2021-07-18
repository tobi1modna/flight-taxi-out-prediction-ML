import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.svm import SVR
from sklearn import metrics
from yellowbrick.regressor import ResidualsPlot

random_state = 17

pd.set_option('display.expand_frame_repr',
              False)  # to show all the columns in the console
sns.set(style="white")


df = pd.read_csv('dataset/M1_final.csv')

##################################################################################################################
#                                              PRE-PROCESSING                                                   #
##################################################################################################################

df.dropna(axis=0, inplace = True)
df.drop('TAIL_NUM',
        axis=1,
        inplace = True)
df['Dew Point'] = df['Dew Point'].str.strip()
df['Dew Point'] = df['Dew Point'].astype(int)

##DISTANCE E ELAPSED TIME SONO ESTREMAMENTE COLLINEARI (0.99 COME COEFFICIENTE DI CORRELAZIONE)
##QUINDI DECIDO DI CANCELLARE LA FEATURE 'DISTANCE'
#DECIDO ANCHE DI CANCELLARE TAIL NUM PERCHE NON HA ALCUN SIGNIFICATO
df.drop(['DISTANCE'], axis = 1,
        inplace = True)


## ENCODING CATEGORICAL FEATURES WITH LABEL ENCODER

labeled_df = df.copy()
label_encoder = LabelEncoder()
categorical = (labeled_df.dtypes == 'object')
numerical = (labeled_df.dtypes != 'object')
categorical_labels = list(categorical[categorical].index)
numerical_labels = list(numerical[numerical].index)
numerical_labels.remove('TAXI_OUT')
print(categorical_labels)
for column in categorical_labels:
    labeled_df[column] = label_encoder.fit_transform(df[column])


#guardando il grafico distribuzione di departure delay, noto che è sbilanciatissimo sull'asse x, tutto attaccato all'asse y.
# Inoltre lungo la lunghezza dell'asse x noto che sono quasi tutti outliers (si vede bene dal box plot)
# Decido quindi di applicare qualche trasformazione su questa feature, in modo da rendere la distribuzione un po'
# più simile ad una gaussiana, per quanto possibile. Questa operazione la considero come parte dei dati raw (non standardizzati)
# quindi per i modelli che utilizzano i dati raw, questa modifica sarà apportata.
###DEPARTURE DELAY HO FATTO LO SCALING PER TOGLIERE I VALORI NEGATIVI POI HO FATTO ESPONENZIALE E LOGARITMO
x = labeled_df['DEP_DELAY'].values.reshape(28818, 1)
np.amin(x)
x = x + np.abs(np.amin(x))
x_logged = np.log(x + 1)
sns.distplot(x)
plt.show()
sns.boxplot(x)
plt.show()
sns.distplot(x_logged)
plt.show()
sns.boxplot(x_logged)
plt.show()
#LA RE-INSERISCO NEL DATAFRAME
x_logged = x_logged.reshape(28818,)
labeled_df['DEP_DELAY'] = x_logged.tolist()#il problema dell'errore che dava qui l'ho risolto inizializzando num con una copia profonda di df

# noto che la distribuzione è parecchio migliorata rispetto a prima.
# num è quindi il dataframe solo numerico che andrà in pasto al minmax scaler per comporre il set preprocessato



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
#                                      CONTROLLO SE CE' LINEARITA' NEI DATI                                      #
##################################################################################################################


##################################################################################################################
#                                        SPLITTING   &   PREPROCESSING 2                                         #
##################################################################################################################

X = labeled_df.copy()
X.drop('TAXI_OUT', axis = 1, inplace = True)
y = labeled_df['TAXI_OUT']

X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, random_state=random_state, test_size=0.2)

minmaxScal = MinMaxScaler()
minmaxScal.fit(X_train_raw[numerical_labels].astype('float64'))
X_train_scaled_num = pd.DataFrame(minmaxScal.transform(X_train_raw[numerical_labels].astype('float64')), columns=numerical_labels)
X_test_scaled_num = pd.DataFrame(minmaxScal.transform(X_test_raw[numerical_labels].astype('float64')), columns=numerical_labels)

X_train_raw.reset_index(drop = True, inplace = True)
X_test_raw.reset_index(drop = True, inplace = True)
y_train.reset_index(drop = True, inplace = True)
y_test.reset_index(drop = True, inplace = True)

X_train_scaled = pd.concat([X_train_scaled_num, X_train_raw[categorical_labels]], axis = 1)
X_test_scaled = pd.concat([X_test_scaled_num, X_test_raw[categorical_labels]], axis = 1)

##################################################################################################################
#                                                     MODELS                                                     #
##################################################################################################################

lasso_model = Lasso()
svm_model = SVR()
nn_model = MLPRegressor()

##################################################################################################################
#                                      CROSS-VALIDATION FOR MODEL SELECTION                                      #
##################################################################################################################

## PER LASSO POTRO FARE O CONTROLLO CON LASSOCV E KFOLD, POI CONFRONTO GLI ALPHA, OPPURE USO SOLO GRIDCV
# E VADO CON IL METODO DEI RAFFINAMENTI SUCCESSIVI. MI SEMBRA LA SCELTA MIGLIORE.




