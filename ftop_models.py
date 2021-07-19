import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, validation_curve
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor
from sklearn import metrics
import time
from yellowbrick.regressor import ResidualsPlot

random_state = 17
np.random.seed(17)
# imposto il np random seed perchè gridSearchCv per la cross-validation
# non accetta il random state come parametro ma lavora direttamente con il seed di numpy
# lo imposto in modo tale da avere consistenza di risultati da un tentativo all'altro.

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

# per la standardizzazione utilizzo lo standard Scaler perchè sulla documentazione di scikit learn
# consiglia di standardizzare con questo se viene utilizzata la Lasso Regression.
stdScal = StandardScaler()
stdScal.fit(X_train_raw[numerical_labels].astype('float64'))
X_train_scaled_num = pd.DataFrame(stdScal.transform(X_train_raw[numerical_labels].astype('float64')), columns=numerical_labels)
X_test_scaled_num = pd.DataFrame(stdScal.transform(X_test_raw[numerical_labels].astype('float64')), columns=numerical_labels)

X_train_raw.reset_index(drop = True, inplace = True)
X_test_raw.reset_index(drop = True, inplace = True)
y_train.reset_index(drop = True, inplace = True)
y_test.reset_index(drop = True, inplace = True)

X_train_scaled = pd.concat([X_train_scaled_num, X_train_raw[categorical_labels]], axis = 1)
X_test_scaled = pd.concat([X_test_scaled_num, X_test_raw[categorical_labels]], axis = 1)

##################################################################################################################
#                                                     MODELS                                                     #
##################################################################################################################
#                                                                                                                #
#                                      CROSS-VALIDATION FOR MODEL SELECTION                                      #
#                                                       +                                                        #
#                                                    TRAINING                                                    #
#                                                                                                                #
##################################################################################################################

lasso_raw_model = Lasso(random_state=random_state)
lasso_std_model = Lasso(random_state=random_state)
knn_raw_model = KNeighborsRegressor()
knn_std_model = KNeighborsRegressor()
mlp_raw_model = MLPRegressor(early_stopping=True,
                             solver='sgd',
                             validation_fraction=0.125,  #perchè il 10% del dataset (per la validation) equivale al 12.5% del training set
                             verbose=True,
                             max_iter=1000,
                             random_state=random_state)
mlp_std_model = MLPRegressor(early_stopping=True,
                             solver='sgd',
                             validation_fraction=0.125,  #perchè il 10% del dataset (per la validation) equivale al 12.5% del training set
                             verbose=True,
                             max_iter=1000,
                             random_state=random_state)

def cv_model(model, h_params, X_train, y_train):
    model_cv = GridSearchCV(estimator=model, param_grid=h_params, scoring='neg_root_mean_squared_error', refit=False)
    print('Fitting all models ...  ...  ...')
    model_cv.fit(X_train, y_train)
    return model_cv

def print_cv_result(model_cv, desc):
    print(desc)
    for param in model_cv.best_params_.keys():
        print('best -', param, 'is', model_cv.best_params_.get(param))
    print('These parameters scored a negative-RMSE of -->', model_cv.best_score_, '\n')

def print_metrics(desc, y_test, y_pred):
    print(desc, 'metrics:')
    print('RMSE -->', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    print('R2-score -->', metrics.r2_score(y_test, y_pred))


# come metrica per lo scoring di cross-validation utilizzerò RMSE perchè voglio confrontare le
# performance tra i vari modelli, e voglio che gli errori più grandi abbiano più peso. (MAE non va bene).

#---------  1: LASSO RAW ---------------------------------------------------------

alphas = np.arange(0.0001, 0.1, 0.0005)
# ho provato con questi range e il migliore è 0.0016 quindi ora raffino il range e riprovo
alphas = np.round(np.arange(0.0007, 0.003, 0.0001), 5)
# con questo range il valore di alpha migliore è rimasto 0.0016
# Quindi confermo questo iperparametro.
parameters = {'alpha' : alphas}

lasso_raw = cv_model(lasso_raw_model, parameters, X_train_raw, y_train)
print_cv_result(lasso_raw, 'LASSO Regression (RAW DATA)')

# neg-RMSE =  -6.567514581292189

#---------  2: LASSO PRE-PROCESSED -----------------------------------------------

alphas = np.arange(0.0001, 0.1, 0.0005)
# ho provato con questi range e il migliore è 0.0056 quindi ora raffino il range e riprovo
alphas = np.arange(0.0001, 0.01, 0.0001)
# con questo range il valore di alpha migliore è 0.0057, molto simile al primo.
# Quindi confermo questo iperparametro.
parameters = {'alpha' : alphas}

lasso_std = cv_model(lasso_std_model, parameters, X_train_scaled, y_train)
print_cv_result(lasso_std, 'LASSO Regression (PRE-PROCESSED DATA)')

# neg-RMSE =  -6.5672752218770025
#migliore

#---------  4: KNN RAW ---------------------------------------------------------

parameters = {'n_neighbors': list(range(1, 12, 2)), 'weights': ['uniform', 'distance']}

knn_raw = cv_model(knn_raw_model, parameters, X_train_raw, y_train)
print_cv_result(knn_raw, 'KNN Regressor (RAW DATA)')

# neg-RMSE = -6.610380820947623
#migliore

#---------  5: KNN PRE-PROCESSED ---------------------------------------------------------

parameters = {'n_neighbors': list(range(1, 12, 2)), 'weights': ['uniform', 'distance']}

knn_std = cv_model(knn_std_model, parameters, X_train_scaled, y_train)
print_cv_result(knn_std, 'KNN Regressor (PRE-PROCESSED DATA)')

# neg-RMSE = -6.63741941865838

#---------  7: MLPRegressor RAW ---------------------------------------------------------

h_layers = [(500,),(200,),(200,200),(100, 100),(200, 100, 100),(50, 50, 50, 50),(25, 25, 25, 25, 25, 25, 25, 25)]
parameters = {'hidden_layer_sizes': h_layers, 'activation': ['identity','tanh', 'relu','logistic']}
#best = 100,100 , tanh

mlp_raw = cv_model(mlp_raw_model, parameters, X_train_raw, y_train)
print_cv_result(mlp_raw, 'Neural-Network MLP Regressor (RAW DATA)')

# neg-RMSE =  -6.87219092230963


#---------  8: MLPRegressor PRE-PROCESSED ---------------------------------------------------------

h_layers = [(500,),(200,),(200,200),(100, 100),(200, 100, 100),(50, 50, 50, 50),(25, 25, 25, 25, 25, 25, 25, 25)]
parameters = {'hidden_layer_sizes': h_layers, 'activation': ['identity','tanh', 'relu','logistic']}
#best= (200, 100, 100), logistic

mlp_std = cv_model(mlp_std_model, parameters, X_train_scaled, y_train)
print_cv_result(mlp_std, 'KNN Regressor (PRE-PROCESSED DATA)')

#questo è il migliore dei due neg-RMSE =  -6.497198276125597
#ha impiegato circa 40 minuti


# per l'ensamble utilizzo l'algoritmo di bagging, perchè nonostante il bias sia ancora abbastanza elevato,
# i modelli utilizzati non sono weak (es. linear regression, o neural network con soltanto 1 neurone)
# pertanto utilizzo un algotirmo di bagging per calare la varianza.

#---------  3: LASSO ENSEMBLE (PRE-PROCESSED) ---------------------------------------------------------
lasso_ensemble = BaggingRegressor(lasso_std.best_estimator_, max_samples=0.875)
lasso_ensemble.fit(X_train_scaled, y_train)
y_pred_lasso_ensemble = lasso_ensemble.predict(X_test_scaled)
np.sqrt(metrics.mean_squared_error(y_test, y_pred_lasso_ensemble))

#---------  6: KNN ENSEMBLE (RAW) ---------------------------------------------------------------------
knn_ensemble = BaggingRegressor(knn_raw.best_estimator_, max_samples=0.875)
knn_ensemble.fit(X_train_raw, y_train)
y_pred_knn_ensemble = knn_ensemble.predict(X_test_raw)
np.sqrt(metrics.mean_squared_error(y_test, y_pred_knn_ensemble))


#---------  9: MLP ENSEMBLE (PRE-PROCESSED) -----------------------------------------------------------
mlp_ensemble = BaggingRegressor(lasso_std.best_estimator_, max_samples=0.875)
mlp_ensemble.fit(X_train_scaled, y_train)
y_pred_mlp_ensemble = mlp_ensemble.predict(X_test_scaled)
np.sqrt(metrics.mean_squared_error(y_test, y_pred_mlp_ensemble))


##################################################################################################################
#                                           TRAINING + TEST                                                      #
##################################################################################################################

# c'è bisogno di effettuare il re-training dei modelli perchè quando è stata effettuata
# la model selection, gridSearchCV aveva di l'attributo Refit=False
# Questo comporta che il refit sia necessario, utilizzando l'intero training-set.
# Questa scelta è stata compiuta per effettuare il re-training manualmente in modo tale
# da misurare i tempi di calcolo.


official_lasso_raw = lasso_raw.best_estimator_
official_lasso_std = lasso_std.best_estimator_
official_lasso_ensemble = lasso_ensemble
official_knn_raw = knn_raw.best_estimator_
official_knn_std = knn_std.best_estimator_
official_knn_ensemble = knn_ensemble
official_mlp_raw = mlp_raw.best_estimator_
official_mlp_std = mlp_std.best_estimator_
official_mlp_ensemble = mlp_ensemble

#---------  1: LASSO RAW ---------------------------------------------------------

#train
start = time.perf_counter()
official_lasso_raw.fit(X_train_raw, y_train)
stop = time.perf_counter()
lasso_raw_performance = stop - start

#predict
y_pred_lasso_raw = official_lasso_raw.predict(X_test_raw)
print_metrics('LASSO RAW', y_test, y_pred_lasso_raw)

#---------  2: LASSO PRE-PROCESSED -----------------------------------------------

#train
start = time.perf_counter()
official_lasso_std.fit(X_train_scaled, y_train)
stop = time.perf_counter()
lasso_std_performance = stop - start

#predict
y_pred_lasso_std = official_lasso_std.predict(X_test_scaled)
print_metrics('LASSO PRE-PROCESSED', y_test, y_pred_lasso_std)


#---------  3: LASSO ENSEMBLE (PRE-PROCESSED) ---------------------------------------------------------

start = time.perf_counter()
official_lasso_ensemble.fit(X_train_scaled, y_train)
stop = time.perf_counter()
lasso_ensemble_performance = stop - start

#predict
y_pred_lasso_ensemble = official_lasso_ensemble.predict(X_test_scaled)
print_metrics('LASSO ENSEMBLE PRE-PROCESSED', y_test, y_pred_lasso_ensemble)

#---------  4: KNN RAW ---------------------------------------------------------

#train
start = time.perf_counter()
official_knn_raw.fit(X_train_raw, y_train)
stop = time.perf_counter()
knn_raw_performance = stop - start

#predict
y_pred_knn_raw = official_knn_raw.predict(X_test_raw)
print_metrics('KNN RAW', y_test, y_pred_knn_raw)



#---------  5: KNN PRE-PROCESSED ---------------------------------------------------------

#train
start = time.perf_counter()
official_knn_std.fit(X_train_scaled, y_train)
stop = time.perf_counter()
knn_std_performance = stop - start

#predict
y_pred_knn_std = official_knn_std.predict(X_test_scaled)
print_metrics('KNN PRE-PROCESSED', y_test, y_pred_knn_std)

#---------  6: KNN ENSEMBLE (RAW) ---------------------------------------------------------------------

start = time.perf_counter()
official_knn_ensemble.fit(X_train_raw, y_train)
stop = time.perf_counter()
knn_ensemble_performance = stop - start

#predict
y_pred_knn_ensemble = official_knn_ensemble.predict(X_test_raw)
print_metrics('KNN ENSEMBLE RAW', y_test, y_pred_knn_ensemble)


#---------  7: MLPRegressor RAW ---------------------------------------------------------

#train
start = time.perf_counter()
official_mlp_raw.fit(X_train_raw, y_train)
stop = time.perf_counter()
mlp_raw_performance = stop - start

#predict
y_pred_mlp_raw = official_mlp_raw.predict(X_test_raw)
print_metrics('MLP RAW', y_test, y_pred_mlp_raw)


#---------  8: MLPRegressor PRE-PROCESSED ---------------------------------------------------------

#train
start = time.perf_counter()
official_mlp_std.fit(X_train_scaled, y_train)
stop = time.perf_counter()
mlp_std_performance = stop - start

#predict
y_pred_mlp_std = official_mlp_std.predict(X_test_scaled)
print_metrics('MLP PRE-PROCESSED', y_test, y_pred_mlp_std)


#---------  9: MLP ENSEMBLE (PRE-PROCESSED) -----------------------------------------------------------

#train
start = time.perf_counter()
official_mlp_ensemble.fit(X_train_scaled, y_train)
stop = time.perf_counter()
mlp_ensemble_performance = stop - start

#predict
y_pred_mlp_ensemble = official_mlp_ensemble.predict(X_test_scaled)
print_metrics('MLP ENSEMBLE PRE-PROCESSED', y_test, y_pred_mlp_ensemble)

##################################################################################################################
#                                               EVALUATION                                                       #
##################################################################################################################

#GRAFICI, COFRONTI, TEMPI DI PERFORMANCE, EVALUATIONS FINALI