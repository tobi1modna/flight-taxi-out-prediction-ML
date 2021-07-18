import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.svm import SVR
from sklearn import metrics
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
#                                                                                                                #
##################################################################################################################

lasso_raw_model = Lasso(random_state=random_state)
lasso_std_model = Lasso(random_state=random_state)
knn_model = KNeighborsRegressor()
knn_std_model = KNeighborsRegressor()
svm_model = SVR()
svm_std_model = SVR()
nn_model = MLPRegressor(random_state=random_state)
nn_std_model = MLPRegressor()

#---------  1: LASSO RAW ---------------------------------------------------------

alphas = np.arange(0.0001, 0.1, 0.0005)
# ho provato con questi range e il migliore è 0.0016 quindi ora raffino il range e riprovo
alphas = np.round(np.arange(0.0007, 0.003, 0.0001), 5)
# con questo range il valore di alpha migliore è rimasto 0.0016
# Quindi confermo questo iperparametro.
parameters = {'alpha' : alphas}

lasso_raw = GridSearchCV(estimator=lasso_raw_model, param_grid=parameters, scoring='r2')
lasso_raw.fit(X_train_raw, y_train)

print('Lasso With raw data')
print('Overall, the best value for parameter alpha is ', lasso_raw.best_params_.get('alpha'),
      ' since it leads to r2-score = ', lasso_raw.best_score_, '\n')

#---------  2: LASSO PRE-PROCESSED -----------------------------------------------

alphas = np.arange(0.0001, 0.1, 0.0005)
# ho provato con questi range e il migliore è 0.0056 quindi ora raffino il range e riprovo
alphas = np.arange(0.0001, 0.01, 0.0001)
# con questo range il valore di alpha migliore è 0.0057, molto simile al primo.
# Quindi confermo questo iperparametro.
parameters = {'alpha' : alphas}

lasso_std = GridSearchCV(estimator=lasso_std_model, param_grid=parameters, scoring='r2')
lasso_std.fit(X_train_scaled, y_train)

print('Lasso preprocessed data')
print('Overall, the best value for parameter alpha is ', lasso_std.best_params_.get('alpha'),
      ' since it leads to r2-score = ', lasso_std.best_score_, '\n')

#---------  3: KNN RAW ---------------------------------------------------------

parameters = {'n_neighbors': list(range(1, 12, 2))}

knn_raw = GridSearchCV(estimator=knn_model, param_grid=parameters, scoring='neg_root_mean_squared_error')
knn_raw.fit(X_train_raw, y_train)

print('KNN With raw data')
print('Overall, the best value for parameter K is ', knn_raw.best_params_.get('n_neighbors'),
      ' since it leads to r2-score = ', knn_raw.best_score_, '\n')

# SVM ANCHE ADDESTRANDO SOLO UN MODELLO, SENZA CROSS-VALIDATION, E CON TOLLERANZA MOLTO ALTA (1)
# CON KERNEL LINEARE, CI METTE PIU DI MEZZORA NON SO QUANTO NON HA MAI FINITO.

# MENTRE KNN HA UN TEMPO MOLTO ACCETTABILE





#---------  5: MLPRegressor RAW ---------------------------------------------------------

nn_model.fit(X_train_raw, y_train)




#parameters = {'hidden_layer_sizes': ['linear'], 'activation': 'identity'}
parameters = {'activation': ['identity', 'logistic', 'tanh', 'relu']}

nn_raw = GridSearchCV(estimator=nn_model, param_grid=parameters, scoring='r2')
nn_raw.fit(X_train_raw, y_train)

print('NN With raw data')
print('Overall, the best value for parameter activation is ', nn_raw.best_params_.get('activation'),
      ' since it leads to r2-score = ', nn_raw.best_score_, '\n')

#CI HA MESSO CIRCA 3 MINUTI E L'OUTPUT è STATO:
#NN With raw data
#Overall, the best value for parameter activation is  logistic  since it leads to r2-score =  0.06788356737736634

#DATI I TEMPI POSSO PERMETTERMI DI TESTARE MOLTI PIU' IPERPARAMETRI SULLA RETE NEURALE

