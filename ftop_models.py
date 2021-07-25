import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.ensemble import BaggingRegressor
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

# ri-apporto tutte le modifiche che sono state fatte al dataset nella fase di EDA.

df.dropna(axis=0, inplace=True)
df.drop('TAIL_NUM',
        axis=1,
        inplace=True)
df['Dew Point'] = df['Dew Point'].str.strip()
df['Dew Point'] = df['Dew Point'].astype(int)

##DISTANCE E ELAPSED TIME SONO ESTREMAMENTE COLLINEARI (0.99 COME COEFFICIENTE DI CORRELAZIONE)
##QUINDI DECIDO DI CANCELLARE LA FEATURE 'DISTANCE'
# DECIDO ANCHE DI CANCELLARE TAIL NUM PERCHE NON HA ALCUN SIGNIFICATO
df.drop(['DISTANCE'], axis=1,
        inplace=True)

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

# guardando il grafico distribuzione di departure delay, noto che è sbilanciatissimo sull'asse x, tutto attaccato all'asse y.
# Inoltre lungo la lunghezza dell'asse x noto che sono quasi tutti outliers (si vede bene dal box plot)
# Decido quindi di applicare qualche trasformazione su questa feature, in modo da rendere la distribuzione un po'
# più simile ad una gaussiana, per quanto possibile. Questa operazione la considero come parte dei dati raw (non standardizzati)
# quindi per i modelli che utilizzano i dati raw, questa modifica sarà apportata.

x = labeled_df['DEP_DELAY'].values.reshape(28818, 1)
np.amin(x)
x = x + np.abs(np.amin(x))
x_logged = np.log(x + 1)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
sns.distplot(ax=axes[0, 0], x=x)
sns.boxplot(ax=axes[1, 0], x=x)
sns.distplot(ax=axes[0, 1], x=x_logged)
sns.boxplot(ax=axes[1, 1], x=x_logged)
axes[0, 0].set_title('DEP_DELAY', fontsize=16)
axes[0, 1].set_title('log(DEP_DELAY + |min(DEP_DELAY)| + 1)', fontsize=16)
axes[0, 0].set_ylabel('Distribution', fontsize=15)
axes[0, 1].set_ylabel('')
axes[1, 0].set_ylabel('Box Distribution', fontsize=15)
axes[1, 1].set_ylabel('')
plt.show()

# LA RE-INSERISCO NEL DATAFRAME
x_logged = x_logged.reshape(28818, )
labeled_df['DEP_DELAY'] = x_logged.tolist()

# noto che la distribuzione è parecchio migliorata rispetto a prima.

##################################################################################################################
#                                      CONTROLLO SE CE' LINEARITA' NEI DATI                                      #
##################################################################################################################

X = labeled_df.copy()
X.drop('TAXI_OUT', axis=1, inplace=True)
y = labeled_df['TAXI_OUT']
# questo splitting è al 100% provvisorio, solo per tracciare il residual plot
x_tr, x_t, y_tr, y_t = train_test_split(X, y, test_size=0.2, random_state=0)

lr = LinearRegression()
residualPlot = ResidualsPlot(lr)

residualPlot.fit(x_tr, y_tr)
residualPlot.score(x_t, y_t)
residualPlot.show()

##################################################################################################################
#                                        SPLITTING   &   PREPROCESSING 2                                         #
##################################################################################################################

X = labeled_df.copy()
X.drop('TAXI_OUT', axis=1, inplace=True)
y = labeled_df['TAXI_OUT']

X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, random_state=random_state, test_size=0.2)

# per la standardizzazione utilizzo lo standard Scaler perchè sulla documentazione di scikit learn
# consiglia di standardizzare con questo se viene utilizzata la Lasso Regression.
stdScal = StandardScaler()
stdScal.fit(X_train_raw[numerical_labels].astype('float64'))
X_train_scaled_num = pd.DataFrame(stdScal.transform(X_train_raw[numerical_labels].astype('float64')),
                                  columns=numerical_labels)
X_test_scaled_num = pd.DataFrame(stdScal.transform(X_test_raw[numerical_labels].astype('float64')),
                                 columns=numerical_labels)

X_train_raw.reset_index(drop=True, inplace=True)
X_test_raw.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)
y_test.reset_index(drop=True, inplace=True)

X_train_scaled = pd.concat([X_train_scaled_num, X_train_raw[categorical_labels]], axis=1)
X_test_scaled = pd.concat([X_test_scaled_num, X_test_raw[categorical_labels]], axis=1)

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
                             validation_fraction=0.125,
                             # perchè il 10% del dataset (per la validation) equivale al 12.5% del training set
                             max_iter=1000,
                             random_state=random_state)
mlp_std_model = MLPRegressor(early_stopping=True,
                             solver='sgd',
                             validation_fraction=0.125,
                             # perchè il 10% del dataset (per la validation) equivale al 12.5% del training set
                             max_iter=1000,
                             random_state=random_state)


def cv_model(model, h_params, X_train, y_train):
    model_cv = GridSearchCV(estimator=model, param_grid=h_params, scoring='neg_root_mean_squared_error',
                            return_train_score=True)
    print('Fitting all models ...  ...  ...')
    model_cv.fit(X_train, y_train)
    return model_cv


def print_cv_result(model_cv, desc):
    print(desc)
    for param in model_cv.best_params_.keys():
        print('best', param, 'is -->', model_cv.best_params_.get(param))
    print('These parameters scored a negative-RMSE of -->', model_cv.best_score_, '\n')


def print_metrics(desc, y_test, y_pred, time):
    print(desc, 'metrics:')
    print('RMSE -->', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    print('MAE -->', metrics.mean_absolute_error(y_test, y_pred))
    print('Training Time -->', time, '\n')


def get_rmse(y_test, y_pred):
    return np.sqrt(metrics.mean_squared_error(y_test, y_pred))


def get_mae(y_test, y_pred):
    return metrics.mean_absolute_error(y_test, y_pred)


# come metrica per lo scoring di cross-validation utilizzerò RMSE perchè voglio confrontare le
# performance tra i vari modelli, e voglio che gli errori più grandi abbiano più peso.
# in seguito utilizzerò anche MAE


# ---------  1: LASSO RAW ---------------------------------------------------------

alphas = np.arange(0.0001, 0.1, 0.0005)
# ho provato con questi range e il migliore è 0.0016 quindi ora raffino il range e riprovo
alphas = np.round(np.arange(0.0007, 0.003, 0.0001), 5)
# con questo range il valore di alpha migliore è rimasto 0.0015
# Quindi confermo questo iperparametro.
parameters = {'alpha': alphas}

t = time.time()
lasso_raw = cv_model(lasso_raw_model, parameters, X_train_raw, y_train)
t1 = time.time() - t
print_cv_result(lasso_raw, 'LASSO Regression (RAW DATA)')

# ---------  2: LASSO PRE-PROCESSED -----------------------------------------------

alphas = np.arange(0.0001, 0.1, 0.0005)
# ho provato con questi range e il migliore è 0.0056 quindi ora raffino il range e riprovo
alphas = np.arange(0.0001, 0.01, 0.0001)

# con questo range il valore di alpha migliore è 0.0057, molto simile al primo.
# Quindi confermo questo iperparametro.
parameters = {'alpha': alphas}

t = time.time()
lasso_std = cv_model(lasso_std_model, parameters, X_train_scaled, y_train)
t2 = time.time() - t
print_cv_result(lasso_std, 'LASSO Regression (PRE-PROCESSED DATA)')

# migliore


# ---------  4: KNN RAW ---------------------------------------------------------

parameters = {'n_neighbors': list(range(1, 12, 2)), 'weights': ['uniform', 'distance']}

t = time.time()
knn_raw = cv_model(knn_raw_model, parameters, X_train_raw, y_train)
t4 = time.time() - t
print_cv_result(knn_raw, 'KNN Regressor (RAW DATA)')

# migliore

res = pd.DataFrame(knn_raw.cv_results_)
results = ['mean_test_score',
           'mean_train_score',
           'std_test_score',
           'std_train_score']


def pooled_var(stds):
    n = 5
    return np.sqrt(sum((n - 1) * (stds ** 2)) / len(stds) * (n - 1))


fig, axes = plt.subplots(1, len(parameters),
                         figsize=(5 * len(parameters), 7),
                         sharey='row',
                         squeeze=True)

for idx, (param_name, param_range) in enumerate(parameters.items()):
    grouped_df = res.groupby(f'param_{param_name}')[results] \
        .agg({'mean_train_score': 'mean',
              'mean_test_score': 'mean',
              'std_train_score': pooled_var,
              'std_test_score': pooled_var})

    previous_group = res.groupby(f'param_{param_name}')[results]
    lw = 2
    axes[idx].plot(param_range, grouped_df['mean_train_score'], label="Training score",
                   color="darkorange", lw=lw)
    axes[idx].fill_between(param_range, grouped_df['mean_train_score'] - grouped_df['std_train_score'],
                           grouped_df['mean_train_score'] + grouped_df['std_train_score'], alpha=0.2,
                           color="darkorange", lw=lw)
    axes[idx].plot(param_range, grouped_df['mean_test_score'], label="Cross-validation score",
                   color="navy", lw=lw)
    axes[idx].fill_between(param_range, grouped_df['mean_test_score'] - grouped_df['std_test_score'],
                           grouped_df['mean_test_score'] + grouped_df['std_test_score'], alpha=0.2,
                           color="navy", lw=lw)

handles, labels = axes[0].get_legend_handles_labels()
fig.suptitle('Validation curves', fontsize=20)
fig.legend(handles, labels, loc=8, ncol=2, fontsize=18)

fig.subplots_adjust(bottom=0.25, top=0.85)
plt.show()

# ---------  5: KNN PRE-PROCESSED ---------------------------------------------------------

parameters = {'n_neighbors': list(range(1, 12, 2)), 'weights': ['uniform', 'distance']}

t = time.time()
knn_std = cv_model(knn_std_model, parameters, X_train_scaled, y_train)
t5 = time.time() - t
print_cv_result(knn_std, 'KNN Regressor (PRE-PROCESSED DATA)')

# ---------  7: MLPRegressor RAW ---------------------------------------------------------

h_layers = [(500,), (200,), (200, 200), (100, 100), (200, 100, 100), (50, 50, 50, 50), (25, 25, 25, 25, 25, 25)]
parameters = {'hidden_layer_sizes': h_layers, 'activation': ['identity', 'tanh', 'relu', 'logistic']}

t = time.time()
mlp_raw = cv_model(mlp_raw_model, parameters, X_train_raw, y_train)
t7 = time.time() - t
print_cv_result(mlp_raw, 'Neural-Network MLP Regressor (RAW DATA)')

# ---------  8: MLPRegressor PRE-PROCESSED ---------------------------------------------------------

h_layers = [(500,), (200,), (200, 200), (100, 100), (200, 100, 100), (50, 50, 50, 50), (25, 25, 25, 25, 25, 25)]
parameters = {'hidden_layer_sizes': h_layers, 'activation': ['identity', 'tanh', 'relu', 'logistic']}
# best= (200, 100, 100), logistic

t = time.time()
mlp_std = cv_model(mlp_std_model, parameters, X_train_scaled, y_train)
t8 = time.time() - t
print_cv_result(mlp_std, 'Neural-Network MLP Regressor (PRE-PROCESSED DATA)')

# migliore

# grafico per rappresentare il tempo di calcolo delle model-selection.

sns.barplot(x=['Lasso-raw',
               'Lasso-std',
               'KNN-raw',
               'KNN-std',
               'MLP-raw',
               'MLP-std'],
            y=[t1, t2, t4, t5, t7, t8])
plt.ylabel('seconds')
plt.title('Cross-validation for model selection time', fontsize=18)
plt.show()

# ---------  3: LASSO ENSEMBLE (PRE-PROCESSED) ---------------------------------------------------------
lasso_ensemble = BaggingRegressor(lasso_std.best_estimator_)

# ---------  6: KNN ENSEMBLE (RAW) ---------------------------------------------------------------------
knn_ensemble = BaggingRegressor(knn_raw.best_estimator_)

# ---------  9: MLP ENSEMBLE (PRE-PROCESSED) -----------------------------------------------------------
mlp_ensemble = BaggingRegressor(mlp_std.best_estimator_)

##################################################################################################################
#                                           TRAINING + TEST                                                      #
##################################################################################################################

# nonostante il parametro refit di gridsearch è stato lasciato di default su "True"
# scelgo comunque di re-trainare tutti i modelli manualmente, in modo da poter misurare i tempi di calcolo.
# ho deciso di lasciare refit a True perchà altrimenti non avrei l'attributo "best estimator"


official_lasso_raw = lasso_raw.best_estimator_
official_lasso_std = lasso_std.best_estimator_
official_lasso_ensemble = lasso_ensemble
official_knn_raw = knn_raw.best_estimator_
official_knn_std = knn_std.best_estimator_
official_knn_ensemble = knn_ensemble
official_mlp_raw = mlp_raw.best_estimator_
official_mlp_std = mlp_std.best_estimator_
official_mlp_ensemble = mlp_ensemble

# ---------  1: LASSO RAW ---------------------------------------------------------

# train
start = time.time()
official_lasso_raw.fit(X_train_raw, y_train)
lasso_raw_train_time = time.time() - start

# predict
y_pred_lasso_raw = official_lasso_raw.predict(X_test_raw)
print_metrics('LASSO RAW',
              y_test,
              y_pred_lasso_raw,
              lasso_raw_train_time)

# ---------  2: LASSO PRE-PROCESSED -----------------------------------------------

# train
start = time.time()
official_lasso_std.fit(X_train_scaled, y_train)
lasso_std_train_time = time.time() - start

# predict
y_pred_lasso_std = official_lasso_std.predict(X_test_scaled)
print_metrics('LASSO PRE-PROCESSED',
              y_test,
              y_pred_lasso_std,
              lasso_std_train_time)

# ---------  3: LASSO ENSEMBLE (PRE-PROCESSED) ---------------------------------------------------------

start = time.time()
official_lasso_ensemble.fit(X_train_scaled, y_train)
lasso_ensemble_train_time = time.time() - start

# predict
y_pred_lasso_ensemble = official_lasso_ensemble.predict(X_test_scaled)
print_metrics('LASSO ENSEMBLE PRE-PROCESSED',
              y_test,
              y_pred_lasso_ensemble,
              lasso_ensemble_train_time)

# ---------  4: KNN RAW ---------------------------------------------------------

# train
start = time.time()
official_knn_raw.fit(X_train_raw, y_train)
knn_raw_train_time = time.time() - start

# predict
y_pred_knn_raw = official_knn_raw.predict(X_test_raw)
print_metrics('KNN RAW',
              y_test,
              y_pred_knn_raw,
              knn_raw_train_time)

# ---------  5: KNN PRE-PROCESSED ---------------------------------------------------------

# train
start = time.time()
official_knn_std.fit(X_train_scaled, y_train)
knn_std_train_time = time.time() - start

# predict
y_pred_knn_std = official_knn_std.predict(X_test_scaled)
print_metrics('KNN PRE-PROCESSED',
              y_test,
              y_pred_knn_std,
              knn_std_train_time)

# ---------  6: KNN ENSEMBLE (RAW) ---------------------------------------------------------------------

start = time.time()
official_knn_ensemble.fit(X_train_raw, y_train)
knn_ensemble_train_time = time.time() - start

# predict
y_pred_knn_ensemble = official_knn_ensemble.predict(X_test_raw)
print_metrics('KNN ENSEMBLE RAW',
              y_test,
              y_pred_knn_ensemble,
              knn_ensemble_train_time)

# ---------  7: MLPRegressor RAW ---------------------------------------------------------

# train
start = time.time()
official_mlp_raw.fit(X_train_raw, y_train)
mlp_raw_train_time = time.time() - start

# predict
y_pred_mlp_raw = official_mlp_raw.predict(X_test_raw)
print_metrics('MLP RAW',
              y_test,
              y_pred_mlp_raw,
              mlp_raw_train_time)

# ---------  8: MLPRegressor PRE-PROCESSED ---------------------------------------------------------

# train
start = time.time()
official_mlp_std.fit(X_train_scaled, y_train)
mlp_std_train_time = time.time() - start

# predict
y_pred_mlp_std = official_mlp_std.predict(X_test_scaled)
print_metrics('MLP PRE-PROCESSED',
              y_test,
              y_pred_mlp_std,
              mlp_std_train_time)

# ---------  9: MLP ENSEMBLE (PRE-PROCESSED) -----------------------------------------------------------

# train
start = time.time()
official_mlp_ensemble.fit(X_train_scaled, y_train)
mlp_ensemble_train_time = time.time() - start

# predict
y_pred_mlp_ensemble = official_mlp_ensemble.predict(X_test_scaled)
print_metrics('MLP ENSEMBLE PRE-PROCESSED',
              y_test,
              y_pred_mlp_ensemble,
              mlp_ensemble_train_time)

##################################################################################################################
#                                               EVALUATION                                                       #
##################################################################################################################

# GRAFICI, COFRONTI, TEMPI DI CALCOLO, EVALUATIONS FINALI

# EVALUATION LASSO (CV vs TEST)
# RMSE
plt.figure(figsize=(10, 7))
sns.barplot(x=['Lasso-raw (CV)',
               'Lasso-raw (TEST)',
               'Lasso-std (CV)',
               'Lasso-std (TEST)'],
            y=[(lasso_raw.best_score_ * -1),
               get_rmse(y_test, y_pred_lasso_raw),
               (lasso_std.best_score_ * -1),
               get_rmse(y_test, y_pred_lasso_std)])
plt.ylabel('RMSE', size=20)
plt.xticks(size=18)
plt.yticks(size=20)
plt.title('Performance Lasso (CV vs TEST)', fontsize=25)
plt.show()

# EVALUATION LASSO (TEST-STD vs TEST-ENSEMBLE)
sns.barplot(x=['Lasso-std (TEST)', 'Lasso-Ensemble (TEST)'],
            y=[get_rmse(y_test, y_pred_lasso_std),
               get_rmse(y_test, y_pred_lasso_ensemble)])
plt.ylabel('RMSE', size=20)
plt.xticks(size=20)
plt.yticks(size=20)
plt.title('Performance Lasso vs Ensemble', fontsize=25)
plt.show()

# EVALUATION KNN (CV vs TEST)
plt.figure(figsize=(10, 7))
sns.barplot(x=['KNN-raw (CV)',
               'KNN-raw (TEST)',
               'KNN-std (CV)',
               'KNN-std (TEST)'],
            y=[(knn_raw.best_score_ * -1),
               get_rmse(y_test, y_pred_knn_raw),
               (knn_std.best_score_ * -1),
               get_rmse(y_test, y_pred_knn_std)])
plt.ylabel('RMSE', size=20)
plt.xticks(size=18)
plt.yticks(size=20)
plt.title('Performance KNN (CV vs TEST)', fontsize=25)
plt.show()

# EVALUATION KNN (TEST-STD vs TEST-ENSEMBLE)
sns.barplot(x=['KNN-std (TEST)', 'KNN-Ensemble (TEST)'],
            y=[get_rmse(y_test, y_pred_knn_std),
               get_rmse(y_test, y_pred_knn_ensemble)])
plt.ylabel('RMSE', size=20)
plt.xticks(size=20)
plt.yticks(size=20)
plt.title('Performance KNN vs Ensemble', fontsize=25)
plt.show()

# EVALUATION MLP (CV vs TEST)
plt.figure(figsize=(10, 7))
sns.barplot(x=['MLP-raw (CV)',
               'MLP-raw (TEST)',
               'MLP-std (CV)',
               'MLP-std (TEST)'],
            y=[(mlp_raw.best_score_ * -1),
               get_rmse(y_test, y_pred_mlp_raw),
               (mlp_std.best_score_ * -1),
               get_rmse(y_test, y_pred_mlp_std)])
plt.ylabel('RMSE', size=20)
plt.xticks(size=18)
plt.yticks(size=20)
plt.title('Performance MLP (CV vs TEST)', fontsize=25)
plt.show()

# EVALUATION MLP (TEST-STD vs TEST-ENSEMBLE)
sns.barplot(x=['MLP-std (TEST)', 'MLP-Ensemble (TEST)'],
            y=[get_rmse(y_test, y_pred_mlp_std),
               get_rmse(y_test, y_pred_mlp_ensemble)], capsize=20)
plt.ylabel('RMSE', size=20)
plt.xticks(size=20)
plt.yticks(size=20)
plt.title('Performance MLP vs Ensemble', fontsize=25)
plt.show()

# MAE (ONLY IN TEST)
plt.figure(figsize=(9, 6))
sns.barplot(x=['Lasso\nraw',
               'Lasso\nstd',
               'Lasso\nensemble',
               'KNN\nraw',
               'KNN\nstd',
               'KNN\nensemble',
               'MLP\nraw',
               'MLP\nstd',
               'MLP\nensemble'],
            y=[get_mae(y_test, y_pred_lasso_raw),
               get_mae(y_test, y_pred_lasso_std),
               get_mae(y_test, y_pred_lasso_ensemble),
               get_mae(y_test, y_pred_knn_raw),
               get_mae(y_test, y_pred_knn_std),
               get_mae(y_test, y_pred_knn_ensemble),
               get_mae(y_test, y_pred_mlp_raw),
               get_mae(y_test, y_pred_mlp_std),
               get_mae(y_test, y_pred_mlp_ensemble)])
plt.ylabel('MAE', size=18)
plt.title('MAE on test', fontsize=18)
plt.subplots_adjust(bottom=0.2)
plt.xticks(size=15)
plt.yticks(size=15)
plt.show()

# RMSE (ONLY IN TEST)
plt.figure(figsize=(9, 6))
sns.barplot(x=['Lasso\nraw',
               'Lasso\nstd',
               'Lasso\nensemble',
               'KNN\nraw',
               'KNN\nstd',
               'KNN\nensemble',
               'MLP\nraw',
               'MLP\nstd',
               'MLP\nensemble'],
            y=[get_rmse(y_test, y_pred_lasso_raw),
               get_rmse(y_test, y_pred_lasso_std),
               get_rmse(y_test, y_pred_lasso_ensemble),
               get_rmse(y_test, y_pred_knn_raw),
               get_rmse(y_test, y_pred_knn_std),
               get_rmse(y_test, y_pred_knn_ensemble),
               get_rmse(y_test, y_pred_mlp_raw),
               get_rmse(y_test, y_pred_mlp_std),
               get_rmse(y_test, y_pred_mlp_ensemble)])
plt.ylabel('RMSE', size=18)
plt.title('RMSE on test', fontsize=18)
plt.subplots_adjust(bottom=0.2)
plt.xticks(size=15)
plt.yticks(size=15)
plt.show()

# TEMPI DI TRAINING
plt.figure(figsize=(9, 6))
sns.barplot(x=['Lasso\nraw',
               'Lasso\nstd',
               'Lasso\nensemble',
               'KNN\nraw',
               'KNN\nstd',
               'KNN\nensemble',
               'MLP\nraw'],
            y=[lasso_raw_train_time,
               lasso_std_train_time,
               lasso_ensemble_train_time,
               knn_raw_train_time,
               knn_std_train_time,
               knn_ensemble_train_time,
               mlp_raw_train_time])
plt.ylabel('SECONDS', size=18)
plt.title('Training Time', fontsize=18)
plt.subplots_adjust(bottom=0.2)
plt.xticks(size=15)
plt.yticks(size=15)
plt.show()

# TEMPI DI TRAINING
plt.figure(figsize=(9, 6))
sns.barplot(x=['MLP\nraw',
               'MLP\nstd',
               'MLP\nensemble'],
            y=[mlp_raw_train_time,
               mlp_std_train_time,
               mlp_ensemble_train_time])
plt.ylabel('SECONDS', size=18)
plt.title('Training Time', fontsize=18)
plt.subplots_adjust(bottom=0.2)
plt.xticks(size=15)
plt.yticks(size=15)
plt.show()
