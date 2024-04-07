import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score, f1_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.decomposition import PCA

# Importing dataset
data = pd.read_csv('Data/data_preprocessed.csv')

# Addressing data imbalance by downsampling
# Separate majority and minority classes
majority_class = data[data['fraud'] == 0]
minority_class = data[data['fraud'] == 1]

# Downsample majority class
majority_downsampled = majority_class.sample(n=len(minority_class), random_state=42)

# Combine minority class and downsampled majority class
data = pd.concat([majority_downsampled, minority_class])

# Shuffle the dataset
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# X and y arrays
X = data.drop(['fraud'], axis=1).values
y = data['fraud'].values

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# standardization
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



################################### MODELS ###################################

# Logistic Regression
lr = LogisticRegression()
params = {'penalty':['l1', 'l2', 'elasticnet', None],
          'dual':[True,False],
          'tol':[0.000001,0.00001, 0.0001, 0.001, 0.01],
          'C':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],
          'solver':['liblinear', 'newton-cholesky']}
grid_search = GridSearchCV(estimator=lr,
                           param_grid=params,
                           scoring=['recall','precision'],
                           refit='recall',
                           cv=10,
                           n_jobs=-1)
grid_search.fit(X_train, y_train)
results_lr = grid_search.cv_results_
best_recall_lr = grid_search.best_score_
best_parameters_lr = grid_search.best_params_
print("Best Recall: {:.2f} %".format(best_recall_lr*100))
print("Best Parameters:", best_parameters_lr)


# Perceptron
percept = Perceptron()
params = {'penalty':['l1','l2','elasticnet',None],
          'alpha':[0.00001,0.0001,0.001,0.01,0.1,1.0,10],
          'fit_intercept':[True,False],
          'shuffle':[True,False]}
grid_search = GridSearchCV(estimator=percept,
                           param_grid=params,
                           scoring=['recall','precision'],
                           refit='recall',
                           cv=10,
                           n_jobs=-1)
grid_search.fit(X_train, y_train)
results_percept = grid_search.cv_results_
best_recall_percept = grid_search.best_score_
best_parameters_percept = grid_search.best_params_
print("Best Recall: {:.2f} %".format(best_recall_percept*100))
print("Best Parameters:", best_parameters_percept)



# SVM
svc = SVC()
params = {'C':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],
          'kernel':['linear','rbf','sigmoid','poly'],
          'degree':[3,5,7],
          'gamma':['auto','scale'],
          'shrinking':[True,False]}
grid_search = GridSearchCV(estimator=svc,
                           param_grid=params,
                           scoring=['recall','precision'],
                           refit='recall',
                           cv=10,
                           n_jobs=-1)
grid_search.fit(X_train, y_train)
results_svc = grid_search.cv_results_
best_recall_svc = grid_search.best_score_
best_parameters_svc = grid_search.best_params_
print("Best Recall: {:.2f} %".format(best_recall_svc*100))
print("Best Parameters:", best_parameters_svc)


################################### EXPORTING THE MODEL ###################################

import pickle
model_file = "FlaskAPI/Models/model_file.p"
with open(model_file, 'wb') as file:
    pickle.dump({'model': grid_search.best_estimator_, 'scaler': sc}, file)

