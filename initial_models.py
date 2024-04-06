import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score, f1_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


def accuracy_scores(models):
    cross_val_acc = []
    cross_val_std = []
    accuracy = []
    recall = []
    precision = []
    f1Score = []
    train_time = []
    conf_matrix = []
    
    for model in models:
        start_time = pd.datetime.now()
        model.fit(X_train, y_train)
        end_time = pd.datetime.now()
        
        y_pred = model.predict(X_test)
        cross_score = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=10)
        
        cross_val_acc.append(round(cross_score.mean(),4))
        cross_val_std.append(round(cross_score.std(),6))
        accuracy.append(round(accuracy_score(y_test, y_pred),4))
        recall.append(round(recall_score(y_test, y_pred),4))
        precision.append(round(precision_score(y_test, y_pred),4))
        f1Score.append(round(f1_score(y_test, y_pred),4))
        conf_matrix.append(confusion_matrix(y_test, y_pred))
        train_time.append(end_time - start_time)
    
    return accuracy, recall, precision, f1Score, train_time, conf_matrix, cross_val_acc, cross_val_std


data = pd.read_csv('Data/data_cleaned.csv')

# data cleaning
data = data[data['gender'] != 'U']

# data preprocessing
# removing customer and merchant columns
# there are too many unique values for each of these to account for in a model
# it could very well add a lot of noise to our model
# I may do a second implementation later encoding the customer and merchant columns
# and using some dimentionality reduction technique and see how the model performs
data.drop(['customer','merchant'], axis=1, inplace=True)

# encoding variables
data = pd.get_dummies(data, columns=['category'])
data['gender'] = data['gender'].map({'M':0, 'F':1})

# getting X and y variables
y = data.iloc[:,4].values
X = data.drop(['fraud'], axis=1).values

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# standardization
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Logistic Regression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(sc.transform(X_test))
cross_val = cross_val_score(classifier, X_train, y_train, scoring='accuracy', cv=10)
print('\nAvg Cross Val Score: ', cross_val.mean())
print('Cross Val Std: ', cross_val.std())
print('Accuracy score: ', accuracy_score(y_test, y_pred))
print('Recall score: ', recall_score(y_test, y_pred))
print('Precision score: ', precision_score(y_test, y_pred))
print('F1 score: ', f1_score(y_test, y_pred))
print('Confusion Matrix: \n', confusion_matrix(y_test, y_pred))


# Just the standard logistic regression model without tuning or
# dimentionality reduction is giving an accuracy of 90.5%

# Perceptron
classifier = Perceptron(alpha = 0.01)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(sc.transform(X_test))
cross_val = cross_val_score(classifier, X_train, y_train, scoring='accuracy', cv=10)
print('\nAvg Cross Val Score: ', cross_val.mean())
print('Cross Val Std: ', cross_val.std())
print('Accuracy score: ', accuracy_score(y_test, y_pred))
print('Recall score: ', recall_score(y_test, y_pred))
print('Precision score: ', precision_score(y_test, y_pred))
print('F1 score: ', f1_score(y_test, y_pred))
print('Confusion Matrix: \n', confusion_matrix(y_test, y_pred))
# Mind-blowing 98.9% accuracy

# SVM
classifier = SVC(kernel='rbf')
classifier.fit(X_train, y_train)
y_pred = classifier.predict(sc.transform(X_test))
cross_val = cross_val_score(classifier, X_train, y_train, scoring='accuracy', cv=10)
print('\nAvg Cross Val Score: ', cross_val.mean())
print('Cross Val Std: ', cross_val.std())
print('Accuracy score: ', accuracy_score(y_test, y_pred))
print('Recall score: ', recall_score(y_test, y_pred))
print('Precision score: ', precision_score(y_test, y_pred))
print('F1 score: ', f1_score(y_test, y_pred))
print('Confusion Matrix: \n', confusion_matrix(y_test, y_pred))
# Wow SVC did terribly 47.1%

# K Nearest Neighbors
classifier = KNeighborsClassifier(n_neighbors=5, algorithm='auto',metric='minkowski', n_jobs=-1)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(sc.transform(X_test))
cross_val = cross_val_score(classifier, X_train, y_train, scoring='accuracy', cv=10)
print('\nAvg Cross Val Score: ', cross_val.mean())
print('Cross Val Std: ', cross_val.std())
print('Accuracy score: ', accuracy_score(y_test, y_pred))
print('Recall score: ', recall_score(y_test, y_pred))
print('Precision score: ', precision_score(y_test, y_pred))
print('F1 score: ', f1_score(y_test, y_pred))
print('Confusion Matrix: \n', confusion_matrix(y_test, y_pred))
# 98.8% accuracy

# Decision Trees
classifier = DecisionTreeClassifier(criterion='gini', random_state=0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(sc.transform(X_test))
cross_val = cross_val_score(classifier, X_train, y_train, scoring='accuracy', cv=10)
print('\nAvg Cross Val Score: ', cross_val.mean())
print('Cross Val Std: ', cross_val.std())
print('Accuracy score: ', accuracy_score(y_test, y_pred))
print('Recall score: ', recall_score(y_test, y_pred))
print('Precision score: ', precision_score(y_test, y_pred))
print('F1 score: ', f1_score(y_test, y_pred))
print('Confusion Matrix: \n', confusion_matrix(y_test, y_pred))
# Terrible accuracy 65.4%

# Random Forrest
classifier = RandomForestClassifier(n_estimators = 100, criterion='gini', n_jobs=-1, random_state=0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(sc.transform(X_test))
cross_val = cross_val_score(classifier, X_train, y_train, scoring='accuracy', cv=10)
print('\nAvg Cross Val Score: ', cross_val.mean())
print('Cross Val Std: ', cross_val.std())
print('Accuracy score: ', accuracy_score(y_test, y_pred))
print('Recall score: ', recall_score(y_test, y_pred))
print('Precision score: ', precision_score(y_test, y_pred))
print('F1 score: ', f1_score(y_test, y_pred))
print('Confusion Matrix: \n', confusion_matrix(y_test, y_pred))
# Amazing accuracy of 98.7%

# Gradient Boosting Classifier
classifier = GradientBoostingClassifier(loss='log_loss',learning_rate=0.1, n_estimators=100, random_state=0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(sc.transform(X_test))
cross_val = cross_val_score(classifier, X_train, y_train, scoring='accuracy', cv=10)
print('\nAvg Cross Val Score: ', cross_val.mean())
print('Cross Val Std: ', cross_val.std())
print('Accuracy score: ', accuracy_score(y_test, y_pred))
print('Recall score: ', recall_score(y_test, y_pred))
print('Precision score: ', precision_score(y_test, y_pred))
print('F1 score: ', f1_score(y_test, y_pred))
print('Confusion Matrix: \n', confusion_matrix(y_test, y_pred))
# 98.8 accuracy
