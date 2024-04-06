import pandas as pd
from datetime import datetime
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
    tp = []
    fp = []
    tn = []
    fn = []
    
    for model in models:
        start_time = datetime.now()
        model.fit(X_train, y_train)
        end_time = datetime.now()
        
        y_pred = model.predict(X_test)
        cross_score = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=10)
        cm = confusion_matrix(y_test, y_pred)
        
        cross_val_acc.append(round(cross_score.mean(),4))
        cross_val_std.append(round(cross_score.std(),6))
        accuracy.append(round(accuracy_score(y_test, y_pred),4))
        recall.append(round(recall_score(y_test, y_pred),4))
        precision.append(round(precision_score(y_test, y_pred),4))
        f1Score.append(round(f1_score(y_test, y_pred),4))
        tp.append(cm[1,1])
        fp.append(cm[0,1])
        tn.append(cm[0,0])
        fn.append(cm[1,0])
        train_time.append(end_time - start_time)
    
    return accuracy, recall, precision, f1Score, train_time, cross_val_acc, cross_val_std, tp, fp, tn, fn


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
lr = LogisticRegression()

# Perceptron
percept = Perceptron(alpha = 0.01)

# SVM
svc = SVC(kernel='rbf')

# K Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors=5, algorithm='auto',metric='minkowski', n_jobs=-1)

# Decision Trees
dt = DecisionTreeClassifier(criterion='gini', random_state=0)

# Random Forrest
rf = RandomForestClassifier(n_estimators = 100, criterion='gini', n_jobs=-1, random_state=0)

# Gradient Boosting Classifier
gbc = GradientBoostingClassifier(loss='log_loss',learning_rate=0.1, n_estimators=100, random_state=0)


# training all models and getting their metric scores
models = [lr, percept, svc, knn, dt, rf, gbc]
model_names = ['Logistic Regression','Perceptron','SVC','KNN','Decision Trees','Random Forest','Gradient Boosting Classifier']
accuracy, recall, precision, f1Score, train_time, cross_val_acc, cross_val_std, tp, fp, tn, fn = accuracy_scores(models)
accuracy_matrix = pd.DataFrame({
    'Accuracy':accuracy,
    'Recall':recall,
    'Precision':precision,
    'F1 Score':f1Score,
    'Cross Validation Accuracy':cross_val_acc,
    'Cross Validation Std':cross_val_std,
    'True Positive':tp,
    'False Positive':fp,
    'True Negative':tn,
    'False Negative':fn,
    'Training Time':train_time},index=model_names)

data.to_csv('Data/data_preprocessed.csv', index=False)
accuracy_matrix.to_csv('Data/accuracy_scores_initial_models.csv')

