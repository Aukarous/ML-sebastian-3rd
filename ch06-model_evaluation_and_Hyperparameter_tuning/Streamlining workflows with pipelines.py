import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import validation_curve
from sklearn.model_selection import learning_curve
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import LooseVersion as Version
from scipy import __version__ as scipy_version
# from scipy import interp
from numpy import interp
from sklearn.utils import resample


def debug_with_learning_curves():
    # diagnosing bias and variance problems with learning curves
    pipe_lr = make_pipeline(StandardScaler(),
                            LogisticRegression(penalty='l2', random_state=1, solver='lbfgs', max_iter=10000))
    train_sizes, train_scores, test_scores = learning_curve(estimator=pipe_lr, X=X_train, y=y_train,
                                                            train_sizes=np.linspace(0.1, 1.0, 10), cv=10, n_jobs=-1)
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.mean(test_scores, axis=1)

    plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='Training accuracy')
    plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
    plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5,
             label='Validation accuracy')
    plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
    plt.grid()
    plt.xlabel('Number of training examples')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.ylim([0.8, 1.03])
    plt.tight_layout()
    plt.show()

    # addressing over- and underfitting with validation curves
    param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    train_scores, test_scores = validation_curve(estimator=pipe_lr,
                                                 X=X_train,
                                                 y=y_train,
                                                 param_name='logisticregression_C',
                                                 param_range=param_range,
                                                 cv=10)




if __name__ == "__main__":
    df = pd.read_csv("wdbc.data", header=None)
    X = df.loc[:, 2:].values
    y = df.loc[:, 1].values
    le = LabelEncoder()
    y = le.fit_transform(y)
    print(le.classes_)
    le.transform(['M', 'B'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=1)

    # combining transformes and estimators in a pipeline
    pipe_lr = make_pipeline(StandardScaler(), PCA(n_components=2),
                            LogisticRegression(random_state=1, solver='lbfgs'))
    pipe_lr.fit(X_train, y_train)
    y_pred = pipe_lr.predict(X_test)
    print('Test Accuracy: %.3f' % pipe_lr.score(X_test, y_test))

    # using k-fold cross validation to assess model performance
    # The holdout method
    # K-fold cross-validation
    kfold = StratifiedKFold(n_splits=10).split(X_train, y_train)
    scores = []
    for k, (train, test) in enumerate(kfold):
        pipe_lr.fit(X_train[train], y_train[train])
        score = pipe_lr.score(X_train[test], y_train[test])
        scores.append(score)
        print('Fold: %2d, Class dist.: %s, Acc:%.3f' % (k + 1, np.bincount(y_train[train]), score))

        scores = cross_val_score(estimator=pipe_lr, X=X - train, y=y_train, cv=10, n_jobs=1)
        print('CV accuracy scores:%s' % scores)
        print(f'CV accuracy:{np.mean(scores):.3f}+/-{np.std(scores):.3f}')

    debug_with_learning_curves()
