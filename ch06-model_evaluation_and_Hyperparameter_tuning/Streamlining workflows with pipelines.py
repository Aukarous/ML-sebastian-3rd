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


def optimize_precision_and_recall_of_classification_model():
    print('Precision:%.3f' % precision_score(y_true=y_test, y_pred=y_pred))
    print('Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_pred))
    print('F1:%.3f' % f1_score(y_true=y_test, y_pred=y_pred))
    scorer = make_scorer(f1_score, pos_label=0)
    c_gamma_range = [0.01, 0.1, 1.0, 10.0]
    pipe_svc = make_pipeline(StandardScaler(), SVC(random_state=1))
    pipe_svc.fit(X_train, y_train)
    param_grid = [{'svc_C': c_gamma_range, 'svc_kernel': ['linear']},
                  {'svc_C': c_gamma_range, 'svc_kernel': ['rbf']}]
    grid_search = GridSearchCV(estimator=pipe_svc, param_grid=param_grid, scoring=scorer, cv=10, n_jobs=-1)
    grid_search_fitted = grid_search.fit(X_train, y_train)
    print('grid search best score: {}'.format(grid_search_fitted.best_score_))
    print('grid search best params: {}'.format(grid_search_fitted.best_params_))

    # plotting a receiver operating characteristic, scipy.__version__ == '1.6.2'
    pipeline_lr = make_pipeline(StandardScaler(), PCA(n_components=2), LogisticRegression(penalty='l2', random_state=1,
                                                                                          solver='lbfgs', C=100
                                                                                          ))
    X_train2 = X_train[:, [4, 14]]
    cv = list(StratifiedKFold(n_splits=3).split(X_train, y_train))
    fig = plt.figure(figsize=(7, 5))
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    all_tpr = []

    for i, (train, test) in enumerate(cv):
        probas = pipeline_lr.fit(X_train2[train], y_train[train].predict_proba(X_train2[test]))
        fpr, tpr, thresholds = roc_curve(y_train[test], probas[:, 1], pos_label=1)
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label='ROC fold %d (area=%0.2f') % (i + 1, roc_auc)

    plt.plot([0, 1], [0, 1], linestyle='--', color=(0.6, 0.6, 0.6), label='Random guessing')
    mean_tpr /= len(cv)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, 'k--', label='Mean ROC (area=%0.2f)' % mean_auc, lw=2)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()

    # The scoring metrics for multiclass classification
    pre_scorer = make_scorer(score_func=precision_score, pos_label=1, greater_is_better=True, average='micro')

    # dealing with class imbalance
    X_imb = np.vstack((X[y == 0], X[y == 1][:40]))
    y_imb = np.vstack((y[y == 0], y[y == 1][:40]))

    y_pred_t = np.zeros(y_imb.shape[0])
    np.mean(y_pred_t == y_imb) * 100
    print('Number of class 1 examples before:', X_imb[y_imb == 1].shape[0])

    X_upsampled, y_upsampled = resample(X_imb[y_imb == 1], y_imb[y_imb == 1], replace=True, n_samples=X_imb[y_imb == 0].
                                        shape[0], random_state=123)
    print('Number of class 1 examples after:', X_upsampled.shape[0])

    X_bal = np.vstack((X[y == 0], X_upsampled))
    y_bal = np.vstack((y[y == 0], y_upsampled))

    y_pred_t = np.zeros(y_bal.shape[0])
    np.mean(y_pred_t == y_bal) * 100


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
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    plt.plot(param_range, train_mean, color='blue', marker='o', markersize=5, label='Training accuracy')
    plt.fill_between(param_range, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
    plt.plot(param_range, test_mean, color='green', linestyle='--', marker='s', markersize=5,
             label='Validation accuracy')
    plt.fill_between(param_range, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
    plt.grid()
    plt.xscale('log')
    plt.legend(loc='lower right')
    plt.xlabel('Parameter C')
    plt.ylabel('Accuracy')
    plt.ylim([0.8, 1.0])
    plt.tight_layout()
    plt.show()


def fine_tuning_ml_via_grid_search():
    # tuning hyper_parameters via grid search
    pipe_svc = make_pipeline(StandardScaler(), SVC(random_state=1))
    param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    param_grid = [{'svc_C': param_range, 'svc_kernel': ['linear']},
                  {'svc_C': param_range, 'svc_gamma': param_range, 'svc_kernel': ['rbf']}]
    gs = GridSearchCV(estimator=pipe_svc, param_grid=param_grid, scoring='accuracy', refit=True, cv=10, n_jobs=-1)
    gs = gs.fit(X_train, y_train)
    print(gs.best_score_)
    print(gs.best_params_)

    clf = gs.best_estimator_
    """note that we do not need to refit the classifier because this is done automatically via refit=True."""
    # clf.fit(X_train,y_train)
    print('Test accuracy:%.3f' % clf.score(X_test, y_test))

    # algorithm selection with nested cross-validation
    gs = GridSearchCV(estimator=pipe_svc, param_grid=param_grid, scoring='accuracy', cv=2)
    scores_1 = cross_val_score(gs, X_train, y_train, scoring='accuracy', cv=5)
    print('CV accuracy: {:.3f} +/- {:.3f}'.format(np.mean(scores_1), np.std(scores_1)))

    gs = GridSearchCV(eatimator=DecisionTreeClassifier(random_state=0),
                      param_grid=[{'max_depth': [1, 2, 3, 4, 5, 6, 7, None]}],
                      scoring='accuracy',
                      cv=2)
    scores_2 = cross_val_score(gs, X_train, y_train, scoring='accuracy', cv=5)
    print('CV accuracy:{:.3f}+/-{:.3f}'.format(np.mean(scores_2), np.std(scores_2)))


def different_performance_evaluation_metrics():
    # Reading a confusion matrix
    pipe_svc = make_pipeline(StandardScaler(), SVC(random_state=1))
    pipe_svc.fit(X_train, y_train)
    y_predi = pipe_svc.predict(X_test)
    conf_matr = confusion_matrix(y_true=y_test, y_pred=y_predi)
    print(conf_matr)
    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    ax.matshow(conf_matr, cmap=plt.cm.Blues, alpha=0.3)  # Blues?
    for i in range(conf_matr.shape[0]):
        for j in range(conf_matr.shape[1]):
            ax.text(x=j, y=i, s=conf_matr[i, j], va='center', ha='center')

    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.tight_layout()
    plt.show()
    # Additional Note: Remember that we previously encoded the class labels so that *malignant* examples are the
    # "positive" class (1), and *benign* examples are the "negative" class (0):

    tmp_le = LabelEncoder()
    tmp_le.transform(['M', 'B'])
    conf_matr2 = confusion_matrix(y_true=y_test, y_pred=y_pred)
    print(conf_matr2)

    """
    # or, print the confusion matrix like so:
    conf_matr2 = confusion_matrix(y_true=y_test, y_pred=y_pred, labels=[1, 0])
    print(conf_matr2)
    """
    """
    假设 class 1 （恶性）是本示例中的正类，我们的模型将属于 class 0 （真负）的示例中的 71 个和属于 class 1 （真正）的 40 
    个示例进行了正确分类。然而，模型也错误地将0类的1个示例误划为class 1（假阳），并将2个示例预测为良性的，尽管它是恶性肿瘤（假阴）。
    """
    optimize_precision_and_recall_of_classification_model()


if __name__ == "__main__":
    df = pd.read_csv("wdbc.data", header=None)
    X = df.loc[:, 2:].values
    y = df.loc[:, 1].values
    le = LabelEncoder()
    y = le.fit_transform(y)
    print(le.classes_)
    le.transform(['M', 'B'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=1)

    # combining transformers and estimators in a pipeline
    pipe_lr = make_pipeline(StandardScaler(), PCA(n_components=2),
                            LogisticRegression(random_state=1, solver='lbfgs'))
    pipe_lr.fit(X_train, y_train)
    y_pred = pipe_lr.predict(X_test)
    print('Test Accuracy: %.3f' % pipe_lr.score(X_test, y_test))

    # using k-fold cross validation to assess model performance
    # The holdout method
    # K-fold cross-validation
    k_fold = StratifiedKFold(n_splits=10).split(X_train, y_train)
    scores = []
    for k, (train, test) in enumerate(k_fold):
        pipe_lr.fit(X_train[train], y_train[train])
        score = pipe_lr.score(X_train[test], y_train[test])
        scores.append(score)
        print('Fold: %2d, Class dist.: %s, Acc:%.3f' % (k + 1, np.bincount(y_train[train]), score))

        scores = cross_val_score(estimator=pipe_lr, X=X - train, y=y_train, cv=10, n_jobs=1)
        print('CV accuracy scores:%s' % scores)
        print(f'CV accuracy:{np.mean(scores):.3f}+/-{np.std(scores):.3f}')

    debug_with_learning_curves()

    fine_tuning_ml_via_grid_search()

    different_performance_evaluation_metrics()
