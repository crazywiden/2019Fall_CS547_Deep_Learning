import os
import time
import pickle
import random
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.utils import class_weight
import matplotlib.pyplot as plt

class time_counter(object):
    """
    a decorator to count time used for a function
    """
    def __init__(self, name):
        self.name = name

    def __call__(self, func, *args, **kwargs):
        def decorator(*args, **kwargs):
            print("start doing %s ..." % self.name)
            start = time.perf_counter()
            res = func(*args, **kwargs)
            end = time.perf_counter()
            print("time needed for %s is: %s" % (self.name, str(end-start)))
            return res
        return decorator

def random_guess(train_label, test_label):
    """
    get number of classes n from train_label
    give prediction of each class with p=1/n_class
    only for classification
    @parameters:
    train_label, test_label -- np.ndarray
    @returns:
    pred_Y
    """
    return random.choices(np.unique(train_label), k=len(test_label))

def educated_guess(train_label, test_label):
    """
    get number of classes n from train_label
    give prediction of each class with the frequency in train_label
    only for classification
    @parameters:
    train_label, test_label -- np.ndarray
    @returns:
    pred_Y
    """
    all_class = np.unique(train_label)
    N = len(train_label)
    all_p = []
    for label in all_class:
        all_p.append(len(np.where(train_label==label)[0])/N)
    return random.choices(population=all_class, weights=all_p, k=len(test_label))


def majority_guess(train_label, test_label):
    """
    get number of classes n from train_label
    guess with highest 
    only for classification
    @parameters:
    train_label, test_label -- np.ndarray
    @returns:
    pred_Y
    """
    all_class = np.unique(train_label)
    all_p = []
    N = len(test_label)
    for label in all_class:
        all_p.append(len(np.where(train_label==label)[0])/N)
    pred = all_class[np.argmax(all_p)]
    return [pred for i in range(N)]


def summary(X_train, X_test, y_train, y_test, model, benchmark=None):
    """
    summarize the performance of model on train set and test set
    also compare the performance with benchmark
    @parameters:
    X_train -- dataframe
        each column is a feature
    X_test -- dataframe
        each column is a feature
    y_train -- dataframe | np.ndarray
        if y_train is a dataframe, then regard y_train as one-hot encoded
    y_test -- dataframe | np.ndarray
        if y_test is a dataframe, then regard y_train as one-hot encoded
    model -- user-defined class
        must have a method: model.predict_classes()
    benchmark -- string
        one of ["all", "random", "educated", "majority"]
        if benchmark == "all", then output random guess, educated guess and majority guess together
    """
    if len(y_train.shape)>1 and y_train.shape[1]>1: # in this case y_train is one-hot
        y_train = y_train.argmax(axis=1)
        y_test = y_test.argmax(axis=1)

    print("==========test set performance===========")
    y_pred_test = model.predict_classes(X_test)
    print(classification_report(y_test, y_pred_test))
    print(confusion_matrix(y_test, y_pred_test))
    print("==========train set performance===========")
    y_pred_train = model.predict_classes(X_train)
    print(classification_report(y_train, y_pred_train))
    print(confusion_matrix(y_train, y_pred_train))

    if not benchmark:
        assert benchmark in ["all", "random", "educated", "majority"], TypeError("benchmark must in ['all', 'random', 'educated', 'majority']")
        print("==========%s performance===========" % benchmark)
        if benchmark in ["random", "all"]:
            y_pred_benchmark = random_guess(y_train, y_test)
            print(classification_report(y_test, y_pred_benchmark))
            print(confusion_matrix(y_test, y_pred_benchmark)) 

        if benchmark in ["educated", "all"]:
            y_pred_benchmark = educated_guess(y_train, y_test)
            print(classification_report(y_test, y_pred_benchmark))
            print(confusion_matrix(y_test, y_pred_benchmark)) 

        if benchmark in ["educated", "all"]:
            y_pred_benchmark = majority_guess(y_train, y_test)
            print(classification_report(y_test, y_pred_benchmark))
            print(confusion_matrix(y_test, y_pred_benchmark)) 



def gen_sample_weight(label, method="balanced"):
    """
    @parameters:
    label -- np.ndarray
        either one hot encoded or just original
    method -- string or dict
        If 'balanced', class weights will be given by n_samples / (n_classes * np.bincount(y)). 
        If a dictionary is given, keys are classes and values are corresponding class weights. 
        If None is given, the class weights will be uniform.
    @returns:
    balanced_weights -- np.ndarray
        length is equal to the number of classes
    """
    if len(label.shape)!=1 and label.shape[1] !=1: # if label is one-hot encoded
        label = label.argmax(axis=1)
        
    class_weights = class_weight.compute_class_weight(method,
                                                 np.unique(label), label)
    
    balanced_weights = np.zeros(label.shape[0])
    for i in range(len(class_weights)):
        balanced_weights[label == i] = class_weights[i]

    return balanced_weights


def split(X, y, train_pct, encoding=None):
    """
    this function did the following things:
    1. split train and test set
    2. normalize numerical type of data 
    @parameters:
    X -- dataframe
    y -- np.ndarray | list | pd.Series
    train_pct -- float
        between 0 and 1, indicate how many percentage of data should be used as train data
    encoding -- string
        encoding method to deal with categorical data
        default is None
    @returns:
    X_train, X_test -- dataframe
    y_train, y_test -- np.ndarray
        shape could be N by 1 or N by k
        N is the number of data, k is number of class
    """
    from data_process import one_hot, normalization
    if not encoding:
        pass
    elif encoding == "one_hot":
        X = one_hot(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
        train_size=train_pct, stratify=y)
    X_train, X_test = normalization(X_train, X_test) 

    encoder = LabelBinarizer()
    y_train = encoder.fit_transform(y_train)
    y_test = encoder.fit_transform(y_test)
    return X_train, X_test, y_train, y_test


def label_distribution(label):
    """
    @parameters:
    label -- list or np.ndarray
    at least containing label, ev_time
    @return:
    dist -- dataframe
    count each class and corresponding frequency
    """
    all_cate, cnts = np.unique(label, return_counts=True)
    freq = cnts/len(label)
    return pd.DataFrame({"label":all_cate, "count":cnts, "frequency":freq})


def metric_from_confusion_mat(confusion_mat, name=None):
    """
    @parameters:
    confusion_mat -- np.ndarray
    name -- list of string
    @return:
    metrics -- dataframe
    """
    if not name:
        name = np.arange(len(confusion_mat))
        
    metrics = pd.DataFrame(columns=["precision", "recall", "f1score"])
    print(confusion_mat)
    for i in range(len(confusion_mat)):
        TP = confusion_mat[i][i]
        FP = sum(confusion_mat[:,i]) - TP
        FN = sum(confusion_mat[i,:]) - TP
        tmp = pd.DataFrame({"precision": TP/(TP + FP), 
                            "recall": TP/(TP + FN), 
                            "f1score": 2*TP/(2*TP + FP + FN)}, index=[name[i]])
        metrics = pd.concat([metrics, tmp])
    return metrics

def binary_ROC(estimator, test_X, test_Y, name=None):
    """
    plot ROC curve for binary classification
    save the plot in current working directory
    @parameters:
    estimator  -- model
        estimator must be trained and has a method: estimator.predict_proba()
    test_X -- dataframe
    test_y -- n by 1 vector
        each element is a class, only had two choices
    @returns:
    fpr -- np.ndarray
        false positive rates
        fpr[i] is the false positive rate of predictions with score >= threshold[i].
    tpr -- np.ndarray
        true positive rates
        tpr[i] is the true positive rate of predictions with score >= threshold[i].
    theshold -- np.ndarray
        Decreasing thresholds on the decision function used to compute fpr and tpr. 
        thresholds[0] represents no instances being predicted and is arbitrarily set to max(y_score) + 1.
    """
    y_score = model.predict_proba(test_X)
    preds = probs[:,1]
    fpr, tpr, threshold = roc_curve(test_Y, preds)
    roc_auc = auc(fpr, tpr)

    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    if name:
        title = 'ROC for class %s_%s' % (all_class[i], name)
        plt.title(title)
    else:
        title = 'ROC for class %s' % all_class[i]
        plt.title(title)
    plt.legend(loc="lower right")
    plt.savefig(title+".png")

    return fpr, tpr, threshold

def multi_ROC(estimator, test_X, test_Y, name=None):
    """
    plot ROC curve for each label category
    save the plot in current working directory
    @parameters:
    estimator  -- model
        estimator must be trained
    test_X -- dataframe
    test_y -- n by 1 vector
        each element is a class
    @returns:
    fpr -- dictionary
        each key is the name of a class
        each value is an array, represents the false positive rate of given class
        more details see docstring of binary_ROC
    tpr -- dictionary
        each key is the name of a class
        each value is an array, represents the true positive rate of given class
        more details see docstring of binary_ROC
    thres -- dictionary
        each key is the name of a class
        each value is an array, represents the threshold of given class
        more details see docstring of binary_ROC
    """
    all_class = np.unique(test_Y)
    n_classes = len(np.unique(test_Y))
    y = label_binarize(test_Y, classes=all_class)
    y_score = estimator.predict_proba(test_X)
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    thres = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], thres[i] = roc_curve(y[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Plot of a ROC curve for a specific class
    for i in range(n_classes):
        plt.figure()
        plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        if name:
            title = 'ROC for class %s_%s' % (all_class[i], name)
            plt.title(title)
        else:
            title = 'ROC for class %s' % all_class[i]
            plt.title(title)
        plt.legend(loc="lower right")
        plt.savefig(title+".png")
    return fpr, tpr, thres



def reduce_memory(df):
    """
    reduce dataframe memory by changing datatype 
    if column datatype is float or int, change to nearest datatype
    if column datatype is string, change to category
    @parameter:
    df -- dataframe
    @return:
    df -- dataframe
    """
    init_memory = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(init_memory))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_memory = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_memory))
    print('Decreased by {:.1f}%'.format(100 * (init_memory - end_memory) / end_memory))
    
    return df