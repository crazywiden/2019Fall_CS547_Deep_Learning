import os
import time
import pickle
import numpy as np 
import pandas as pd 
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import roc_curve, auc
from utilis import label_distribution, multi_ROC, binary_ROC, metric_from_confusion_mat


class cross_validation:
    """
    for cross validation
    """
    def __init__(self, X, y):
        """
        @parameters:
        X -- dataframe
        y -- list | np.ndarray
        """
        self.X = X
        self.y = y
        self.num_class = len(np.unique(y))
        self.train_idx = []
        self.test_idx = []
        self.useful_feature = [] # a list of dataframe, each dataframe has two columns: name, score
        self.estimators = []
        self.report_is = {}
        self.report_os = {}
        self.confusion_mat_is = {}
        self.confusion_mat_os = {}
        self.metrics_is = pd.DataFrame(columns=["precision", "recall", "f1score"])
        self.metrics_os = pd.DataFrame(columns=["precision", "recall", "f1score"])
    
    
    def train(self, estimator, show_distr=True, whiten=False, verbose=False):
        """
        @parameters:
        estimator -- user-defined class
            at least has method:
            estimator.fit(X_train, y_train)
            estimator.predict(X_test)
        """

        label_value = self.y
        all_cate = len(np.unique(label_value))
        confusion_mat_is_all = np.zeros((all_cate, all_cate))
        confusion_mat_os_all = np.zeros((all_cate, all_cate))
        for i in range(len(self.train_idx)):
            
            train_X = self.X.iloc[self.train_idx[i], :]
            train_Y = label_value[self.train_idx[i]]
            
            
            if show_distr:
                print("Distribution of labels:")
                print(label_distribution(train_Y))
                
            test_X = self.X.iloc[self.test_idx[i], :]
            test_Y = label_value[self.test_idx[i]]

            if whiten:
                avg, dev = np.mean(train_X), np.std(train_X)
                train_X = (train_X - avg)/dev
                test_X = (test_X - avg)/dev
           
            
            train_start = time.perf_counter()
            print(train_Y.shape)
            estimator.fit(train_X, train_Y)
            self.estimators.append(estimator)            
            train_end = time.perf_counter()

                
            if verbose:
                print("==============training session: %s/%s=================" % (i, len(self.train_idx)))
                print("training time:", train_end - train_start)
                
            rf_pred_os = estimator.predict(test_X) # out of sample
            rf_pred_is = estimator.predict(train_X) # in sample
            
            p, r, f, _ = score(test_Y, rf_pred_os, average='macro')
            self.metrics_os = pd.concat([self.metrics_os, 
                                         pd.DataFrame({"precision":p, "recall":r,
                                                              "f1score":f}, index=[i])])
            p, r, f, _ = score(train_Y, rf_pred_is, average='macro')
            self.metrics_is = pd.concat([self.metrics_is, 
                                         pd.DataFrame({"precision":p, "recall":r,
                                                              "f1score":f}, index=[i])])
            if verbose:
                print("=========out of sample performance========")
                self.report_os[str(i)] = classification_report(test_Y, rf_pred_os)
                print(classification_report(test_Y, rf_pred_os))
                self.confusion_mat_os[str(i)] = confusion_matrix(test_Y, rf_pred_os)
                confusion_mat_os_all += confusion_matrix(test_Y, rf_pred_os)
                print(confusion_matrix(test_Y, rf_pred_os))
                print("=========in sample performance========")
                self.report_is[str(i)] = classification_report(train_Y, rf_pred_is)
                print(classification_report(train_Y, rf_pred_is))
                self.confusion_mat_is[str(i)] = confusion_matrix(train_Y, rf_pred_is)
                confusion_mat_is_all += confusion_matrix(train_Y, rf_pred_is)
                print(confusion_matrix(train_Y, rf_pred_is))
                
        # get all micro average merics
        print("==============================summary=================================")
        micro_metric_is = metric_from_confusion_mat(confusion_mat_is_all, name=[-1,0,1])
        micro_metric_os = metric_from_confusion_mat(confusion_mat_os_all, name=[-1,0,1])
        print("out of sample performance all")
        print(micro_metric_os)
        print("in sample performance all")
        print(micro_metric_is)
        print("=====================================================================")
        return self.estimators
            
    
    def save_res(self, path, name):
        if not os.path.exists(path):
            os.makedirs(path)
        all_metrics = {"feature_impt": self.useful_feature,
                       "report_is":self.report_is,
                       "report_os":self.report_os,
                       "confusion_mat_is":self.confusion_mat_is,
                       "confusion_mat_os":self.confusion_mat_os,
                       "metric_is":self.metrics_is,
                       "metric_os":self.metrics_os}
        
        with open(os.path.join(path, name),"wb") as f:
            pickle.dump(all_metrics, f)
        txt_name = os.path.splitext(name)[0] + ".txt"
        
        num_session = len(self.report_is.keys())
        with open(os.path.join(path, txt_name), "a") as file:
            for i in range(num_session):
                file.write("==============training session: %s/%s=================" % (i, num_session))
                file.write("\n")
                file.write("=============in sample===============")
                file.write("\n")
                file.write(self.report_is[str(i)])
                file.write("\n")
                file.write(np.array2string(self.confusion_mat_is[(str(i))], separator=', '))
                file.write("\n")
                file.write("=============out of sample===============")
                file.write("\n")
                file.write(self.report_os[str(i)])
                file.write("\n")
                file.write(np.array2string(self.confusion_mat_os[(str(i))], separator=', '))
                file.write("\n")