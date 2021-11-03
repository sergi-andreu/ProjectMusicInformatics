import numpy as np

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold

import warnings
from sklearn.exceptions import ConvergenceWarning

class LearningMachine:
    
    def __init__(self, data_class):
        
        self.name = None
        
        self.input = data_class
        
        self.labels = data_class.labels
        
        self.methods = ["LDA", "RDA"]
        
        self.best_parameters = {}

        self.metrics = {}
        self.parameters = {}
        
        self.fitting_parameters = {}
        
        self.model = None

        self.run = 0
        self.run_cross_validation = 0


        self.parameter_labels = {"LDA" : "Solver", "QDA" : "Solver", "KNearestNeighbors" : "N neighbors",
                                "LogisticRegression" : "Penalty", "NN" : "Hidden layers",
                                "SVM" : "Kernel",
                                "DTs" : "Criterion", "RandomForest" : "N estimators", "bagging" : "N estimators"}


        self.evaluation_labels = {"accuracy" : ["Train Accuracy", "Test Accuracy"],
                            "recall" : ["Train Recall", "Test Recall"],
                            "precision" : ["Train Precision", "Test Precision"],
                            "f1" : ["Train F1 Score", "Test F1 Score"]}
        
    def __str__(self):
        return str(self.name)
    
        
    def _fit(self):
        raise ValueError("The _fit method for the given subclass is not implemented.")
        pass

    def _get_fitting_parameters(self):
        return 0
    
    def _predict(self):
        raise ValueError("The _predict method for the given subclass is not implemented.")
        pass
        

    def _evaluate(self, X=None, Y=None, measure=None):

        pass

    def _get_confusion_matrix(self, X=None, Y=None):

        try:
            if X==None:
                X = self.input.df
        except: pass

        try:
            if Y==None:
                Y = self.input.labels
        except: pass

        prediction = self._predict(X)

        cm = confusion_matrix(Y, prediction)

        return cm

    def _compute_metrics(self, X=None, Y=None, compute_accuracy=True, compute_recall=True, compute_precision=True, compute_F1=True):

        metrics = {}
        cm = np.array(self._get_confusion_matrix(X=X, Y=Y), dtype=np.int64)

        if compute_accuracy:
            acc = (cm[0][0] + cm[1][1])/np.sum(cm)
            metrics["accuracy"] = acc

        if compute_recall:
            if cm[0][0] + cm[0][1] != 0.0 : rec = (cm[0][0]) / (cm[0][0] + cm[0][1])
            else: rec = np.nan
            metrics["recall"] = rec

        if compute_precision:
            if cm[0][0] + cm[1][0] != 0.0 : pre = (cm[0][0]) / (cm[0][0] + cm[1][0])
            else: pre = np.nan
            metrics["precision"] = pre

        if compute_F1:
            if pre + rec != 0 : fscore = (2*pre*rec)/(pre+rec)
            else: fscore = np.nan
            metrics["F1"] = fscore

        return metrics


    def train_once(self, fraction_test_data=0.1, shuffle=False):

        total_input = self.input.df
        total_labels = self.input.labels

        n_data = len(self.input.df)
        n_test_data = int(fraction_test_data * n_data)

        if shuffle:
            idx = np.random.permutation(n_data)
            total_input, total_labels = total_input.reindex(idx), total_labels.reindex(idx)

        train_df = total_input[:n_data-n_test_data]
        train_labels = total_labels[:n_data-n_test_data]
        
        test_df = total_input[n_data - n_test_data:]
        test_labels = total_labels[n_data - n_test_data:]

        self._fit(input=train_df, labels=train_labels)
        
        training_sc = self._evaluate(X=train_df, Y=train_labels, measure="accuracy")
        test_sc = self._evaluate(X=test_df, Y=test_labels, measure="accuracy")

        self.metrics["Training_Accuracy_"+str(self.run)] = training_sc
        self.metrics["Test_Accuracy_"+str(self.run)] = test_sc

        print(self._compute(X=train_df, Y=train_labels, accuracy=True))

        self.run += 1


    def cross_validation(self, k=3, shuffle=False, compute_accuracy=True, compute_recall=True, compute_precision=True, compute_F1=True):


        kfold_train_metrics = []
        kfold_test_metrics = []

        total_input = self.input.df

        total_labels = self.input.labels

        n_data = len(self.input.df)

        if shuffle:
            idx = np.random.permutation(n_data)
            total_input, total_labels = total_input.reindex(idx), total_labels.reindex(idx)

        cv = KFold(n_splits=k)

        try: 
            if self.name == "LDA" or self.name == "QDA": 
                self._fit()
        except: pass

        for train_index, test_index in cv.split(total_input):

            train_df, train_labels = total_input.iloc[train_index], total_labels.iloc[train_index]
            test_df, test_labels = total_input.iloc[test_index], total_labels.iloc[test_index]

            mean = train_df.mean()
            std = train_df.std()

            train_df = (train_df - mean) / std
            test_df = (test_df - mean) / std

            self._fit(input=train_df, labels=train_labels, print_message=False)

            kfold_train_metrics.append(self._compute_metrics(X=train_df, Y=train_labels, 
                                                            compute_accuracy=compute_accuracy, compute_recall=compute_recall,
                                                            compute_precision=compute_precision, compute_F1=compute_F1))
            kfold_test_metrics.append(self._compute_metrics(X=test_df, Y=test_labels, 
                                                            compute_accuracy=compute_accuracy, compute_recall=compute_recall,
                                                            compute_precision=compute_precision, compute_F1=compute_F1))
            
        
        assert len(kfold_train_metrics) == len(kfold_test_metrics)

        L = len(kfold_train_metrics)

        kfold_metrics = {}

        if compute_accuracy: 
            train_acc_list = [kfold_train_metrics[m]["accuracy"] for m in range(L)]
            test_acc_list = [kfold_test_metrics[m]["accuracy"] for m in range(L)]

            train_acc = [np.nanmean(train_acc_list), np.nanstd(train_acc_list)]
            test_acc = [np.nanmean(test_acc_list), np.nanstd(test_acc_list)]

            kfold_metrics["Train Accuracy"] = train_acc
            kfold_metrics["Test Accuracy"] = test_acc

            #self.metrics[self.run_cross_validation] = {"Train Accuracy": train_acc, "Test Accuracy": test_acc}


        if compute_recall:
            train_rec_list = [kfold_train_metrics[m]["recall"] for m in range(L)]
            test_rec_list = [kfold_test_metrics[m]["recall"] for m in range(L)]

            train_rec = [np.nanmean(train_rec_list), np.nanstd(train_rec_list)]
            test_rec = [np.nanmean(test_rec_list), np.nanstd(test_rec_list)]

            kfold_metrics["Train Recall"] = train_rec
            kfold_metrics["Test Recall"] = test_rec

            #self.metrics[self.run_cross_validation] = {"Train Recall": train_rec, "Test Recall": test_rec}


        if compute_precision:
            train_prec_list = [kfold_train_metrics[m]["precision"] for m in range(L)]
            test_prec_list = [kfold_test_metrics[m]["precision"] for m in range(L)]

            train_prec = [np.nanmean(train_prec_list), np.nanstd(train_prec_list)]
            test_prec = [np.nanmean(test_prec_list), np.nanstd(test_prec_list)]

            kfold_metrics["Train Precision"] = train_prec
            kfold_metrics["Test Precision"] = test_prec

            #self.metrics[self.run_cross_validation] = {"Train Precision": train_prec, "Test Precision": test_prec}


        if compute_F1:
            train_F1_list = [kfold_train_metrics[m]["F1"] for m in range(L)]
            test_F1_list = [kfold_test_metrics[m]["F1"] for m in range(L)]

            train_F1 = [np.nanmean(train_F1_list), np.nanstd(train_F1_list)]
            test_F1 = [np.nanmean(test_F1_list), np.nanstd(test_F1_list)]

            kfold_metrics["Train F1 Score"] = train_F1
            kfold_metrics["Test F1 Score"] = test_F1

            #self.metrics[self.run_cross_validation] = {"Train F1 score": train_F1, "Test F1 score": test_F1}

        self.metrics[self.run_cross_validation] = kfold_metrics
        self.fitting_parameters[self.run_cross_validation] = self._get_fitting_parameters()

        self.run_cross_validation += 1

        
    def parameter_search(self, parameter_list, k=3, parameter_label = None, plot_variable = "accuracy", layers_or_width="layers"):

        assert plot_variable in ["accuracy", "recall", "precision", "f1"], "The plot_variable should be accuracy, recall, precision or f1"

        for parameter in parameter_list:
            self._change_parameters(parameter)
            self.cross_validation(k=k)

            self.parameters[self.run_cross_validation - 1] = parameter

        import matplotlib.pyplot as plt

        if parameter_label == None:
            parameter_label = self.parameter_labels[self.name]

        train_variable = self.evaluation_labels[plot_variable][0]
        val_variable = self.evaluation_labels[plot_variable][1]

        train_ = []
        train_err_ = []

        test_ = []
        test_err_ = []

        for i in range(self.run_cross_validation):

            train_.append(self.metrics[i][train_variable][0])
            train_err_.append(self.metrics[i][train_variable][1])

            test_.append(self.metrics[i][val_variable][0])
            test_err_.append(self.metrics[i][val_variable][1])

        ind_max = np.argmax(test_) # Index at which the chosen test metric is maximum
        self.best_parameters[plot_variable] = [ parameter_list[ind_max] , str(round(test_[ind_max], 2)) + u" \u00B1 " + str(round(test_err_[ind_max],2)) ]

        if self.name == "NN":
            if layers_or_width == "layers":
                parameter_list = [len(parameter_list[i]) for i in range(len(parameter_list))]
                
            if layers_or_width == "width":
                parameter_list = [parameter_list[i][0] for i in range(len(parameter_list))]

            if layers_or_width == "both":
                parameter_list = [str(parameter_list[i]) for i in range(len(parameter_list))]



        plt.plot(parameter_list, train_ , c="r", label="Training")
        plt.fill_between(parameter_list, np.array(train_) - np.array(train_err_),
                        np.array(train_) + np.array(train_err_), color="r", alpha=0.35)

        plt.plot(parameter_list, test_, c="b", label="Validation")
        plt.fill_between(parameter_list, np.array(test_) - np.array(test_err_),
                        np.array(test_) + np.array(test_err_), color="b", alpha=0.35)
        
        plt.xlabel(parameter_label)
        plt.ylabel(plot_variable)

        plt.show()



    def fit_with_all_data(self, visualize = True, diagonal="hist", print_metrics = True):

        self._fit()

        if print_metrics:
            metrics = self._compute_metrics()

            for key in metrics:
                print(f"{key} : {metrics[key]}")


        if visualize:
            from matplotlib import colors

            pred = self._predict()
            labels = self.input.labels

            pred_0 = (pred == 0)
            pred_1 = (pred == 1)

            labels_0 = (labels == 0)
            labels_1 = (labels == 1)

            TP = (pred_1 & labels_1)
            FP = (pred_1 & labels_0)
            TN = (pred_0 & labels_0)
            FN = (pred_0 & labels_1)

            C = 1*TP + 2*FP + 3*TN + 4*FN

            cmap = colors.ListedColormap(['g', 'k', 'b', 'r'], 4)
            
            self.input.visualize(cmap=cmap, labels=C, diagonal=diagonal)


        pass


    def fit_all_data_with_best_parameters(self, criterion="accuracy", visualize=True, diagonal="hist", print_metrics = True):

        assert self.best_parameters != None, "Please use the parameter_search functionality for finding the best parameters"

        self._change_parameters(self.best_parameters[criterion][0])

        print(f"The model has been trained using {self.best_parameters[criterion][0]}, whih maximizes the test {criterion} to {self.best_parameters[criterion][1]}")

        self.fit_with_all_data(visualize = visualize, diagonal=diagonal, print_metrics = print_metrics)

        pass


    def _get_best_parameters(self):

        dict_parameters = {}
        dict_metrics = {}
        dict_metrics_err = {}

        for key in self.evaluation_labels.keys():
            dict_parameters[key] = []
            dict_metrics[key] = []
            dict_metrics_err[key] = []

        for i in range(self.run_cross_validation):
            for key in self.evaluation_labels.keys():
                dict_parameters[key].append(self.parameters[i])
                dict_metrics[key].append(self.metrics[i][self.evaluation_labels[key][1]][0])
                dict_metrics_err[key].append(self.metrics[i][self.evaluation_labels[key][1]][1])

        
        for key in self.evaluation_labels.keys():
            ind_max = np.argmax(dict_metrics[key])

            self.best_parameters[key] = [ dict_parameters[key][ind_max], 
                                        str(round(dict_metrics[key][ind_max], 2)) + u" \u00B1 " + str(round(dict_metrics_err[key][ind_max], 2))]

        pass

    def get_table_with_results(self, show_train_results = False, hide_index_name = True):

        self._get_best_parameters()
        
        import pandas as pd
        results_df = pd.DataFrame()

        for i in range(self.run_cross_validation):
            results_df[f"{self.parameters[i]} {self.parameter_labels[self.name]}"] = [str(round(100*self.metrics[i][key][0],1)) + u"\u00B1" + str(round(100*self.metrics[i][key][1],1)) + "%" for key in self.metrics[i].keys()]

        results_df["Metrics"] = [key for key in self.metrics[i].keys()]
        results_df = results_df.set_index(["Metrics"], append=False)

        if not show_train_results:
            for idx, row in results_df.iterrows():
                if idx[:5] == "Train":
                    results_df = results_df.drop(index=idx)
            #results_df = results_df.reset_index(drop=True)

            new_keys = [key[5:] for key, row in results_df.iterrows()]

            results_df["Metrics"] = new_keys
            results_df = results_df.set_index(["Metrics"], append=False)
                

        order_labels = ["accuracy", "recall", "precision", "f1"]

        if hide_index_name:
            results_df.index.name = None

        return results_df

class LDA(LearningMachine):
    
    def __init__(self, data):
        super().__init__(data)
        self.name = "LDA"
        
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        
        self.model = LinearDiscriminantAnalysis()

        pass

    def _fit(self, input = None, labels = None, print_message=False):

        try: 
            if input == None: input = self.input.df
        except: pass

        try: 
            if labels == None: labels = self.input.labels
        except: pass

        self.model.fit(input, labels)
        
        if print_message:
            print(f"The {self.name} model has been trained on the given data")
        pass
    
    def _predict(self, X=None):
        
        """
        X is a dataframe containing the features of the samples for which we want to predict their labels
        """
        
        try:
            if X == None:
                X = self.input.df
        except:
            pass


        return self.model.predict(X)
        
        pass

    def _get_fitting_parameters(self):

        return self.model.get_params()


    def _change_parameters(self, solver):

        self.model.solver = solver

class QDA(LearningMachine):
    
    def __init__(self, data):
        super().__init__(data)
        self.name = "QDA"
        
        from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
        
        self.model = QuadraticDiscriminantAnalysis()

        pass

    def _fit(self, input = None, labels = None, print_message=False):

        warnings.filterwarnings("ignore", message="Variables are collinear")

        try: 
            if input == None: input = self.input.df
        except: pass

        try: 
            if labels == None: labels = self.input.labels
        except: pass

        self.model.fit(input, labels)
        
        if print_message:
            print(f"The {self.name} model has been trained on the given data")
        pass
    
    def _predict(self, X=None):
        
        """
        X is a dataframe containing the features of the samples for which we want to predict their labels
        """
        
        try:
            if X == None:
                X = self.input.df
        except:
            pass


        return self.model.predict(X)
        
        pass

    def _get_fitting_parameters(self):

        return self.model.get_params()

    def _change_parameters(self, solver):

        self.model.solver = solver

class KNearestNeighbors(LearningMachine):
    
    def __init__(self, data, n_neighbors = 5):
        super().__init__(data)
        self.name = "KNearestNeighbors"

        self.n_neighbors = n_neighbors
        
        from sklearn.neighbors import KNeighborsClassifier
        
        self.model = KNeighborsClassifier(n_neighbors = self.n_neighbors)

        pass

    def _change_parameters(self, n_neighbors):
        from sklearn.neighbors import KNeighborsClassifier
        self.n_neighbors = n_neighbors
        self.model = KNeighborsClassifier(n_neighbors = n_neighbors)


    def _fit(self, input = None, labels = None, print_message=False):

        try: 
            if input == None: input = self.input.df
        except: pass

        try: 
            if labels == None: labels = self.input.labels
        except: pass

        self.model.fit(input, labels)
        
        if print_message:
            print(f"The {self.name} model has been trained on the given data")
        pass
    
    def _predict(self, X=None):
        
        """
        X is a dataframe containing the features of the samples for which we want to predict their labels
        """
        
        try:
            if X == None:
                X = self.input.df
        except:
            pass


        return self.model.predict(X)
        
        pass

    def _get_fitting_parameters(self):

        return self.model.get_params()

class LogisticRegression(LearningMachine):

    
    def __init__(self, data):
        super().__init__(data)
        self.name = "LogisticRegression"
        
        from sklearn.linear_model import LogisticRegression
        
        self.model = LogisticRegression()

        pass

    def _fit(self, input = None, labels = None, print_message=False):

        warnings.filterwarnings("ignore", category=ConvergenceWarning)

        try: 
            if input == None: input = self.input.df
        except: pass

        try: 
            if labels == None: labels = self.input.labels
        except: pass

        self.model.fit(input, labels)
        
        if print_message:
            print(f"The {self.name} model has been trained on the given data")
        pass
    
    def _predict(self, X=None):
        
        """
        X is a dataframe containing the features of the samples for which we want to predict their labels
        """
        
        try:
            if X == None:
                X = self.input.df
        except:
            pass


        return self.model.predict(X)
        
        pass

    def _get_fitting_parameters(self):

        return self.model.get_params()


    def _change_parameters(self, penalty):

        self.model.l1_ratio = None

        assert penalty in ["l1", "l2", "elasticnet", "none"], "Choose a valid penalty (l1, l2, elasticnet or none)"
        self.model.penalty = penalty

        if penalty == "l1": self.model.solver = "liblinear"
        if penalty == "elasticnet" : 
            self.model.solver = "saga"
            self.model.l1_ratio = 0.5

class NeuralNetwork(LearningMachine):
    
    def __init__(self, data):
        super().__init__(data)
        self.name = "NN"
        
        from sklearn.neural_network import MLPClassifier
        
        self.model = MLPClassifier(solver="adam", hidden_layer_sizes=(1,1), alpha=0.0001, early_stopping=False)
        self.model.max_iter = 1000

        pass

    def _fit(self, input = None, labels = None, print_message=True):

        warnings.filterwarnings("ignore", category=ConvergenceWarning)

        try: 
            if input == None: input = self.input.df
        except: pass

        try: 
            if labels == None: labels = self.input.labels
        except: pass

        self.model.fit(input, labels)
        
        if print_message:
            print(f"The {self.name} model has been trained on the given data")
        pass
    
    def _predict(self, X=None):
        
        """
        X is a dataframe containing the features of the samples for which we want to predict their labels
        """
        
        try:
            if X == None:
                X = self.input.df
        except:
            pass


        return self.model.predict(X)
        
        pass

    def _get_fitting_parameters(self):

        return self.model.get_params()


    def _change_parameters(self, hidden_layers):

        self.model.hidden_layer_sizes = hidden_layers

class SVM(LearningMachine):


    
    def __init__(self, data):
        super().__init__(data)
        self.name = "SVM"
        
        from sklearn import svm
        
        self.model = svm.SVC()

        pass

    def _fit(self, input = None, labels = None, print_message=False):

        try: 
            if input == None: input = self.input.df
        except: pass

        try: 
            if labels == None: labels = self.input.labels
        except: pass

        self.model.fit(input, labels)
        
        if print_message:
            print(f"The {self.name} model has been trained on the given data")
        pass
    
    
    def _predict(self, X=None):
        
        """
        X is a dataframe containing the features of the samples for which we want to predict their labels
        """
        
        try:
            if X == None:
                X = self.input.df
        except:
            pass

        return self.model.predict(X)
        
        pass

    def _get_fitting_parameters(self):

        return self.model.get_params()


    def _change_parameters(self, kernel):

        assert kernel in ["linear", "poly", "rbf", "sigmoid"], "Choose a valid kernel (linear, poly, rbf or sigmoid)"

        self.model.kernel = kernel

class DecisionTrees(LearningMachine):

    
    def __init__(self, data):
        super().__init__(data)
        self.name = "DTs"
        
        from sklearn.tree import DecisionTreeClassifier
        
        self.model = DecisionTreeClassifier()

        pass

    def _fit(self, input = None, labels = None, print_message=False):

        try: 
            if input == None: input = self.input.df
        except: pass

        try: 
            if labels == None: labels = self.input.labels
        except: pass

        self.model.fit(input, labels)
        
        if print_message:
            print(f"The {self.name} model has been trained on the given data")
        pass
    
    
    def _predict(self, X=None):
        
        """
        X is a dataframe containing the features of the samples for which we want to predict their labels
        """
        
        try:
            if X == None:
                X = self.input.df
        except:
            pass

        return self.model.predict(X)
        
        pass

    def _get_fitting_parameters(self):

        return self.model.get_params()


    def _change_parameters(self, criterion):

        assert criterion in ["gini", "entropy"], "Choose a valid criterion (gini or entropy)"

        self.model.criterion = criterion

class RandomForest(LearningMachine):

    
    def __init__(self, data):
        super().__init__(data)
        self.name = "RandomForest"
        
        from sklearn.ensemble import RandomForestClassifier
        
        self.model = RandomForestClassifier()

        pass

    def _fit(self, input = None, labels = None, print_message=False):

        try: 
            if input == None: input = self.input.df
        except: pass

        try: 
            if labels == None: labels = self.input.labels
        except: pass

        self.model.fit(input, labels)
        
        if print_message:
            print(f"The {self.name} model has been trained on the given data")
        pass
    
    
    def _predict(self, X=None):
        
        """
        X is a dataframe containing the features of the samples for which we want to predict their labels
        """
        
        try:
            if X == None:
                X = self.input.df
        except:
            pass

        return self.model.predict(X)
        
        pass

    def _get_fitting_parameters(self):

        return self.model.get_params()


    def _change_parameters(self, n_estimators):

        self.model.n_estimators = n_estimators

class Bagging(LearningMachine):

    
    def __init__(self, data):
        super().__init__(data)
        self.name = "RandomForest"
        
        from sklearn.ensemble import BaggingClassifier
        
        self.model = BaggingClassifier()

        pass

    def _fit(self, input = None, labels = None, print_message=False):

        try: 
            if input == None: input = self.input.df
        except: pass

        try: 
            if labels == None: labels = self.input.labels
        except: pass

        self.model.fit(input, labels)
        
        if print_message:
            print(f"The {self.name} model has been trained on the given data")
        pass
    
    
    def _predict(self, X=None):
        
        """
        X is a dataframe containing the features of the samples for which we want to predict their labels
        """
        
        try:
            if X == None:
                X = self.input.df
        except:
            pass

        return self.model.predict(X)
        
        pass

    def _get_fitting_parameters(self):

        return self.model.get_params()


    def _change_parameters(self, n_estimators):

        self.model.n_estimators = n_estimators