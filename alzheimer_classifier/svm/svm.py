import pandas as pd
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from statistics import mode

from .. common.constants import PATH_TO_SELECTED_COEFFICIENTS_FOLDER

class SVM:
    
    parameters = {'kernel':('linear', 'rbf', 'sigmoid'), 'C': [1, 10, 100, 1000], 
                  'gamma': [1e-1, 1e-2, 1e-3, 1e-4, "scale"]}
    
    def __init__(
        self, 
        kernel=None, 
        c=None,
        gamma=None,
        selection_algorithm="pca",
        wavelet_level=4,
        path_to_folder=PATH_TO_SELECTED_COEFFICIENTS_FOLDER, 
        target_column="group",
        target_score="f1",
        slices=["all"],
        random_state=104729,
        models=None,
    ):
        """ Constructor: initialize an instance of the class.
        
        Args:
            patient_number (str): kernel of the SVM
            c (int): regularization parameter
            gamma (str): kernel coefficient
            selection_algorithm (str): algorithm used to select features
            wavelet_level (int): wavelet level used
            path_to_folder (str): path to coefficient files in local directory
            target_column (str): target column in saved csv
            target_score (str): target score to use for gridseacrh
            slices (list): list of slices to use to build the models (>1 if multiexpert)
            random_state (int): random state parm for train_test_split
            models (list): in case models are loaded from an external source, list of models
        """
        df_coefficients = {}
        df_target = {}
        for slice_number in slices:
            path_to_coefficients_file = f"{path_to_folder}/selected_features_" \
                                        f"{selection_algorithm}_slice_{slice_number}_" \
                                        f"wavelet_{wavelet_level}.csv"
            # Drop residual columns
            df = pd.read_csv(path_to_coefficients_file)
            df = df.drop(list(df.filter(regex='Unnamed')), axis=1)
            df_coefficients[str(slice_number)] = df.drop(target_column, axis=1)
            df_target[str(slice_number)] = df.drop(df.columns.difference([target_column]), 1)
        
        if target_score == "f1":
            score = metrics.make_scorer(metrics.f1_score, average = "weighted")
        elif target_score == "precision":
            score = metrics.make_scorer(metrics.precision_score, average = "weighted")
        elif target_score == "recall":
            score = metrics.make_scorer(metrics.recall_score, average = "weighted")
        else:
            raise Exception(f"Error: target_score={target_score} not available")

        self.df_coefficients = df_coefficients
        self.df_target = df_target
        self.score = score
        self.target_column = target_column
        self.kernel = kernel
        self.c = c
        self.gamma = gamma
        self.models = {} if not models else models
        self.random_state = random_state
    
    def find_best_paremeters(self, print_results=True, train_size=0.75, slice_number=None):
        """Finds the best parameters to build and train the model
        
           Args:
               print_results (bool): True if results should be printed
               train_size (float): 0-1 percentage of sample to use to 
                                   train the model
        """
        if not slice_number:
            slice_number = list(self.df_coefficients.keys())[0]

        X_train, X_test, Y_train, Y_test = train_test_split(
            self.df_coefficients[str(slice_number)], 
            self.df_target[str(slice_number)], 
            train_size=train_size,
            random_state=self.random_state, 
            stratify=self.df_target[str(slice_number)]
        )
        print("Getting scores...")
        svc = SVC()
        clf = GridSearchCV(estimator=svc, param_grid=self.parameters, scoring=self.score)
        clf.fit(X_train, Y_train.values.ravel())
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        y_true, y_pred = Y_test, clf.predict(X_test)

        
        if print_results:
            print("Best parameters set found on development set:")
            print()
            print(clf.best_params_)
            print()
            print("Grid scores on development set:")
            print()
            for mean, std, params in zip(means, stds, clf.cv_results_['params']):
                print("%0.3f (+/-%0.03f) for %r"
                      % (mean, std * 2, params))
            print()
            print("Detailed classification report:")
            print()
            print("The model is trained on the full development set.")
            print("The scores are computed on the full evaluation set.")
            print()
            print(classification_report(y_true, y_pred, digits=4))
            print()
        
        self.c = clf.best_params_["C"]
        self.gamma = clf.best_params_["gamma"]
        self.kernel = clf.best_params_["kernel"]

    def train_model(self, train_size=0.75, slice_number=None):
        """Builds and trains the model using a specific slice in the class
        
           Args:
               train_size (float): 0-1 percentage of sample to use to 
                                   train the model
                slice_number (str). slice number to use to build the model
        """
        
        if not slice_number:
            slice_number = list(self.df_coefficients.keys())[0]
        
        X_train, X_test, Y_train, Y_test = train_test_split(
            self.df_coefficients[str(slice_number)],
            self.df_target[str(slice_number)], 
            train_size=train_size, 
            random_state=self.random_state, 
            stratify=self.df_target[str(slice_number)]
        )
        print("training model...")
        model = SVC(kernel=self.kernel, C=self.c, gamma=self.gamma)
        model.fit(X_train, Y_train.values.ravel())
        y_hat = [x for x in model.predict(X_test)]
        print(classification_report(Y_test, y_hat, digits=4))
        self.models[str(slice_number)] = model
        
    def train_multiexpert_model(self, train_size=0.75):
        """Builds and trains the model as a multiexpert (multimage)
           model, using all the slices available in the class
        
           Args:
               train_size (float): 0-1 percentage of sample to use to 
                                   train the model
        """
        
        def get_average_pred(models, X_test):
            for i in range(0, len(models.keys())):
                y_hats[i] = models

        print("training models...")
        models = {}
        Y_train, Y_test = train_test_split(
            self.df_target[list(self.df_target.keys())[0]], 
            train_size=train_size, 
            random_state=self.random_state, 
            stratify=self.df_target[list(self.df_target.keys())[0]]
        )
        y_hats = []
        for slice_number in self.df_coefficients.keys():
            X_train, X_test = train_test_split(
                self.df_coefficients[str(slice_number)],         
                train_size=train_size, 
                random_state=self.random_state, 
                stratify=self.df_target[str(slice_number)]
            )
            model = SVC(kernel=self.kernel, C=self.c, gamma=self.gamma)
            model.fit(X_train, Y_train.values.ravel())
            models[str(slice_number)] = model
            y_hats.append([x for x in model.predict(X_test)])
        
        y_hat_final = []
        for i in range(0, len(y_hats[0])):
            y_hat_final.append(mode([item[i] for item in y_hats]))
            
        print(classification_report(Y_test, y_hat_final, digits=4))
        self.models = models
        
        
    def make_prediction(self, X_to_predict, slice_number=None):
        """Use the models available to make predictions based on the
           input passed as parameter
        
           Args:
               X_to_predict (array-like): transformed array to pass as input 
                                          to the model
           Returns:
               
        """
        if not slice_number:
            y_hats = []
            for model in self.models.keys():
                y_hats.append([x for x in self.models[str(model)].predict(X_to_predict)])
            
            y_hat_final = []
            for i in range(0, len(y_hats[0])):
                y_hat_final.append(mode([item[i] for item in y_hats]))
            return y_hat_final
        else:
            y_hats = [x for x in self.models[str(slice_number)].predict(X_to_predict)]
            return y_hats
        
