from patient import Patient
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
import pymrmr
from pymrmre import mrmr
import pandas as pd
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

parameters = {'kernel':('linear', 'rbf', 'sigmoid'), 'C': [1, 10, 100, 1000], 'gamma': [1e-1, 1e-2, 1e-3, 1e-4]}
score1 = metrics.make_scorer(metrics.precision_score, average = None)
score2 = metrics.make_scorer(metrics.recall_score, average = None)
scores = [score1,
          score2
          ]
MAX_COEFFS_BATCH = 10000

"""
INDEX_OF_AD_PATIENTS = {"first": 1, "last": 1497}
INDEX_OF_CN_PATIENTS = {"first": 1501, "last": 3000}
INDEX_OF_EMCI_PATIENTS = {"first": 3001, "last": 3388}
INDEX_OF_LMCI_PATIENTS = {"first": 70, "last": 1500}
INDEX_OF_MCI_PATIENTS = {"first": 1501, "last": 3000}
INDEX_OF_SMC_PATIENTS = {"first": 3001, "last": 3662}


coefficients_ad_matrix = []
group_ad_array = []
coefficients_cn_matrix = []
group_cn_array = []
coefficients_mci_matrix = []
group_mci_array = []
coefficients_emci_matrix = []
group_emci_array = []
coefficients_lmci_matrix = []
group_lmci_array = []
coefficients_smc_matrix = []
group_smc_array = []


print("loading ad patients...")
for i in range(INDEX_OF_AD_PATIENTS["first"], INDEX_OF_AD_PATIENTS["last"]):
    print(f" {i}")
    try:
        new_patient = Patient(i, "ad")
        coeffs = new_patient.get_wavelet_coefficients(wavelet_name="bior3.3", level=4)
    except:
        print(f"Couldn't load patient {i}")
        continue
    coefficients_ad_matrix.append(coeffs)
    group_ad_array.append("ad")

print(len(coefficients_ad_matrix[0]))
np.save("coefficients_ad_matrix.npy", coefficients_ad_matrix)
coefficients_ad_matrix = []
print("loading cn patients...")
for i in range(INDEX_OF_CN_PATIENTS["first"], INDEX_OF_CN_PATIENTS["last"]):
    print(f" {i}")
    try:
        new_patient = Patient(i, "cn")
        coeffs = new_patient.get_wavelet_coefficients(wavelet_name="bior3.3", level=4)
    except:
        print(f"Couldn't load patient {i}")
        continue
    coefficients_cn_matrix.append(coeffs)
    group_cn_array.append("cn")

np.save("coefficients_cn_matrix.npy", coefficients_cn_matrix)
coefficients_cn_matrix = []


print("loading mci patients...")
for i in range(INDEX_OF_MCI_PATIENTS["first"], INDEX_OF_MCI_PATIENTS["last"]):
    print(f" {i}")
    try:
        new_patient = Patient(i, "mci")
        coeffs = new_patient.get_wavelet_coefficients(wavelet_name="bior3.3", level=4)
    except:
        print(f"Couldn't load patient {i}")
        continue
    coefficients_mci_matrix.append(coeffs)
    group_mci_array.append("mci")
np.save("coefficients_mci_matrix.npy", coefficients_mci_matrix)
coefficients_mci_matrix = []

print("loading emci patients...")
for i in range(INDEX_OF_EMCI_PATIENTS["first"], INDEX_OF_EMCI_PATIENTS["last"]):
    print(f" {i}")
    try:
        new_patient = Patient(i, "emci")
        coeffs = new_patient.get_wavelet_coefficients(wavelet_name="bior3.3", level=4)
    except:
        print(f"Couldn't load patient {i}")
        continue
    coefficients_emci_matrix.append(coeffs)
    group_emci_array.append("emci")
np.save("coefficients_emci_matrix.npy", coefficients_emci_matrix)
coefficients_emci_matrix = []

print("loading lmci patients...")
for i in range(INDEX_OF_LMCI_PATIENTS["first"], INDEX_OF_LMCI_PATIENTS["last"]):
    print(f" {i}")
    try:
        new_patient = Patient(i, "lmci")
        coeffs = new_patient.get_wavelet_coefficients(wavelet_name="bior3.3", level=4)
    except:
        print(f"Couldn't load patient {i}")
        continue
    coefficients_lmci_matrix.append(coeffs)
    group_lmci_array.append("lmci")
np.save("coefficients_lmci_matrix.npy", coefficients_lmci_matrix)
coefficients_lmci_matrix = []

print("loading smc patients...")
for i in range(INDEX_OF_SMC_PATIENTS["first"], INDEX_OF_SMC_PATIENTS["last"]):
    print(f" {i}")
    try:
        new_patient = Patient(i, "smc")
        coeffs = new_patient.get_wavelet_coefficients(wavelet_name="bior3.3", level=4)
    except:
        print(f"Couldn't load patient {i}")
        continue
    coefficients_smc_matrix.append(coeffs)
    group_smc_array.append("smc")
np.save("coefficients_smc_matrix.npy", coefficients_smc_matrix)
coefficients_smc_matrix = []

"""
print("loading models...")
coefficients_ad_matrix = np.load("coefficients_ad_matrix.npy")
group_ad_array = [1]*len(coefficients_ad_matrix)
coefficients_cn_matrix = np.load("coefficients_cn_matrix.npy")
group_cn_array = [2]*len(coefficients_cn_matrix)
coefficients_mci_matrix = np.load("coefficients_mci_matrix.npy")
group_mci_array = [3]*len(coefficients_mci_matrix)
coefficients_lmci_matrix = np.load("coefficients_lmci_matrix.npy")
group_lmci_array = [4]*len(coefficients_lmci_matrix)
coefficients_emci_matrix = np.load("coefficients_emci_matrix.npy")
group_emci_array = [5]*len(coefficients_emci_matrix)
coefficients_smc_matrix = np.load("coefficients_smc_matrix.npy")
group_smc_array = [6]*len(coefficients_smc_matrix)

print("concatenating matrix...")
coefficients_matrix = np.concatenate([coefficients_ad_matrix, coefficients_cn_matrix])
coefficients_ad_matrix = None
coefficients_cn_matrix = None
coefficients_matrix = np.concatenate([coefficients_matrix, coefficients_mci_matrix])
coefficients_mci_matrix = None
coefficients_matrix = np.concatenate([coefficients_matrix, coefficients_lmci_matrix])
coefficients_lci_matrix = None
coefficients_matrix = np.concatenate([coefficients_matrix, coefficients_emci_matrix])
coefficients_emci_matrix = None
coefficients_matrix = np.concatenate([coefficients_matrix, coefficients_smc_matrix])
coefficients_smc_matrix = None

print("concatenating group...")
group_array = np.concatenate([group_ad_array, group_cn_array])
group_ad_array = None
group_cn_array = None
group_array = np.concatenate([group_array, group_mci_array])
group_mci_array = None
group_array = np.concatenate([group_array, group_emci_array])
group_emci_array = None
group_array = np.concatenate([group_array, group_lmci_array])
group_lmci_array = None
group_array = np.concatenate([group_array, group_smc_array])
group_smc_array = None

#group_array = group_array.astype('str')

print("Standarizing...")
scaler = StandardScaler()
dataset = scaler.fit_transform(coefficients_matrix)

print("Adding features...")
features_list = [None]*len(dataset[0])
for i in range(0, len(features_list)):
    features_list[i] = f"feature{i}"

df_coefficients = pd.DataFrame(data=dataset, columns=features_list)
df_group = pd.DataFrame(data=group_array, columns=["group"])
#df.insert(0, "group", group_array, True)
dataset = None

def mrmr_selector(df, df_target, max_coeffs_batch, total_features=100):
    
    features_list = df.columns
    number_of_coeffs = len(df.columns)
    number_subsets = number_of_coeffs // max_coeffs_batch
    module_subsets = False if number_of_coeffs % max_coeffs_batch == 0 else True
    selected_features = []
    if number_subsets == 0:
        print("Processing last subset...")
        result = mrmr.mrmr_ensemble(features=df, targets=df_target, solution_length=total_features)
        print(result.iloc[0][0])
        selected_features.extend(result.iloc[0][0])
        return selected_features
    else:
        print("Processing subsets...")
        for i in range(0, number_subsets):
            print(f" subset {i}")
            subset = df[features_list[i*MAX_COEFFS_BATCH:(i+1)*MAX_COEFFS_BATCH]]

            #subset.insert(0, "group", target, True)
            #selected_features.extend(pymrmr.mRMR(subset, 'MIQ', 100))
            result = mrmr.mrmr_ensemble(features=subset, targets=df_target, solution_length=total_features)
            print(result.iloc[0][0])
            selected_features.extend(result.iloc[0][0])
        if module_subsets: 
            print(f" subset {number_subsets}")
            subset = df[features_list[(number_subsets-1)*MAX_COEFFS_BATCH:]]
            result = mrmr.mrmr_ensemble(features=subset, targets=df_target, solution_length=total_features)
            print(result.iloc[0][0])
            selected_features.extend(result.iloc[0][0])
            
    selected_features = mrmr_selector(df[selected_features], df_target, max_coeffs_batch)
    return selected_features

"""
print("Applying MRMR")
selected_features = mrmr_selector(df_coefficients, df_group, MAX_COEFFS_BATCH)
print(selected_features)


np.save("selected_features.npy", selected_features)
"""
selected_features = np.load("selected_features.npy")
print(selected_features)
df_coefficients = df_coefficients[selected_features]

"""
print("pca...")
pca = PCA(n_components = 100, svd_solver = 'full')
df_coefficients = pca.fit_transform(df_coefficients)
"""

X_train, X_test, Y_train, Y_test = train_test_split(df_coefficients, df_group, train_size=0.8,random_state=11, stratify=df_group)

for score in scores:
    svc = SVC()
    clf = GridSearchCV(estimator=svc, param_grid=parameters, scoring=score)
    clf.fit(X_train, Y_train.values.ravel())
    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = Y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()
    

"""
#model = MLPClassifier(hidden_layer_sizes=(100,50,25),max_iter=2000)
model.fit(X_train, Y_train.values.ravel())
y_hat = [x for x in model.predict(X_test)]
ascore = accuracy_score(Y_test, y_hat)

print(ascore)
"""
