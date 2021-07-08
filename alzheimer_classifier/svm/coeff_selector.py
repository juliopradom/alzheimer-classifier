import pandas as pd
import numpy as np
import typer

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from ..common.constants import PATH_TO_COEFFICIENTS_FOLDER, PATH_TO_SELECTED_COEFFICIENTS_FOLDER, GROUPS, SLICES
from pymrmre import mrmr

available_algorithms = ["pca", "mrmr"]
MAX_COEFFS_BATCH = 1000


def mrmr_selector(df, df_target, max_coeffs_batch, total_features=100):
    """ Recursively selects the best {total_features} features from the DataFrame passed 
        as argument, processing only {max_coeffs_batch} columns per iteration.
        
        Args:
            df (DataFrame): contains the columns with values that we want to select
            df_target (DataFrame): contains the class associated with each row in df
            max_coeffs_batch (int): number of columns to process per iteration
            total_features (int): max_number of features to select
        
        Returns:
            selected {total_features} (int)
            
    """       
    def select_features(df, df_target, total_features):
        result = mrmr.mrmr_ensemble(
            features=df, 
            targets=df_target, 
            solution_length=total_features
        )
        return result.iloc[0][0]
    
    features_list = df.columns
    number_of_coeffs = len(df.columns)
    number_subsets = number_of_coeffs // max_coeffs_batch
    module_subsets = False if number_of_coeffs % max_coeffs_batch == 0 else True
    selected_features = []
    # If we are handling the final subset, select and return
    if number_subsets == 0 or (number_subsets == 1 and not module_subsets):
        print("Processing last subset...")
        result = select_features(df, df_target, total_features)
        selected_features.extend([feature for feature in result if feature != "group"])
        return selected_features
    # Otherwise, get selected features for every subset
    else:
        print("Processing subsets...")
        for i in range(0, number_subsets):
            print(f" subset {i}")
            subset = df[features_list[i*MAX_COEFFS_BATCH:(i+1)*MAX_COEFFS_BATCH]]
            result = select_features(subset, df_target, total_features)
            selected_features.extend([feature for feature in result if feature != "group"])
        if module_subsets: 
            print(f" subset {number_subsets}")
            subset = df[features_list[(number_subsets-1)*MAX_COEFFS_BATCH:]]
            result = select_features(subset, df_target, total_features)
            selected_features.extend([feature for feature in result if feature != "group"])
            
    selected_features = mrmr_selector(df[selected_features], df_target, max_coeffs_batch)
    return selected_features


def pca_reductor(df, total_features=100):
    """ Applies dimensionality reduction to fit a maximum of {total_features}
        features
        
        Args:
            df (DataFrame): contains the columns with values that we want to reduct
            total_features (int): max_number of features to select
        
        Returns:
            reducted df (DataFrame)
            
    """ 
    pca = PCA(n_components = 100, svd_solver = 'full')
    df = pca.fit_transform(df)
    return df


def transform_and_select_coefficients(
    patient_slice: str="all", 
    wavelet_level: int=4, 
    algorithm: str="pca",
    standarize: bool=True,
    save=True
):
    """ Apply feature selection and returns a DataFrame with the result
        features
        
        Args:
            patient_slice (str): slice to process
            wavelet_level (int): level of the wavelet coefficients accessed
            algorithm (str): algorithm to use
            standarize (bool): True if data must be standardized
            save (bool): True if resulting DataFrame must be saved
        
        Returns:
            reducted df (DataFrame)
            
    """ 
    # Check if algorithm is available
    if algorithm not in available_algorithms:
        raise Exception(f"algorithm={algorithm} not valid")
    
    # Load coefficients
    all_coefficients = None
    all_targets = None
    group_int = 1
    if patient_slice == "all":
        print("loading coefficients for all slices...")
        for group in GROUPS:
            partial_coefficients = None
            for slice_number in SLICES:
                file = f"{PATH_TO_COEFFICIENTS_FOLDER}/" \
                       f"class_{group}_slice_{slice_number}_level_{wavelet_level}.npy"
                coefficients = np.load(file, allow_pickle=True)
                slice_partial_coefficients = [coeff[0] for coeff in coefficients]
                if partial_coefficients is not None:
                    partial_coefficients = [partial_coefficients[i] + slice_partial_coefficients[i] for i in range(0, len(partial_coefficients))]
                else:
                    partial_coefficients = slice_partial_coefficients
                    
            if all_coefficients is not None:
                all_coefficients = np.concatenate([all_coefficients, partial_coefficients])
            else:
                all_coefficients = partial_coefficients
            partial_targets = [group_int]*len(partial_coefficients)    
            if all_targets is not None:
                all_targets = np.concatenate([all_targets, partial_targets])
            else:
                all_targets = partial_targets
            group_int += 1
    else:
        print(f"loading coefficients for slice={patient_slice}...")
        for group in GROUPS:
            file = f"{PATH_TO_COEFFICIENTS_FOLDER}/" \
                   f"class_{group}_slice_{patient_slice}_level_{wavelet_level}.npy"
            coefficients = np.load(file, allow_pickle=True)
            partial_coefficients = [coeff[0] for coeff in coefficients]
            if all_coefficients is not None:
                all_coefficients = np.concatenate([all_coefficients, partial_coefficients])
            else:
                all_coefficients = partial_coefficients
            partial_targets = [group_int]*len(partial_coefficients)
            if all_targets is not None:
                all_targets = np.concatenate([all_targets, partial_targets])
            else:
                all_targets = partial_targets
            group_int += 1
    
    # Make the mean of the distribution 0
    if standarize:
        print("Standardizing...")
        scaler = StandardScaler()
        all_coefficients = scaler.fit_transform(all_coefficients)
        
    # Create a more readable feature list
    print("Adding features...")
    features_list = [f"feature{i}" for i in range(0, len(all_coefficients[0]))]
    print("length coeff")
    print(len(features_list))
    # Creating DataFrame for algorithm's input
    print("Transforming into DataFrame...")
    df_coefficients = pd.DataFrame(data=all_coefficients, columns=features_list)
    df_target = pd.DataFrame(data=all_targets, columns=["group"])
    
    # Memory concern
    del all_coefficients
    del all_targets
    
    if algorithm == "mrmr":
        print("Applying MRMR...")
        selected_features = mrmr_selector(df_coefficients, df_target, MAX_COEFFS_BATCH)
        df_coefficients = df_coefficients[selected_features]
    
    else:
        print("Applying PCA...")
        df_coefficients = pca_reductor(df_coefficients)
        df_coefficients = pd.DataFrame(data=df_coefficients)
        print(df_coefficients)
        
    df_concat = pd.concat([df_coefficients, df_target], axis=1)
    if save:
        df_concat.to_csv(
            f"{PATH_TO_SELECTED_COEFFICIENTS_FOLDER}/" \
            f"selected_features_{algorithm}_slice_{patient_slice}_wavelet_{wavelet_level}.csv"
        )
    return df_concat


def main(
    patient_slice: str=typer.Argument(None), 
    wavelet_level: int=typer.Argument(None),
    algorithm: str=typer.Argument(None),
    save: bool=typer.Argument(True)
):
    
    transform_and_select_coefficients(
        patient_slice, 
        wavelet_level, 
        algorithm,
        save
    )
    
if __name__ == "__main__":
    typer.run(main)
