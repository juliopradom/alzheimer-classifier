from ..common.patient import Patient
from ..common.constants import PATH_TO_COEFFICIENTS_FOLDER, INDEXES
import numpy as np
import pandas as pd
import pickle
import typer

WAVELET_NAME = "bior3.3"
MAX_SAMPLES_NUMBER = 8000


def save_coefficients(
        patient_slice: int=None, 
        wavelet_level: int=4, 
        result_file: str=None
        ):
    """ Extracts Wavelet coefficients for a specific (or all) slices from the approx image
        in the passed level. The function will append the coefficients for all the patients
        
        Args:
            patient_slice (int): number of the slice to process or "all" if all slices must
                                 be processed together
            wavelet_level (int): level of the approx image
            result_file (str): file to store the results
            
    """
    for patient_class in INDEXES:
        coeff_list = [None]*3000
        coeff_index = 0
        class_name, values = list(patient_class.items())[0]
        print(f"loading {class_name} patient coefficients...")
        for i in range(values["first"], values["last"]):
            print(f" patient {i}")
            try:
                new_patient = Patient(i, class_name)
                if patient_slice:
                    coeffs = new_patient.get_slice_wavelet_coefficients(wavelet_name=WAVELET_NAME, 
                                                                        level=wavelet_level, 
                                                                        slice_number=patient_slice)
                else:
                    coeffs = new_patient.get_wavelet_coefficients(wavelet_name=WAVELET_NAME, 
                                                                  level=wavelet_level)
            except Exception as e:
                print(e)
                print(f" Couldn't load patient {i}")
                continue
            
            coeff_list[coeff_index] = (coeffs, class_name)
            coeff_index += 1
            
        coeff_list = [coeff for coeff in coeff_list if coeff]
        class_name_string = f"class_{class_name}"
        if not result_file:
            slice_string = f"slice_{patient_slice}" if patient_slice else "slice_all_"
            wavelet_level_string = f"level_{wavelet_level}"
            formated_result_file = f"{PATH_TO_COEFFICIENTS_FOLDER}/" \
                                   f"{class_name_string}_{slice_string}_{wavelet_level_string}.npy"
        else:
            formated_result_file = f"{PATH_TO_COEFFICIENTS_FOLDER}/{class_name_string}_{result_file}"
        coeff_array = np.array(coeff_list, dtype=object)
        np.save(formated_result_file, coeff_array)
        print(f"File {formated_result_file} saved") 
        
    
def main(patient_slice: int=typer.Argument(None), 
         wavelet_level: int=typer.Argument(None), 
         result_file: str=typer.Argument(None)
         ):
    
    save_coefficients(patient_slice, 
                      wavelet_level, 
                      result_file
                      )
    
if __name__ == "__main__":
    typer.run(main)
    
    

