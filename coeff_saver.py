from patient import Patient
import numpy as np
import pandas as pd
import pickle
import typer

WAVELET_NAME = "bior3.3"
MAX_SAMPLES_NUMBER = 8000
INDEXES = [
    {"ad": {"first":1, "last": 1497}},
    {"cn": {"first": 1501, "last": 3000}},
    {"emci": {"first": 3001, "last": 3388}},
    {"lmci": {"first": 70, "last": 1500}},
    {"mci": {"first": 1501, "last": 3000}},
    {"smc": {"first": 3001, "last": 3662}},
    ]


def save_coefficients(
        patient_slice: int=None, 
        wavelet_level: int=4, 
        result_file: str=None
        ):
    
    coeff_list = [None]*8000
    coeff_index = 0
    for patient_class in INDEXES:
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
            
    coeff_list = [coeff for coeff in coeff_list if coeff]
    if not result_file:
        slice_string = f"slice_{patient_slice}" if patient_slice else "slice_all_"
        wavelet_level = f"level_{wavelet_level}"
        result_file = f"{slice_string}_{wavelet_level}.pkl"
    open_file = open(result_file, "wb")
    pickle.dump(coeff_list, open_file)
    open_file.close()
        
    
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
    
    

