import numpy as np
import typer

from ..common.patient import Patient
from ..common.constants import INDEXES, PATH_TO_RAW_RESHAPED_IMAGES_FOLDER

def get_raw_images(
    patient_slice: int, 
    result_file: str=None,
    save: bool=True,
):
    """ Extracts image array for a specific slice. The function will append 
        the arrays for all the patients
        
        Args:
            patient_slice (int): number of the slice to process 
            result_file (str): file to store the results
            save (bool): True if array must be saved
        
        Returns:
            np-array: array of tuples. Each tuple contains the values of one patient 
                      plus the class it belongs to
            
    """
    array_list = [None]*8000
    array_index = 0
    for patient_class in INDEXES:
        class_name, values = list(patient_class.items())[0]
        print(f"loading {class_name} patient arrays...")
        for i in range(values["first"], values["last"]):
            print(f" patient {i}")
            try:
                new_patient = Patient(
                    i, 
                    class_name, 
                    normalize_colour=False, 
                    resize_image=True, 
                    path_to_folder="./alzheimer_classifier/images"
                )
                if patient_slice:
                    array = new_patient.get_single_image_array(slice_number=patient_slice)
                else:
                    array = new_patient.get_single_image_array(slice_number=patient_slice)
            except Exception as e:
                print(e)
                print(f" Couldn't load patient {i}")
                continue
            
            array_list[array_index] = (array, class_name)
            array_index += 1
            
    array_list = [item for item in array_list if item]
    if not result_file:
        slice_string = f"slice_{patient_slice}" 
        formated_result_file = f"{PATH_TO_RAW_RESHAPED_IMAGES_FOLDER}/" \
                               f"{slice_string}.npy"
    else:
        formated_result_file = f"{PATH_TO_RAW_RESHAPED_IMAGES_FOLDER}/{result_file}.npy"
    array_object = np.array(array_list, dtype=object)
    if save:
        np.save(formated_result_file, array_object)
        print(f"File {formated_result_file} saved") 
    return array_object
        
    
def main(
    patient_slice: int=typer.Argument(None), 
    result_file: str=typer.Argument(None),
    save: bool=typer.Argument(True)
):
    
    get_raw_images(
        patient_slice, 
        result_file,
        save
    )
    
if __name__ == "__main__":
    typer.run(main)

