from PIL import Image
import cv2
from numpy import asarray
import numpy
import pywt
import numpy as np

GROUPS = {"ad", "cn", "emci", "mci", "lmci", "smc"}
SLICES = {55, 56, 61, 72, 82, 90, 104, 106, 114}
COEFF_DETAILS = {"ad", "da", "dd"}
PATH_TO_FOLDER = "./images"
IMAGE_FORMAT = "{group}_wc1_{patient_number}_{slice_number}.jpg"

class Patient:
    
    def __init__(self, patient_number, group, path_to_folder=PATH_TO_FOLDER):
        if group.lower() not in GROUPS:
            raise Exception(f"Error: group={group} not valid")
        group = group.upper()
        images = {}
        for slice_number in SLICES:
            image_file_name = IMAGE_FORMAT.format(group=group, 
                                                  patient_number=patient_number, 
                                                  slice_number=slice_number
                                                  )
            path_to_slice = f"{path_to_folder}/{group}/{image_file_name}"
            try:
                image = cv2.imread(path_to_slice)
            except:
                raise Exception(f"Error: could not load slice={slice_number} \
                                for patient={patient_number}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = np.float32(image)
            image /= 255
            images[str(slice_number)] = image     
        self.images = images
        self.group = group.lower()
    
    def get_slice_wavelet_coefficients(self, wavelet_name, level, slice_number):
        
        coefficients = pywt.wavedec2(data=self.images[str(slice_number)], wavelet=wavelet_name, \
                                     level=level)
        arr, coeff_slices, coeff_shapes = pywt.ravel_coeffs(coefficients)
        
        return arr
    
    def get_wavelet_coefficients(self, wavelet_name, level):
        
        coefficients_list = []
        for slice_number in self.images:
            coefficients_item = self.get_slice_wavelet_coefficients(wavelet_name, level, slice_number)
            print(len(coefficients_item))
            coefficients_list = numpy.concatenate([coefficients_list, coefficients_item])
        return coefficients_list
    