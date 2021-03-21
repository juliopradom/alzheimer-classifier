from PIL import Image
import cv2
from numpy import asarray
import numpy
import pywt
import numpy as np

GROUPS = ["ad", "cn", "emci", "mci", "lmci", "smc"]
SLICES = [55, 56, 61, 72, 82, 90, 104, 106, 114]
COEFF_DETAILS = {"ad", "da", "dd"}
PATH_TO_FOLDER = "./images"
IMAGE_FORMAT = "{group}_wc1_{patient_number}_{slice_number}.jpg"

class Patient:
    
    def __init__(self, patient_number, group, path_to_folder=PATH_TO_FOLDER, normalize=True):
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
                if normalize:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                else:
                    image.resize((224,224,3)) 
                image = np.float32(image)
                image /= 255
                images[str(slice_number)] = image 
            except Exception as e:
                raise Exception(f"Error: could not load slice={slice_number} \
                                for patient={patient_number}, detail={e}") 
        self.images = images
        self.normalized = normalize
        self.group = group.lower()
    
    def get_full_images_array(self):
        
        final_image = self.images[str(SLICES[0])]
        for slice_number in SLICES[1:]:
            final_image = np.append(final_image, self.images[str(slice_number)], axis=0)
        return final_image
    
    def get_single_image_array(self, slice_number):
        
        return self.images[str(slice_number)]
    
    def get_slice_wavelet_coefficients(self, wavelet_name, level, slice_number):
        
        if not self.normalized:
            raise Exception("Error: patient's images are not normalized")
        # Gets only approx layer in the specified level
        raw_coeffs = pywt.wavedec2(data=self.images[str(slice_number)], wavelet=wavelet_name, level=level)
        target_coeffs = raw_coeffs[0]
        coeffs = []
        for i in range(len(target_coeffs)):
            for j in range(len(target_coeffs[0])):
                coeffs.append(target_coeffs[i][j])
        
        return coeffs
    
    def get_wavelet_coefficients(self, wavelet_name, level):
        
        if not self.normalized:
            raise Exception("Error: patient's images are not normalized")
        coefficients_list = []
        for slice_number in self.images:
            coefficients_item = self.get_slice_wavelet_coefficients(wavelet_name, level, slice_number)
            coefficients_list = numpy.concatenate([coefficients_list, coefficients_item])
        return coefficients_list
    