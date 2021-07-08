import cv2
import pywt
import numpy as np

from .constants import GROUPS, SLICES, PATH_TO_IMAGE_FOLDER, IMAGE_FORMAT

class Patient:
    
    def __init__(
        self, 
        patient_number, 
        group, 
        path_to_folder=PATH_TO_IMAGE_FOLDER, 
        normalize_colour=True, 
        resize_image=False,
        resize_shape=(224, 224, 3)
    ):
        """ Constructor: initialize an instance of the class.
        
        Args:
            patient_number (str): number of the patient
            group (str): group to which the patient belongs
            path_to_folder (str): relative path to images folder
            normalize_colour (str): if set to True the image will be 
                                    converted to BGR2GRAY
            resize_image (str): if set to True the image will be resized to 224x224
       """
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
                if resize_image:
                    image.resize(resize_shape) 
                if normalize_colour:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = np.float32(image)
                image /= 255
                images[str(slice_number)] = image 
            except Exception as e:
                raise Exception(f"Error: could not load slice={slice_number}" \
                                f"for patient={patient_number}, detail={e}") 
        self.images = images
        self.normalized_colour = normalize_colour
        self.group = group.lower()
    
    def get_full_images_array(self):
        """ Gets the all the flatten images array concatenated in a single 1D array
        Returns:
            np.array 
       """
        
        final_image = self.images[str(SLICES[0])]
        for slice_number in SLICES[1:]:
            final_image = np.append(
                                final_image, 
                                self.get_single_image_array(slice_number), 
                                axis=0
                            )
        return final_image
    
    def get_single_image_array(self, slice_number):
        """ Gets the flatten array corresponding to the slice specified in the 
            parameter
        Args:
            slice_number (int): number of the slice to access
        Returns:
            np.array 
       """
        return self.images[str(slice_number)]
    
    def get_slice_wavelet_coefficients(self, wavelet_name, level, slice_number):
        """ Gets the wavelet coefficients associated with the slice_number 
            specified in the parameter
        Args:
            wavelet_name (str): wavelet type to use
            level (int): level (depth) of the approximation layer to access
            slice_number (int): number of the slice to access    
        Returns:
            list
       """
        if not self.normalized_colour:
            raise Exception("Error: patient's images are not normalized")
        # Gets only approx layer in the specified level
        raw_coeffs = pywt.wavedec2(
                            data=self.get_single_image_array(slice_number), 
                            wavelet=wavelet_name, 
                            level=level
                        )
        target_coeffs = raw_coeffs[0]
        coeffs = []
        for i in range(len(target_coeffs)):
            for j in range(len(target_coeffs[0])):
                coeffs.append(target_coeffs[i][j])
        
        return coeffs
    
    def get_wavelet_coefficients(self, wavelet_name, level):
        """ Gets all the wavelet coefficients associated with the patient
        Args:
            wavelet_name (str): wavelet type to use
            level (int): level (depth) of the approximation layer to access         
        Returns:
            list
       """
       
        if not self.normalized_colour:
            raise Exception("Error: patient's images are not normalized")
        coefficients_list = []
        for slice_number in self.images:
            coefficients_item = self.get_slice_wavelet_coefficients(
                                    wavelet_name, 
                                    level, 
                                    slice_number
                                )
            coefficients_list = np.concatenate([coefficients_list, coefficients_item])
        return coefficients_list
    