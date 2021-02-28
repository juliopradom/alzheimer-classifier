from patient import Patient

NUMBER_OF_AD_PATIENTS =
patients_ad_List = []
new_patient = Patient(1, "ad")
coefficients = new_patient.get_wavelet_coefficients(wavelet_name="bior3.3", level=2)

print(len(coefficients))
#print(len(coefficients))