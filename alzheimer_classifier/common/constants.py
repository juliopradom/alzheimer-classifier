GROUPS = ["ad", "cn", "emci", "mci", "lmci", "smc"]
SLICES = [55, 56, 61, 72, 82, 90, 104, 106, 114]
PATH_TO_IMAGE_FOLDER = "./alzheimer_classifier/images"
PATH_TO_COEFFICIENTS_FOLDER = "./alzheimer_classifier/svm/coefficients"
PATH_TO_SELECTED_COEFFICIENTS_FOLDER = "./alzheimer_classifier/svm/coefficients_selected"
PATH_TO_RAW_RESHAPED_IMAGES_FOLDER = "./alzheimer_classifier/cnn/reshaped_images"
PATH_TO_CNN_FOLDER = "./alzheimer_classifier/cnn"
IMAGE_FORMAT = "{group}_wc1_{patient_number}_{slice_number}.jpg"
INDEXES = [
    {"ad": {"first":1, "last": 1497}},
    {"cn": {"first": 1501, "last": 3000}},
    {"emci": {"first": 3001, "last": 3388}},
    {"lmci": {"first": 70, "last": 1500}},
    {"mci": {"first": 1501, "last": 3000}},
    {"smc": {"first": 3001, "last": 3662}},
    ]
