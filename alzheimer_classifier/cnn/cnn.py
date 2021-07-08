import numpy as np
from tensorflow.keras import datasets, layers, metrics, models
from tensorflow.keras.applications import VGG16
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from statistics import mode

from .. common.constants import PATH_TO_RAW_RESHAPED_IMAGES_FOLDER, PATH_TO_CNN_FOLDER, GROUPS, SLICES

class CNN:
    
    def __init__(
        self, 
        path_to_raw_folder=PATH_TO_RAW_RESHAPED_IMAGES_FOLDER, 
        path_to_cnn_folder=PATH_TO_CNN_FOLDER,
        target_column="group",
        target_score="f1",
        slices=[],
        input_shape=(224,224,3),
        random_state=104729,
        class_number=6,
        model_list=None
    ):
        """ Constructor: initialize an instance of the class.
        
        Args:
            path_to_raw_folder (str): path to array files in local system
            path_to_cnn_folder (str): path to current directory in local system
            target_column (str): target column in saved csv
            target_score (str): target score to use for training
            slices (list): list of slices to use to build the models (>1 if multiexpert)
            input_shape (tuple): size of the images to process
            random_state (int): random state parm for train_test_split
            class_number (int): number of existing classes
            model_list (list): in case models are loaded from an external source, list of models
        
        """
        image_arrays = {}
        image_targets = {}
        for slice_number in slices:
            path_to_arrays_file = f"{path_to_raw_folder}/slice_{slice_number.split('_')[0]}.npy"
            # Drop residual columns
            arrays = np.load(path_to_arrays_file, allow_pickle=True)
            hey = np.array([array[0] for array in arrays])
            print(hey.shape)
            image_arrays[str(slice_number)] = np.array([array[0] for array in arrays])
            image_targets[str(slice_number)] = np.array([GROUPS.index(array[1]) for array in arrays])
            
        
        if target_score == "f1":
            scores = [metrics.Precision(), metrics.Recall()]
        elif target_score == "precision":
            scores = [metrics.Precision()]
        elif target_score == "recall":
            scores = [metrics.Recall()]
        else:
            raise Exception(f"Error: target_score={target_score} not available")
            
        self.path_to_cnn_folder = path_to_cnn_folder
        self.input_shape = input_shape
        self.class_number = class_number
        self.image_arrays = image_arrays
        self.image_targets = image_targets
        self.scores = scores
        self.target_column = target_column
        models_dict = {}
        if model_list:
            for model_slice in model_list:
                models_dict[str(model_slice)] = models.load_model(f"{self.path_to_cnn_folder}/{model_slice}.h5")
        
        self.models_dict = models_dict
        self.random_state = random_state
    

    def train_model(self, train_size=0.75, slice_number=None):
        """Builds and trains the model using a specific slice in the class
        
           Args:
               train_size (float): 0-1 percentage of sample to use to 
                                   train the model
               slice_number (str): slice number to use to build the model
        """
        
        if not slice_number:
            slice_number = list(self.image_arrays.keys())[0]
        
        cnn = models.Sequential([
            layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=self.input_shape),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(self.class_number, activation='softmax')
        ])
        
            
        cnn.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=["accuracy"])
        
        X_train, X_test, Y_train, Y_test = train_test_split(
            self.image_arrays[str(slice_number)],
            self.image_targets[str(slice_number)], 
            train_size=train_size, 
            random_state=self.random_state, 
            stratify=self.image_targets[str(slice_number)]
        )
        print("Training model...")
        print(Y_train)
        cnn.fit(X_train, Y_train, epochs=10)
        cnn.evaluate(X_test, Y_test)
        y_pred = cnn.predict(X_test)
        y_pred_classes = [np.argmax(element) for element in y_pred]
        
        print()
        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        print(classification_report(Y_test, y_pred_classes, digits=4))
        print()
        
        print("Saving model...")
        cnn.save(f"{self.path_to_cnn_folder}/{slice_number}.h5")
        self.models_dict[str(slice_number)] = cnn
    
    def train_model_tl(self, train_size=0.75, slice_number=None):
        """Builds and trains the model using a specific slice in the class
           and VGG16 model
        
           Args:
               train_size (float): 0-1 percentage of sample to use to 
                                   train the model
               slice_number (str): slice number to use to build the model
        """
        
        if not slice_number:
            slice_number = list(self.image_arrays.keys())[0]
        
       
        # add preprocessing layer to the front of VGG
        vgg = VGG16(input_shape=list(self.input_shape), weights='imagenet', include_top=False)
        
        # don't train existing weights
        for layer in vgg.layers:
          layer.trainable = False
          
        
        # our layers - you can add more if you want
        x = layers.Flatten()(vgg.output)
        # x = Dense(1000, activation='relu')(x)
        prediction = layers.Dense(6, activation='softmax')(x)
        
        # create a model object
        cnn = models.Model(inputs=vgg.input, outputs=prediction)
        
        # view the structure of the model
        cnn.summary()
        
        # tell the model what cost and optimization method to use
        cnn.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=["accuracy"])
        
        X_train, X_test, Y_train, Y_test = train_test_split(
            self.image_arrays[str(slice_number)],
            self.image_targets[str(slice_number)], 
            train_size=train_size, 
            random_state=self.random_state, 
            stratify=self.image_targets[str(slice_number)]
        )
        print("Training model...")
        print(Y_train)
        cnn.fit(X_train, Y_train, epochs=10)
        cnn.evaluate(X_test, Y_test)
        y_pred = cnn.predict(X_test)
        y_pred_classes = [np.argmax(element) for element in y_pred]
        
        print()
        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        print(classification_report(Y_test, y_pred_classes, digits=4))
        print()
        
        print("Saving model...")
        cnn.save(f"{self.path_to_cnn_folder}/{slice_number}_tl.h5")
        self.models_dict[str(slice_number)] = cnn
        

        
    def make_prediction(self, X_to_predict=None, slice_number="all", save=False):
        """Use the models available to make predictions based on the
           input passed as parameter
        
           Args:
               X_to_predict (array-like): transformed array to pass as input 
                                          to the model
               slice_number (str): all if all models will be use
               save (bool): True to save the results
           Returns:
               
        """
        if not X_to_predict:
            X_to_predict = train_test_split(
                self.image_arrays[str(slice_number)],
                train_size=0.75, 
                random_state=self.random_state, 
                stratify=self.image_targets[str(slice_number)]
            )[1]
     
        y_hat_final = []
        if slice_number == "all":
            y_hats = []
            for model in self.models.keys():
                y_hats_array = [x for x in self.models_dict[str(model)].predict(X_to_predict)]
                y_hats_argm = [np.argmax(element) for element in y_hats_array]
                y_hats.append(y_hats_argm)
            
            y_hat_final = []
            for i in range(0, len(y_hats[0])):
                y_hat_final.append(mode([item[i] for item in y_hats]))
        else:
            y_hats = [x for x in self.models_dict[str(slice_number)].predict(X_to_predict)]
            y_hat_final = [np.argmax(element) for element in y_hats]
        
        if save:
            array_object = np.array(y_hat_final)
            np.save(f"./alzheimer_classifier/cnn/slice_{slice_number}_prediction.npy", array_object)
        
        return y_hat_final
    
    def get_scores_model(self, train_size=0.75, slice_number=None):
        """Builds and trains the model using a specific slice in the class
           and VGG16 model
        
           Args:
               train_size (float): 0-1 percentage of sample to use to 
                                   train the model
               slice_number (str): slice number to use to build the model
        """
        
        if not slice_number:
            slice_number = list(self.image_arrays.keys())[0]
        
        
        X_train, X_test, Y_train, Y_test = train_test_split(
            self.image_arrays[str(slice_number)],
            self.image_targets[str(slice_number)], 
            train_size=train_size, 
            random_state=self.random_state, 
            stratify=self.image_targets[str(slice_number)]
        )
        print("Training model...")
        y_hats = [x for x in self.models_dict[str(slice_number)].predict(X_test)]
        y_hats = [np.argmax(element) for element in y_hats]
        y_hat_final = y_hats
        
        print()
        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        print(classification_report(Y_test, y_hat_final, digits=4))
        print()