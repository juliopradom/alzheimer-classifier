# alzheimer-classifier
This repo holds the code for the TFG of Julio Prado 2020-2021. In particular, `alzheimer-classifier` is a module
that contains all the logic needed to process biomedical images and train and use custom SVM and CNN models.

## Full Project

The full project and research can be found here.

## How to install
1. Using pip:
```
pip install requirements.txt
```

2. Using poetry:
```
poetry install
```
## How to use

In order to use the code properly it is necessary to get a set of biomedical images to process. For the investigation
carried out the source was [ADNI](http://adni.loni.usc.edu/). Once we have the images downloaded, copy the folder
in the root directory of the module and rename it to `images`. 

To show the functioning of the whole process, let us take as an example the procedure to buld and train an SVM classifier:
1. Extract and save the wavelet coefficients for a specific (or all) slices. For example, to extract the coefficients of the slice
55 at wavelet level 4 for all the patients:
```
python -m alzheimer-classifier.svm.coeff_getter 55 4
```
2. Select the most promising features among the extracted coefficients. For instance, we could use PCA
```
python -m alzheimer-classifier.svm.coeff_selector 55 4 pca
```
3. Use svm class:
```
from alzheimer-classifier.svm.svm import svm

model = SVM(slices=[55])
model.find_best_parameters()
model.train_model()

# load coefficient of a specific image to make predictions in x_to_predict variable
...
prediction = model.make_prediction(x_to_predict)
```


