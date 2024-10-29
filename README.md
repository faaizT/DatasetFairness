
# Dataset Fairness: Achievable Fairness on Your Data With Utility Guarantees

This project is the official implementation of "Dataset Fairness: Achievable Fairness on Your Data With Utility Guarantees"

## Installation
To install this package, run the following command:  
```pip install -r requirements.txt```

## Training the YOTO model
To train the YOTO model, run the following command:  
```python3 train_yoto.py```  

This will run start the training of YOTO models. The arguments can be changed as deemed appropriate. The json files in ```config``` folder specify the hyperparameters used to train the YOTO as well as the separately trained models. To use these hyperparameters, simply use the ```--load_default_config=True``` argument when training the model.

To switch to different datasets you simply need to change the ```--input_dim``` argument. 

| ```input_dim```    | Dataset |
| -------- | ------- |
| 9  | COMPAS    |
| 102 | Adult     |
| 512    | Jigsaw    |
| 12288    | CelebA    |

**Note:** Before training on Jigsaw or CelebA dataset, please ensure that the respective datasets are downloaded and can be accessed in the parent directory ```../```.


## Training models separately
To train the models separately, run the following command:  
```python3 train_separate_nns.py```  

Again, the ```config``` folder contains json files with the appropriate hyperparameters and can be used directly using the ```--load_default_config=True``` argument when training the model.

## Plotting the confidence intervals
Once the models are trained, the confidence intervals can be generated using the ```scripts/results_collector.py``` file. 

**Important:** The values of variables must be modified in ```scripts/results_collector.py``` file as appropriate when generating the plots.