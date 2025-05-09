<h1 align="center">Computer-Vision</h1>

## Contents
- [Contents](#contents)
- [File Description](#file-description)
- [Requirements](#requirements)
- [Model Structure](#structure)
- [Training the Model](#training-the-model)
- [Hyperparameter Searching and Visualization](#hyperparameter-searching-and-vis)
- [Loading and Testing](#loading-and-testing)

***

## File Description
- data_and_test: Contains CIFAR-10 dataset. You can also download the dataset from [here](https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz) , also contains the .ipynb file you need for training and testing.
- model: Contains the best model parameters and itself.
- vis: Contains the visualization of the training process and some other figures.
- src: Contains the source code.
- requirements.txt: Required packages.



## Requirements
This project requires Python >= 3.8. See the requirements.txt file for the required packages. You can install them using the following command:

```cmd
pip install -r requirements.txt
```

You can also run the project using anaconda for convenience.

## Training the Model
Download the repository, and set the working directory to the root directory of the project. Run the following command to train the model:

```cmd
python data_and_test/cifar-10/main.py
```
You can also train the model by running the .ipynb file, open it in an IDE and run the cells sequencially.

The training process will be printed in the console, and the curves of the training process will be presented right after the training process.The whole process takes about 470 minutes on my computer.

To change the hyperparameters, you can modify the `main.py` file directly:

```python
lr = 1e-4
l2_reg = 8e-6
decay = 0.96
batch_size = 1
num_epochs = 20
```

To test the model, you can uncomment the last few lines of the `main.py` file:

```python
test_loss, test_acc = best_model.predict(test_images, test_labels)
print(f"\nFinal Test Accuracy: {test_acc:.4f} | Loss: {test_loss:.4f}")
```

To save the model and its best parameters, you can uncomment the last few lines of the `main.py` file:

```python
save_model(best_model, "best_model.pkl")
print(f"\nBest model saved as 'best_model.pkl' with val acc = {best_val_acc:.4f}")
```

The model will be saved in the current directory.

## Hyperparameter Searching and Visualization
To search for the best hyperparameters and visualize the result, you can check the examples in the `data_and_test/cifar-10/param_and_plot.ipynb` file. Run the cells sequencially to get the results. Or you can simply open the .html file to see my results.


## Loading and Testing the Model
This part is also included in the  `data_and_test/cifar-10/param_and_plot.ipynb` file.
