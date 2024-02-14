# CNN_With_Tensorboard

## Introduction:
In this project, I aimed to train a model using basic CNN architecture with Tensorboard to recognize the some animals images.

## Tensorboard:
TensorBoard, along with saving training or prediction images, allows you to save them in TensorBoard and examine the changes graphically during the training phase by recording scalar values such as loss and accuracy. It's a very useful and practical tool.

## Dataset:
- I used the Animals dataset for this project, which consists of 10 labels (letters) with total 26k images respectively.
- I randomly split the dataset into training, validation, and test sets with ratios of (0.6, 0.2, 0.2) respectively.
- And I divided them into mini-batches with a batch size of 128. 
- Since the sizes of the images were varied, I resized them to (128x128x3) RGB images and applied normalization.
- Link For More information and downloading dataset: https://www.kaggle.com/datasets/alessiocorrado99/animals10

## Train:

- I built the model with a basic CNN structure. I repeated the structure three times, which consists of a convolutional layer (2x2) or  (3x3), Batch Normalization, Dropout (0.1), and Maxpooling (2x2 kernel, stride=2). Each pool time, the image size was halved. 
- Then, I used linear blocks to gradually reduce the units to the size of the label. 
- I chose Adam optimizer with a learning rate of 0.001 and used CrossEntropyLoss as the loss function. I trained the model for 13 epochs.

## Results:
- After 6 epochs, the model achieved approximately 85.5% accuracy on both the training, validation, and test sets, with a loss value of 0.35.


## Usage: 
- You can train the model by setting "TRAIN" to "True" in config file and your checkpoint will save in "config.CALLBACKS_PATH"
- Then you can predict the images placed in the Prediction folder by setting the "LOAD" and "PREDICTION" values to "True" in the config file.




