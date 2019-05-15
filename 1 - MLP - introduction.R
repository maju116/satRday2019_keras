library(keras)
library(tidyverse)

# EXAMPLE
# Load train and test set (data preparation in Data_preprocessing.R)
load("data/boston.RData")

# Check the dimmensions of the data
boston_train_X %>% dim()
boston_train_Y %>% dim()

# Model initialization: keras_model_sequential()
boston_model <- keras_model_sequential()

# Add layer_dense() with 16 neurons and tanh ass activation function - hidden layer
boston_model %>%
  layer_dense(units = 16, activation = "tanh", input_shape = c(13)) # input_shape has to be added in first layer

boston_model

# Why do we have 224 parameters to train?
13 * 16 + 16 # 13 predictors * 16 neurons + biases

# Add another layer_dense() with 1 neuron and linear activation function - output layer
boston_model %>%
  layer_dense(units = 1, activation = "linear")

boston_model

# Model compilation, set optimizer, loss function and additional metrics.
boston_model %>% compile(
  optimizer = "sgd", # Stochastic Gradient Descent as optimizer
  loss = "mse", # Mean Square Error as loss
  metrics = c("mae") # Absolute Error as additional metric
)

# Training
history <- boston_model %>%
  fit(x = boston_train_X, # Predictors
      y = boston_train_Y, # Predicted variables
      validation_split = 0.2, # 20% for validation set
      epochs = 100, # Number of epochs - how many times optimizer will se all dataset (1 epoch = all batches)
      batch_size = 30) # Batch size - in our case (404 observations * 0.8) / 30 = 10.77333 = 11 batches

# Evaluation on test set
boston_model %>%
  evaluate(boston_test_X, boston_test_Y)

# Predictions
boston_predictions <- boston_model %>% predict(boston_test_X)

# Model saving in hdf5
save_model_hdf5(boston_model, "boston_model.hdf5")

# Ex.1 - Create MLP for Fashion MNIST dataset - 10-class classsification (data preparation in Data_preprocessing.R)
load("data/fashion_mnist.RData")

# 1. Change label vector into one-hot-encoding using to_categorical() function

# 2. Normalization - change pixel range from [0, 255] into [0, 1]

# 3. Create MLP described by architecture:
# Dense layer with 512 neurons and "relu" activation (layer_dense)
# Dropout layer with 20% drop rate (layer_dropout)
# Dense layer with 512 neurons and "relu" activation (layer_dense)
# Dropout layer with 20% drop rate (layer_dropout)
# Dense layer for output (how many neurons and which activation you should use?)

# 4. Compile model using SGD and categorical_crossentropy as loss. Select "accuracy" as additional metric.

# 5. Train the model. Use 20% of train set for validation. Use 20 epochs and set batch size to 128 observations.

# 6. Evaluate on test set

# 7. Calculate predictions for test set (use predict_proba i predict_classes functions)

# 8. Save model as 'fashion_model.hdf5'
