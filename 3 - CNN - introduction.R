library(keras)
library(tidyverse)

# EXAMPLE
# Build CNN for Fashion MNIST dataset
load("data/fashion_mnist.RData")

# Change label vector into using to_categorical() function
fashion_mnist_train_Y <- fashion_mnist_train_Y %>% to_categorical(., 10)
fashion_mnist_test_Y <- fashion_mnist_test_Y %>% to_categorical(., 10)

# Normalization - change pixel range from [0, 255] to [0, 1]
fashion_mnist_train_X <- fashion_mnist_train_X / 255
fashion_mnist_test_X <- fashion_mnist_test_X / 255

# Data transformation into 4D tensor
fashion_mnist_train_X <- array_reshape(fashion_mnist_train_X, c(nrow(fashion_mnist_train_X), 28, 28, 1))
fashion_mnist_test_X <- array_reshape(fashion_mnist_test_X, c(nrow(fashion_mnist_test_X), 28, 28, 1))

# Check the dimmensions of the data
dim(fashion_mnist_train_X)

# Model architecture - in CNN we always start with convolutional layer
fmnist_model1 <- keras_model_sequential() %>%
  # 2D convolution, 32 filters of size 3x3, input (28 ,28 ,1) - grayscale
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = 'relu',
                input_shape = c(28, 28, 1))

fmnist_model1

# Why do we have 320 parameters to train?
32 * (3 * 3 * 1 + 1) # 32 filters of size 3x3(x1) + bias for each of them

# Why output shape looks like (None, 26, 26, 32) ?
((28 - 3 + 2 * 0) / 1) + 1 # 28 - input image size, 3 - kernsl size, 0 - padding, 1 - stride

# Add max pooling layer
fmnist_model1 %>%
  # 2D max pooling size 2x2(x1)
  layer_max_pooling_2d(pool_size = c(2, 2))

fmnist_model1

# Why output shape looks like (None, 12, 12, 32) ?
26 / 2 # 26 - input activation map shape, 2 - pool size,

# Add dense layer (classic one from MLP)
fmnist_model1 %>%
  # Tensor flattening into vector form
  layer_flatten() %>%
  # Output layer - 10 classes, softmax activation
  layer_dense(units = 10, activation = 'softmax')

fmnist_model1

# Why output shape looks like in layer_flatten (None, 5408) ?
13 * 13 * 32 # Zerknij na output poprzedniej warstwy

# Why do we have 54090 parameters to train in last layer ?
5408 * 10 + 10 # 5408 - from layer_flatten * 10 neurons + biases

# Model compilation
fmnist_model1 %>% compile(
  loss = "categorical_crossentropy",
  optimizer = optimizer_adadelta(),
  metrics = c('accuracy')
)

# Training
history <- fmnist_model1 %>% fit(
  fashion_mnist_train_X,
  fashion_mnist_train_Y,
  batch_size = 128,
  epochs = 50,
  validation_split = 0.2,
  callbacks = c(callback_model_checkpoint(monitor = "val_acc", # Value to monitor in checkpoint
                                          filepath = "logs/fmnist_model1.weights.{epoch:02d}-{val_acc:.2f}.hdf5"))
) # 02d - two digits 01, 02, ...; .2f - two numbers after decimal; FORMAT OPTIONS

# Model evaluation
fmnist_model1 %>% evaluate(fashion_mnist_test_X, fashion_mnist_test_Y)

# Ex 2. Expand model by adding batch normalization. Add early stopping and Tensorboard callbacks.
# 1. Model architecture:
# 2D convolution with 64 filters of size 3x3, 1x1 stride, 'linear' activation, "same" padding
# Batch normalization layer
# "relu" activation layer
# 2D max pooling size 2x2, 2x2 stride
# dropout layer with 25% drop rate
# Flattening layer
# dense layer with 512 neurons and "relu" activation
# dropout layer with 25% drop rate
# Choose correct layer as output

# 2. Compile model with Adamax optimizer - set learning rate 0.0001, decay = 1e-6.

# 3. Fit the model - beside standart settings add callbacks:
# model checkpoint - same as in the example
# early stopping - will stop training if there's no progress (monitor "val_loss" and don't wait more than 5 epok) - callback_early_stopping
# tensorboard - save logs  to tensorboard in  "logs" folder - callback_tensorboard

# Evaluate the model on test set
