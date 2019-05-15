library(keras)
library(tidyverse)

# PRZYKŁAD
# Zdefiniuj scieżki ze zdjęciami
train_dir <- "data/alien-vs-predator/train/"
validation_dir <- "data/alien-vs-predator/validation"
test_dir <- "data/alien-vs-predator/test"

# Create data generator (with data augumantation) for train set
train_datagen <- image_data_generator(
  rescale = 1/255, # changes pixel range from [0, 255] to [0, 1]
  rotation_range = 35,
  width_shift_range = 0.3,
  height_shift_range = 0.3,
  zoom_range = 0.2,
  horizontal_flip = TRUE
)

# Create data generator (without data augumantation) for validation set
validation_datagen <- image_data_generator(rescale = 1/255) # Dlaczego tylko to ?

# Create flow for train set
train_flow <- flow_images_from_directory(
  directory = train_dir, # Path for train images folder
  generator = train_datagen, # Generator
  color_mode = "rgb", # Images are in color
  target_size = c(150, 150), # Scale all images to 150x150
  batch_size = 32, # Batch size
  class_mode = "categorical" # Classification task
)

# Create flow for validation set
validation_flow <- flow_images_from_directory(
  directory = validation_dir,
  generator = validation_datagen,
  color_mode = "rgb",
  target_size = c(150, 150),
  batch_size = 20,
  class_mode = "categorical"
)

# We can see a sampl augmented batch of images
batch <- generator_next(train_flow)

# Plotujemy pierwsze 4 zdjęcia
for (i in 1:4) {
  plot(as.raster(batch[[1]][i,,,]))
}

# Use VGG16 model as a base
conv_base <- application_vgg16(
  weights = "imagenet", # Weights trained on 'imagenet'
  include_top = FALSE, # Without dense layers on top - we will add them later
  input_shape = c(150, 150, 3) # Same shape as in our generators
)

conv_base

# Freeze weights on the botttom (they won't change in optimization processs)
freeze_weights(conv_base, from = "block1_conv1", to = "block2_pool")

# Final model with VGG16 base
model <- keras_model_sequential() %>%
  conv_base %>%
  layer_flatten() %>%
  layer_dense(units = 256, activation = "relu") %>%
  layer_dense(units = 2, activation = "softmax")

# Compilation
model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = optimizer_rmsprop(lr = 1e-5), # Small lr for fine-tuning
  metrics = c("accuracy"))

model # Take a look at 'Non-trainable params'

# Training
history <- model %>% fit_generator(
  train_flow,
  steps_per_epoch = 22,
  epochs = 15,
  validation_data = validation_flow,
  validation_steps = 6
)

# Create data generator (without data augumantation) for test set
test_datagen <- image_data_generator(rescale = 1/255)

# Create flow for test set
test_generator <- flow_images_from_directory(
  test_dir,
  test_datagen,
  target_size = c(150, 150),
  color_mode = "rgb",
  batch_size = 1,
  class_mode = "categorical"
)

# Evaluate on test set
model %>% evaluate_generator(test_generator,
                             steps = 18)

# Predictions
alien_predator_predictions <- model %>%
  predict_generator(
    test_generator,
    steps = 18)

# Save model
save_model_hdf5(model, "alien_predator_model.hdf5")
