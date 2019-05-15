library(keras)
library(tidyverse)
library(lime)

# Load model created in previous task
model <- load_model_hdf5("alien_predator_model.hdf5")

# Split image into superpixels
test_dir <- "data/alien-vs-predator/test"
test_images <- list.files(test_dir, recursive = TRUE, pattern = ".jpg", full.names = TRUE)
plot_superpixels(test_images[1], # Image
                 n_superpixels = 50) # Superpixels number

# Helper function for image reading
klasy <- c('1' = 'alien', '2' = 'predator')
image_prep <- function(x) {
  arrays <- lapply(x, function(path) {
    img <- image_load(path, target_size = c(150, 150)) # Load image into R
    x <- image_to_array(img) # Transformation into array
    x <- reticulate::array_reshape(x, c(1, dim(x))) # Reshaping into tensor
    x <- x / 255 # Normalization
  })
  do.call(abind::abind, c(arrays, list(along = 1)))
}

# Create expplainer object
explainer <- lime(c(test_images[1], test_images[11]), # Images we want to explain
                  as_classifier(model, klasy), # Model
                  image_prep) # Helper function
explanation <- explain(c(test_images[1], test_images[11]), # Images we want to explain
                       explainer, # explainer
                       n_labels = 1, # Number of classes to explain (n best)
                       n_features = 20, # Number of predictors used to explain
                       n_superpixels = 50, # Number of superpixels
                       background = "white")

# Feature importance
plot_features(explanation, ncol = 2)

# Feature importance on the image
plot_image_explanation(explanation %>% filter(case == '91.jpg'),
                       display = 'outline', threshold = 0.001,
                       show_negative = TRUE)
plot_image_explanation(explanation %>% filter(case == '92.jpg'),
                       display = 'outline', threshold = 0.0001,
                       show_negative = TRUE)

# Based on https://shirinsplayground.netlify.com/2018/06/keras_fruits_lime/
