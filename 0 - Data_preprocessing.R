library(tidyverse)

# Boston housing (https://www.kaggle.com/mlg-ulb/creditcardfraud)
boston <- dataset_boston_housing(path = "boston_housing.npz", test_split = 0.2, seed = 113L)
train_X_mean <- boston$train$x %>% apply(., 2, mean)
train_X_sd <- boston$train$x %>% apply(., 2, sd)
boston_train_X <- boston$train$x %>% sweep(., 2, train_X_mean, "-") %>% sweep(., 2, train_X_sd, "/")
boston_train_Y <- boston$train$y
boston_test_X <- boston$test$x %>% sweep(., 2, train_X_mean, "-") %>% sweep(., 2, train_X_sd, "/")
boston_test_Y <- boston$test$y
save(file = "data/boston.RData",
     list = c("boston_train_X", "boston_train_Y",
              "boston_test_X", "boston_test_Y"))

# Fashion MNIST (https://www.kaggle.com/zalando-research/fashionmnist)
fashion_mnist_train <- read_csv("data/fashion-mnist_train.csv")
fashion_mnist_test <- read_csv("data/fashion-mnist_test.csv")
fashion_mnist_train_X <- fashion_mnist_train %>% select(-label) %>% as.matrix()
fashion_mnist_train_Y <- fashion_mnist_train %>% pull(label)
fashion_mnist_test_X <- fashion_mnist_test %>% select(-label) %>% as.matrix()
fashion_mnist_test_Y <- fashion_mnist_test %>% pull(label)
save(file = "data/fashion_mnist.RData",
     list = c("fashion_mnist_train_X", "fashion_mnist_train_Y",
              "fashion_mnist_test_X", "fashion_mnist_test_Y"))

# German1
g <- read.csv('german1.csv')
g$target <- g$target - 1
g_train_Y <- g$target
g_train_X <- as.matrix(g[,-25])
save(file = "data/german.RData", list = c("g_train_Y", "g_train_X"))
