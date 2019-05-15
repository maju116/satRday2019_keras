# Ex.1
fashion_mnist_train_Y <- fashion_mnist_train_Y %>% to_categorical(., 10)
fashion_mnist_test_Y <- fashion_mnist_test_Y %>% to_categorical(., 10)

fashion_mnist_train_X <- fashion_mnist_train_X / 255
fashion_mnist_test_X <- fashion_mnist_test_X / 255

fashion_model <- keras_model_sequential() %>%
  layer_dense(units = 512, activation = "relu", input_shape = 784) %>%
  layer_dropout(0.2) %>%
  layer_dense(units = 512, activation = "relu") %>%
  layer_dropout(0.2) %>%
  layer_dense(units = 10, activation = "softmax")

fashion_model %>% compile(
  optimizer = "sgd",
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)

history <- fashion_model %>%
  fit(x = fashion_mnist_train_X,
      y = fashion_mnist_train_Y,
      validation_split = 0.2,
      epochs = 20,
      batch_size = 128)

fashion_model %>% evaluate(fashion_mnist_test_X, fashion_mnist_test_Y)

fashion_predictions <- fashion_model %>% predict_proba(fashion_mnist_test_X)
fashion_predictions_class <- fashion_model %>% predict_classes(fashion_mnist_test_X)

save_model_hdf5(fashion_model, 'fashion_model.hdf5')

# Ex.2
fmnist_model2 <- keras_model_sequential() %>%
  layer_conv_2d(
    filter = 64, kernel_size = c(3, 3), padding = "same",
    input_shape = c(28, 28, 1), activation = "linear") %>%
  layer_batch_normalization() %>%
  layer_activation("relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2), strides = c(2, 2)) %>%
  layer_dropout(0.25) %>%
  layer_flatten() %>%
  layer_dense(512, activation = "relu") %>%
  layer_dropout(0.25) %>%
  layer_dense(10, activation = "softmax")

fmnist_model2 %>% compile(
  loss = loss_categorical_crossentropy,
  optimizer = optimizer_adamax(lr = 0.0001, decay = 1e-6),
  metrics = "accuracy"
)

history <- fmnist_model2 %>% fit(
  fashion_mnist_train_X,
  fashion_mnist_train_Y,
  batch_size = 128,
  epochs = 100,
  validation_split = 0.2,
  callbacks = c(callback_model_checkpoint(monitor = "val_acc",
                                          filepath = "logs/fmnist_model1.weights.{epoch:02d}-{val_acc:.2f}.hdf5"),
                callback_early_stopping(monitor = "val_loss", patience = 5),
                callback_tensorboard(log_dir = "logs"))
)

tensorboard("logs")
