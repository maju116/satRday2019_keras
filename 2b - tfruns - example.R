# Create 'flags' - hyperparameters we want to check for different values
FLAGS <- flags(
  flag_numeric("dropout", default = 0.3),
  flag_numeric("batch_size", default = 50)
)

# Define model with flags
model <- keras_model_sequential() %>%
  layer_dense(units = 24, activation = "relu", input_shape = c(24)) %>%
  layer_dropout(FLAGS$dropout) %>% # 'dropout' flag
  layer_dense(units = 1, activation = "sigmoid") %>%
  compile(
    optimizer = 'sgd',
    loss = "binary_crossentropy",
    metrics = "accuracy")

fit(model,
    x = g_train_X,
    y = g_train_Y,
    epochs = 100,
    batch_size = FLAGS$batch_size, # 'batch_size' flag
    view_metrics = FALSE,
    validation_split = 0.2)
