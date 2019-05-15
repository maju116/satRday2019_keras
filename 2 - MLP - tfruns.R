library(keras)
library(tidyverse)
library(tfruns)

# EXAMPLE
# Hyperparameter optimization - tfruns package
load("data/german.RData")

# Define training runs
runs <- tuning_run("2b - tfruns - example.R", flags = list(
  dropout = c(0.2, 0.3, 0.4),
  batch_size = c(10, 50, 100, 200, 400, 700)
))

# Results
runs

# Best results
runs %>% arrange(desc(metric_val_acc)) %>%
  select(run_dir, metric_acc, metric_val_acc, flag_dropout, flag_batch_size) -> runs_order
runs_order

# Go to 'runs' folder (you can set the name of this folder and full path)

# Results for latest run
latest_run()

# Results for selected run (change path after model fitting)
view_run("runs/2019-05-15T18-36-03Z")

# Comparition of two selected runs
compare_runs(runs = runs_order$run_dir[1:2])
