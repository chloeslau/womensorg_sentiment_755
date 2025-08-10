---
editor: visual
execute:
  echo: true
  output: true
title: Neural Network Predictive Modeling
toc-title: Table of contents
---

-   [Setup](#setup){#toc-setup}
-   [Reading data](#reading-data){#toc-reading-data}
-   [Splits](#splits){#toc-splits}
-   [Data prep](#data-prep){#toc-data-prep}
-   [Fit models](#fit-models){#toc-fit-models}
    -   [Prep and bake](#prep-and-bake){#toc-prep-and-bake}
-   [Model fitting](#model-fitting){#toc-model-fitting}
    -   [Hyperparameter
        tuning](#hyperparameter-tuning){#toc-hyperparameter-tuning}
    -   [Best model fit on
        test](#best-model-fit-on-test){#toc-best-model-fit-on-test}
    -   [Visualization](#visualization){#toc-visualization}

# Setup

::: cell
``` {.r .cell-code}
library(tidyverse)
library(tidymodels)
library(discrim, exclude = "smoothness")
library(MASS, exclude = "select")
library(Matrix, exclude = c("expand", "pack", "unpack"))
library(keras, exclude = "get_weights")
library(magrittr, exclude = c("set_names", "extract"))

cl <- parallel::makePSOCKcluster(parallel::detectCores(logical = FALSE))
doParallel::registerDoParallel(cl)

path_data <- "data"
```
:::

# Reading data

::: cell
``` {.r .cell-code}
data_full <- read_csv(here::here(path_data, "sentiment_women_v5.csv"), show_col_types = FALSE)
```
:::

# Splits

:::: cell
``` {.r .cell-code}
# seed set for reproducibility
set.seed(12345)

data <- data_full |>
  select(date, type, statistics.actual.likeCount:account.name, 
         account.subscriberCount, account.pageCreatedDate, 
         acc_category:account.pageDescription_score) |>
  select(-time_posted)

data <- data |>
  mutate(across(where(is.character), factor))

data <- data |> select(-account.pageCreatedDate)

splits <- data |> initial_split(prop = 2/3, strata = "statistics.actual.likeCount", breaks = 3)

data_trn <- splits |> analysis() |> glimpse() # |> 
```

::: {.cell-output .cell-output-stdout}
    Rows: 18,931
    Columns: 15
    $ date                           <dttm> 2010-01-06 18:04:43, 2010-01-04 20:59:…
    $ type                           <fct> link, status, status, link, link, link,…
    $ statistics.actual.likeCount    <dbl> 0, 0, 0, 1, 1, 2, 0, 0, 2, 0, 0, 0, 0, …
    $ statistics.actual.shareCount   <dbl> 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, …
    $ statistics.actual.commentCount <dbl> 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, …
    $ account.name                   <fct> Wisconsin Women's Network, Wisconsin Wo…
    $ account.subscriberCount        <dbl> 4867, 4867, 4867, 2162, 2162, 2162, 193…
    $ acc_category                   <fct> advocacy, advocacy, advocacy, CHARITY_O…
    $ account_verified_binary        <fct> NA, NA, NA, no, no, no, no, no, no, no,…
    $ title_binary                   <fct> has title, no text, no text, has title,…
    $ caption_binary                 <fct> has caption, no text, no text, has capt…
    $ description_binary             <fct> has desc, no text, no text, has desc, h…
    $ date_posted                    <date> 2010-01-06, 2010-01-04, 2010-01-04, 20…
    $ message_score                  <dbl> 0.4926, 0.0258, 0.8225, 0.0000, 0.9841,…
    $ account.pageDescription_score  <dbl> 0.3400, 0.3400, 0.3400, 0.0000, 0.0000,…
:::

``` {.r .cell-code}
  # write_csv(here::here(path_data, "data_trn.csv"))
  
data_test <- splits |> assessment() # |> 
  # write_csv(here::here(path_data, "data_test.csv"))
```
::::

# Data prep

:::: cell
``` {.r .cell-code}
data_trn <- data_trn |>
  mutate(across(where(is.character), factor))

skimr::skim(data_trn)
```

::: cell-output-display
  -------------------------------------------------- ----------
  Name                                               data_trn
  Number of rows                                     18931
  Number of columns                                  15
  \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_     
  Column type frequency:                             
  Date                                               1
  factor                                             7
  numeric                                            6
  POSIXct                                            1
  \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_   
  Group variables                                    None
  -------------------------------------------------- ----------

  : Data summary

**Variable type: Date**

  ---------------------------------------------------------------------------------------------
  skim_variable     n_missing   complete_rate min          max          median         n_unique
  --------------- ----------- --------------- ------------ ------------ ------------ ----------
  date_posted               0               1 2009-02-23   2024-08-10   2019-11-01         4515

  ---------------------------------------------------------------------------------------------

**Variable type: factor**

  -------------------------------------------------------------------------------------------------------
  skim_variable               n_missing   complete_rate ordered     n_unique top_counts
  ------------------------- ----------- --------------- --------- ---------- ----------------------------
  type                                0            1.00 FALSE              9 pho: 9595, lin: 6359, sta:
                                                                             1289, nat: 909

  account.name                        0            1.00 FALSE             17 Lea: 2788, Gir: 2010, Wis:
                                                                             1931, The: 1837

  acc_category                      459            0.98 FALSE              9 NON: 5652, Oth: 4461, POL:
                                                                             2788, adv: 1931

  account_verified_binary          2390            0.87 FALSE              2 no: 12080, yes: 4461

  title_binary                        0            1.00 FALSE              2 no : 12387, has: 6544

  caption_binary                      0            1.00 FALSE              2 no : 12415, has: 6516

  description_binary                  0            1.00 FALSE              2 no : 11165, has: 7766
  -------------------------------------------------------------------------------------------------------

**Variable type: numeric**

  --------------------------------------------------------------------------------------------------------------------------------------
  skim_variable                      n_missing   complete_rate      mean        sd    p0       p25       p50       p75      p100 hist
  -------------------------------- ----------- --------------- --------- --------- ----- --------- --------- --------- --------- -------
  statistics.actual.likeCount                0               1      9.06     62.41     0      1.00      3.00      9.00   8100.00 ▇▁▁▁▁

  statistics.actual.shareCount               0               1      3.65     39.84     0      0.00      0.00      2.00   3016.00 ▇▁▁▁▁

  statistics.actual.commentCount             0               1      0.99     37.75     0      0.00      0.00      0.00   5166.00 ▇▁▁▁▁

  account.subscriberCount                    0               1   4735.25   2738.76   193   2310.00   4061.00   6185.00   9855.00 ▂▇▅▃▃

  message_score                              0               1      0.48      0.45    -1      0.00      0.62      0.88      1.00 ▁▁▅▂▇

  account.pageDescription_score              0               1      0.41      0.26     0      0.32      0.49      0.54      0.82 ▆▁▅▇▅
  --------------------------------------------------------------------------------------------------------------------------------------

**Variable type: POSIXct**

  ------------------------------------------------------------------------------------------------
  skim_variable     n_missing   complete_rate min           max           median          n_unique
  --------------- ----------- --------------- ------------- ------------- ------------- ----------
  date                      0               1 2009-02-23    2024-08-10    2019-11-01         18917
                                              17:43:55      21:00:11      01:00:56      

  ------------------------------------------------------------------------------------------------
:::
::::

# Fit models

## Prep and bake

::: cell
``` {.r .cell-code}
rec <- recipe(statistics.actual.likeCount ~ ., data = data_trn) |>
  # handle datetime variables
  step_time(date, features = c("hour", "minute", "second")) |>
  step_date(date_posted, features = c("year", "month", "mday")) |>
  step_rm(date) |>
  step_rm(date_posted) |>
  # handle missing data
  step_impute_mode(acc_category) |>
  step_impute_mode(account_verified_binary) |>
  step_rm(account_verified_binary) |>
  # normalization
  step_zv(all_predictors()) |>
  step_normalize(all_numeric_predictors()) |>
  step_dummy(all_factor_predictors()) |>
  #decorrelating (important due to account groupings)
  step_pca(all_numeric_predictors())

rec_prep <- rec |>
  prep(data_trn)

feat_trn <- rec_prep |>
  bake(NULL)
```
:::

# Model fitting

::: cell
``` {.r .cell-code}
set.seed(12345)
fit_seeds <- sample.int(10^5, size = 3)
```
:::

## Hyperparameter tuning

::::: cell
``` {.r .cell-code}
set.seed(12345)
splits_val <-
  data_trn |> 
  validation_split(prop = 4/5)

grid_keras <- expand_grid(hidden_units = c(10, 30, 50, 70),
                          activation = c("linear", "softmax", "relu"), 
                          penalty = c(.00001, .0001, .01))

fits_nn <-
  mlp(hidden_units = tune(), activation = tune(), penalty = tune()) |>
  set_mode("regression") |> 
  set_engine("keras", verbose = 0, seeds = fit_seeds) |>
  tune_grid(preprocessor = rec,
              grid = grid_keras,
              resamples = splits_val,
              metrics = metric_set(rmse)
              )
```

::: {.cell-output .cell-output-stdout}
    119/119 - 0s - 290ms/epoch - 2ms/step
    119/119 - 0s - 327ms/epoch - 3ms/step
    119/119 - 0s - 225ms/epoch - 2ms/step
    119/119 - 0s - 246ms/epoch - 2ms/step
    119/119 - 1s - 677ms/epoch - 6ms/step
    119/119 - 0s - 236ms/epoch - 2ms/step
    119/119 - 0s - 241ms/epoch - 2ms/step
    119/119 - 0s - 191ms/epoch - 2ms/step
    119/119 - 0s - 269ms/epoch - 2ms/step
    119/119 - 0s - 235ms/epoch - 2ms/step
    119/119 - 0s - 177ms/epoch - 1ms/step
    119/119 - 0s - 315ms/epoch - 3ms/step
    119/119 - 1s - 649ms/epoch - 5ms/step
    119/119 - 1s - 567ms/epoch - 5ms/step
    119/119 - 0s - 206ms/epoch - 2ms/step
    119/119 - 0s - 229ms/epoch - 2ms/step
    119/119 - 0s - 207ms/epoch - 2ms/step
    119/119 - 0s - 191ms/epoch - 2ms/step
    119/119 - 0s - 191ms/epoch - 2ms/step
    119/119 - 0s - 210ms/epoch - 2ms/step
    119/119 - 1s - 639ms/epoch - 5ms/step
    119/119 - 0s - 325ms/epoch - 3ms/step
    119/119 - 0s - 209ms/epoch - 2ms/step
    119/119 - 0s - 307ms/epoch - 3ms/step
    119/119 - 0s - 174ms/epoch - 1ms/step
    119/119 - 0s - 222ms/epoch - 2ms/step
    119/119 - 0s - 354ms/epoch - 3ms/step
    119/119 - 0s - 190ms/epoch - 2ms/step
    119/119 - 0s - 177ms/epoch - 1ms/step
    119/119 - 0s - 205ms/epoch - 2ms/step
    119/119 - 0s - 332ms/epoch - 3ms/step
    119/119 - 0s - 222ms/epoch - 2ms/step
    119/119 - 0s - 279ms/epoch - 2ms/step
    119/119 - 0s - 224ms/epoch - 2ms/step
    119/119 - 0s - 222ms/epoch - 2ms/step
    119/119 - 0s - 166ms/epoch - 1ms/step
:::

``` {.r .cell-code}
(best_fits <- show_best(fits_nn))
```

::: {.cell-output .cell-output-stdout}
    # A tibble: 5 × 9
      hidden_units penalty activation .metric .estimator  mean     n std_err .config
             <dbl>   <dbl> <chr>      <chr>   <chr>      <dbl> <int>   <dbl> <chr>  
    1           30 0.00001 relu       rmse    standard    15.2     1      NA Prepro…
    2           50 0.01    relu       rmse    standard    15.6     1      NA Prepro…
    3           30 0.0001  relu       rmse    standard    16.0     1      NA Prepro…
    4           50 0.00001 relu       rmse    standard    16.9     1      NA Prepro…
    5           10 0.0001  relu       rmse    standard    17.9     1      NA Prepro…
:::

``` {.r .cell-code}
# best_fits |> write_csv(here::here(path_data, "best_fits.csv"))
```
:::::

:::::: cell
``` {.r .cell-code}
devtools::source_url("https://github.com/jjcurtin/lab_support/blob/main/fun_ml.R?raw=true")
```

::: {.cell-output .cell-output-stderr}
    ℹ SHA-1 hash of file is "32a0bc8ced92c79756b56ddcdc9a06e639795da6"
:::

``` {.r .cell-code}
plot_hyperparameters(fits_nn, hp1 = "penalty", metric = "rmse")
```

::: cell-output-display
![](neural_network_models_files/figure-markdown/unnamed-chunk-8-1.png)
:::

``` {.r .cell-code}
plot_hyperparameters(fits_nn, hp1 = "penalty", hp2 = "hidden_units", metric = "rmse")
```

::: cell-output-display
![](neural_network_models_files/figure-markdown/unnamed-chunk-8-2.png)
:::
::::::

## Best model fit on test

::: cell
``` {.r .cell-code}
best_fit_nn <- 
  mlp(hidden_units = 10, activation = "relu", penalty = 0.01) |>
  set_mode("regression") |> 
  set_engine("keras", verbose = 0, seeds = fit_seeds) |>
  fit(statistics.actual.likeCount ~ ., data = feat_trn)
```
:::

::::::: cell
``` {.r .cell-code}
feat_test <- rec_prep |> bake(data_test)

preds = predict(best_fit_nn, feat_test)$.pred
```

::: {.cell-output .cell-output-stdout}
    296/296 - 1s - 589ms/epoch - 2ms/step
:::

``` {.r .cell-code}
test_preds <- predict(best_fit_nn, new_data = feat_test)
```

::: {.cell-output .cell-output-stdout}
    296/296 - 0s - 409ms/epoch - 1ms/step
:::

``` {.r .cell-code}
test_preds$.pred <- pmax(test_preds$.pred, 0)
# since likes count does not go below zero, we have to bound negative predictions

test_preds <- test_preds %>%
  bind_cols(data_test %>% select(statistics.actual.likeCount))

head(test_preds, 5)
```

::: {.cell-output .cell-output-stdout}
    # A tibble: 5 × 2
      .pred statistics.actual.likeCount
      <dbl>                       <dbl>
    1 0.194                           0
    2 2.62                            2
    3 1.45                            0
    4 1.12                            0
    5 0                               0
:::

``` {.r .cell-code}
test_metrics <- test_preds %>%
  metrics(truth = statistics.actual.likeCount, estimate = .pred)

(test_metrics)
```

::: {.cell-output .cell-output-stdout}
    # A tibble: 3 × 3
      .metric .estimator .estimate
      <chr>   <chr>          <dbl>
    1 rmse    standard      15.9  
    2 rsq     standard       0.395
    3 mae     standard       7.12 
:::

``` {.r .cell-code}
# test_metrics |> write_csv(here::here(path_data, "test_metrics.csv"))
```
:::::::

## Visualization

:::: cell
``` {.r .cell-code}
plot_truth(truth = feat_test$statistics.actual.likeCount, 
           estimate = preds)
```

::: cell-output-display
![](neural_network_models_files/figure-markdown/unnamed-chunk-11-1.png)
:::
::::

> Not particularly meaningful due to the cluster of outliers in the
> extremes (viral posts), and the vast majority of posts have less than
> 10 likes. However, this is generally a decent model; the training and
> test accuracy metrics are quite close, meaning there is no severe
> overfitting. The Root Mean Square Error is 23.4, which is the average
> difference in likes between the predicted and outcome. This model can
> be used on pre-written social media posts, to help predict the
> engagement that they will recieve based on each account's statistics
> and the specifications of the post.
