####    KOBE BRYANT SHOT SELECTION KAGGLE COMPETITION   ####

#Kaggle Score Cutoff <= 0.601

library(tidyverse)
library(tidymodels)
library(vroom)
library(embed)
library(skimr)

setwd("./Kobe")
data <- vroom("data.csv") %>%
  mutate(shot_made_flag = as.factor(shot_made_flag))

data <- data %>%
  mutate(away_game = grepl("@",matchup)) %>%
  mutate(time_remaining = minutes_remaining*60 + seconds_remaining) %>%
  mutate(shot_distance = sqrt((loc_x/10)^2 + (loc_y/10)^2)) %>%
  mutate(game_date = as.numeric(game_date)) %>%
  # mutate(post_achilles = ifelse(game_date > 15807, 1, 0)) %>%
  mutate(season = as.character(season)) %>%
  mutate(season = as.integer(substr(season, nchar(season) - 1, nchar(season)))) %>%
  mutate(loc_x_zero = loc_x == 0) %>%
  mutate(angle = 0)

data$angle[!data$loc_x_zero] = atan(data$loc_y[!data$loc_x_zero] / data$loc_x[!data$loc_x_zero])
data$angle[data$loc_x_zero] = pi / 2

train <- data %>%
  filter(is.na(shot_made_flag) == FALSE)

test <- data %>%
  filter(is.na(shot_made_flag) == TRUE)

recipe <- recipe(shot_made_flag ~ ., data = train) %>%
  step_rm(team_id, team_name, game_id, game_event_id, shot_id,
          shot_zone_range, shot_zone_area, shot_zone_basic, lat, lon,
          loc_x, loc_y, matchup, seconds_remaining, minutes_remaining, loc_x_zero) %>%
  step_mutate_at(c(away_game, period), fn = factor) %>%
  step_dummy(opponent) %>%
  step_novel(all_nominal_predictors()) %>%
  step_unknown(all_nominal_predictors()) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(shot_made_flag)) %>%
  step_normalize(all_numeric_predictors())

baked_data <- bake(prep(recipe), new_data = train)
head(baked_data)

####                 PENALIZED LOGISTIC REGRESSION                 ####

pen_logit_model <- logistic_reg(mixture = tune(),
                                penalty = tune()) %>%
  set_engine("glmnet")

pen_log_wf <- workflow() %>%
  add_recipe(recipe) %>%
  add_model(pen_logit_model)

tuning_grid <- grid_regular(penalty(),
                            mixture(),
                            levels = 5)

folds <- vfold_cv(train, v = 5, repeats=1)

cv_results <- pen_log_wf %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(roc_auc))

bestTune <- cv_results %>%
  select_best()

final_wf <- pen_log_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data = train)

pen_log_preds <- final_wf %>%
  predict(new_data = test, type = "prob")

kag_sub <- pen_log_preds %>%
  bind_cols(., test) %>% #Bind predictions with test data
  select(shot_id, .pred_1) %>% #Just keep datetime and prediction variables
  rename(shot_made_flag = .pred_1) %>% #rename pred to count (for submission to Kaggle)
  mutate(shot_made_flag = pmax(0, shot_made_flag))

vroom_write(x=kag_sub, file="./submission.csv", delim=",")