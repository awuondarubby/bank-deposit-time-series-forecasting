# =============================================================================
# Project: Forecasting Total Deposits at U.S. Commercial Banks
# Course: DSCI 725 Data Mining for Competitive Advantage
# Student: Awuonda Rubby
# Semester: Spring 2026
# Data: FRED series DPSACBW027SBOG
# =============================================================================

# Clear workspace and set seed for reproducibility
rm(list = ls())
set.seed(123)

# Load required libraries
library(tidyverse)
library(forecast)
library(tseries)

# =============================================================================
# STEP 1: LOAD AND PREPARE DATA
# =============================================================================

deposits_raw <- read.csv("DPSACBW027SBOG.csv")
deposits_raw$observation_date <- as.Date(deposits_raw$observation_date)

# Replace missing value using interpolation
deposits <- deposits_raw %>%
  mutate(DPSACBW027SBOG = na.interp(DPSACBW027SBOG))

# Convert to monthly time series
deposits_ts <- ts(
  deposits$DPSACBW027SBOG,
  start = c(1973, 1),
  frequency = 12
)

# =============================================================================
# STEP 2: VISUALIZE ORIGINAL SERIES
# =============================================================================

p1 <- autoplot(deposits_ts) +
  ggtitle("Total Deposits at All U.S. Commercial Banks") +
  xlab("Year") +
  ylab("Deposits (Billions USD)") +
  theme_minimal()

print(p1)
ggsave("Figure1_Total_Deposits.png", plot = p1, width = 8, height = 5, dpi = 300)

# =============================================================================
# STEP 3: STATIONARITY TESTING
# =============================================================================

adf_original <- adf.test(deposits_ts)
adf_diff1 <- adf.test(diff(deposits_ts, differences = 1))
adf_diff2 <- adf.test(diff(deposits_ts, differences = 2))

adf_table <- data.frame(
  Series = c("Original series", "First differenced series", "Second differenced series"),
  ADF_Statistic = round(c(adf_original$statistic, adf_diff1$statistic, adf_diff2$statistic), 4),
  p_value = round(c(adf_original$p.value, adf_diff1$p.value, adf_diff2$p.value), 4),
  Interpretation = c(
    ifelse(adf_original$p.value > 0.05, "Non-stationary", "Stationary"),
    ifelse(adf_diff1$p.value > 0.05, "Non-stationary", "Stationary"),
    ifelse(adf_diff2$p.value > 0.05, "Non-stationary", "Stationary")
  )
)

print(adf_table)
write.csv(adf_table, "adf_results.csv", row.names = FALSE)

deposits_diff1 <- diff(deposits_ts, differences = 1)
deposits_diff2 <- diff(deposits_ts, differences = 2)

# =============================================================================
# STEP 4: ACF AND PACF ANALYSIS
# =============================================================================

png("Figure2_ACF_First_Difference.png", width = 8, height = 5, units = "in", res = 300)
Acf(deposits_diff1, lag.max = 36,
    main = "ACF of First Differenced Deposits Series")
dev.off()

png("Figure3_ACF_Second_Difference.png", width = 8, height = 5, units = "in", res = 300)
Acf(deposits_diff2, lag.max = 36,
    main = "ACF of Second Differenced Deposits Series")
dev.off()

png("Figure4_PACF_First_Difference.png", width = 8, height = 5, units = "in", res = 300)
Pacf(deposits_diff1, lag.max = 36,
     main = "PACF of First Differenced Deposits Series")
dev.off()

png("Figure5_PACF_Second_Difference.png", width = 8, height = 5, units = "in", res = 300)
Pacf(deposits_diff2, lag.max = 36,
     main = "PACF of Second Differenced Deposits Series")
dev.off()

# =============================================================================
# STEP 5: TRAINING AND VALIDATION SPLIT
# =============================================================================

n_total <- length(deposits_ts)
train_end <- floor(0.8 * n_total)

train_ts <- window(deposits_ts, end = time(deposits_ts)[train_end])
valid_ts <- window(deposits_ts, start = time(deposits_ts)[train_end + 1])

cat("Training observations:", length(train_ts), "\n")
cat("Validation observations:", length(valid_ts), "\n")

h_valid <- length(valid_ts)

# =============================================================================
# STEP 6: FIT FORECASTING MODELS
# =============================================================================

meanf_mod <- meanf(train_ts, h = h_valid)
naive_mod <- naive(train_ts, h = h_valid)
snaive_mod <- snaive(train_ts, h = h_valid)

tslm_mod <- tslm(train_ts ~ trend + season)
tslm_fc <- forecast(tslm_mod, h = h_valid)

ets_mod <- ets(train_ts)
ets_fc <- forecast(ets_mod, h = h_valid)

# ARIMA with first differencing forced
arima_d1_mod <- auto.arima(
  train_ts,
  d = 1,
  stepwise = FALSE,
  approximation = FALSE,
  ic = "aicc",
  allowdrift = TRUE
)

arima_d1_fc <- forecast(arima_d1_mod, h = h_valid)

# ARIMA with second differencing forced
arima_d2_mod <- auto.arima(
  train_ts,
  d = 2,
  stepwise = FALSE,
  approximation = FALSE,
  ic = "aicc",
  allowdrift = TRUE
)

arima_d2_fc <- forecast(arima_d2_mod, h = h_valid)

cat("\nARIMA d = 1 model:\n")
print(arima_d1_mod)
cat("AICc d = 1:", arima_d1_mod$aicc, "\n")

cat("\nARIMA d = 2 model:\n")
print(arima_d2_mod)
cat("AICc d = 2:", arima_d2_mod$aicc, "\n")

# =============================================================================
# STEP 7: MODEL ACCURACY COMPARISON
# =============================================================================

acc_meanf <- accuracy(meanf_mod, valid_ts)
acc_naive <- accuracy(naive_mod, valid_ts)
acc_snaive <- accuracy(snaive_mod, valid_ts)
acc_tslm <- accuracy(tslm_fc, valid_ts)
acc_ets <- accuracy(ets_fc, valid_ts)
acc_arima_d1 <- accuracy(arima_d1_fc, valid_ts)
acc_arima_d2 <- accuracy(arima_d2_fc, valid_ts)

accuracy_table <- bind_rows(
  data.frame(Model = "Mean", Dataset = rownames(acc_meanf), acc_meanf),
  data.frame(Model = "Naive", Dataset = rownames(acc_naive), acc_naive),
  data.frame(Model = "Seasonal Naive", Dataset = rownames(acc_snaive), acc_snaive),
  data.frame(Model = "Regression", Dataset = rownames(acc_tslm), acc_tslm),
  data.frame(Model = "ETS", Dataset = rownames(acc_ets), acc_ets),
  data.frame(Model = "ARIMA d = 1", Dataset = rownames(acc_arima_d1), acc_arima_d1),
  data.frame(Model = "ARIMA d = 2", Dataset = rownames(acc_arima_d2), acc_arima_d2)
) %>%
  mutate(
    Dataset = recode(
      Dataset,
      "Training set" = "Training Set",
      "Test set" = "Validation Set"
    )
  ) %>%
  mutate(across(where(is.numeric), ~ round(.x, 3)))

rownames(accuracy_table) <- NULL

print(accuracy_table)
write.csv(accuracy_table, "accuracy_comparison.csv", row.names = FALSE)

validation_accuracy <- accuracy_table %>%
  filter(Dataset == "Validation Set") %>%
  arrange(MASE)

print(validation_accuracy)
write.csv(validation_accuracy, "validation_accuracy_ranked.csv", row.names = FALSE)

# =============================================================================
# STEP 8: MODEL COMPARISON VISUALIZATIONS
# =============================================================================

# Detailed comparison showing d = 1 and d = 2 ARIMA models
p_d_compare <- autoplot(deposits_ts, series = "Actual") +
  autolayer(fitted(tslm_mod), series = "Regression fitted") +
  autolayer(fitted(ets_mod), series = "ETS fitted") +
  autolayer(fitted(arima_d1_mod), series = "ARIMA d = 1 fitted") +
  autolayer(fitted(arima_d2_mod), series = "ARIMA d = 2 fitted") +
  autolayer(tslm_fc$mean, series = "Regression forecast") +
  autolayer(ets_fc$mean, series = "ETS forecast") +
  autolayer(arima_d1_fc$mean, series = "ARIMA d = 1 forecast") +
  autolayer(arima_d2_fc$mean, series = "ARIMA d = 2 forecast") +
  autolayer(naive_mod$mean, series = "Naive forecast") +
  ggtitle("Model Comparison Including ARIMA d = 1 and d = 2") +
  xlab("Year") +
  ylab("Deposits (Billions USD)") +
  theme_minimal()

print(p_d_compare)
ggsave("Figure6_Model_Comparison_d1_d2.png",
       plot = p_d_compare, width = 10, height = 6, dpi = 300)

# Clean final report comparison using the selected ARIMA d = 2 model
p_clean_compare <- autoplot(deposits_ts, series = "Actual") +
  autolayer(fitted(tslm_mod), series = "Regression fitted") +
  autolayer(fitted(ets_mod), series = "ETS fitted") +
  autolayer(fitted(arima_d2_mod), series = "ARIMA fitted") +
  autolayer(tslm_fc$mean, series = "Regression forecast") +
  autolayer(ets_fc$mean, series = "ETS forecast") +
  autolayer(arima_d2_fc$mean, series = "ARIMA forecast") +
  autolayer(naive_mod$mean, series = "Naive forecast") +
  ggtitle("Actual, Fitted, and Forecast Values Across Model Families") +
  xlab("Year") +
  ylab("Deposits (Billions USD)") +
  theme_minimal()

print(p_clean_compare)
ggsave("Figure7_Clean_Model_Comparison.png",
       plot = p_clean_compare, width = 10, height = 6, dpi = 300)

# =============================================================================
# STEP 9: RESIDUAL DIAGNOSTICS FOR d = 1 AND d = 2 ARIMA MODELS
# =============================================================================

png("Figure8_Residuals_ARIMA_d1.png",
    width = 10, height = 6, units = "in", res = 300)
checkresiduals(arima_d1_mod)
dev.off()

png("Figure9_Residuals_ARIMA_d2.png",
    width = 10, height = 6, units = "in", res = 300)
checkresiduals(arima_d2_mod)
dev.off()

lb_d1 <- Box.test(
  residuals(arima_d1_mod),
  lag = 24,
  type = "Ljung-Box",
  fitdf = sum(arimaorder(arima_d1_mod)[c(1, 3, 4, 6)])
)

lb_d2 <- Box.test(
  residuals(arima_d2_mod),
  lag = 24,
  type = "Ljung-Box",
  fitdf = sum(arimaorder(arima_d2_mod)[c(1, 3, 4, 6)])
)

diagnostic_table <- data.frame(
  Model = c("ARIMA d = 1", "ARIMA d = 2"),
  Ljung_Box_Statistic = round(c(lb_d1$statistic, lb_d2$statistic), 4),
  p_value = round(c(lb_d1$p.value, lb_d2$p.value), 4),
  AICc = round(c(arima_d1_mod$aicc, arima_d2_mod$aicc), 3)
)

print(diagnostic_table)
write.csv(diagnostic_table, "arima_diagnostics_d1_d2.csv", row.names = FALSE)

# =============================================================================
# STEP 10: FINAL MODEL SELECTION
# =============================================================================
# The ADF test indicated that first differencing achieved stationarity.
# However, the d = 2 ARIMA model produced better validation accuracy,
# lower AICc, and better residual diagnostics than the d = 1 ARIMA model.
# Therefore, the final model is selected using d = 2.

final_model <- auto.arima(
  deposits_ts,
  d = 2,
  stepwise = FALSE,
  approximation = FALSE,
  ic = "aicc",
  allowdrift = TRUE
)

summary(final_model)

png("Figure10_Final_Model_Residuals.png",
    width = 10, height = 6, units = "in", res = 300)
checkresiduals(final_model)
dev.off()

lb_final <- Box.test(
  residuals(final_model),
  lag = 24,
  type = "Ljung-Box",
  fitdf = sum(arimaorder(final_model)[c(1, 3, 4, 6)])
)

cat("\nFinal ARIMA model:\n")
print(final_model)

cat("\nLjung-Box final model statistic:",
    round(lb_final$statistic, 4),
    "| p-value:",
    round(lb_final$p.value, 4), "\n")

# =============================================================================
# STEP 11: GENERATE 24-MONTH FORECAST
# =============================================================================

fc_24 <- forecast(final_model, h = 24, level = c(80, 95))

p_forecast <- autoplot(fc_24) +
  ggtitle("24 Month Forecast of U.S. Commercial Bank Deposits") +
  xlab("Year") +
  ylab("Deposits (Billions USD)") +
  theme_minimal()

print(p_forecast)
ggsave("Figure11_24Month_Forecast.png",
       plot = p_forecast, width = 8, height = 5, dpi = 300)

# =============================================================================
# STEP 12: BUILD 24-MONTH FORECAST TABLE
# =============================================================================

last_obs_year <- end(deposits_ts)[1]
last_obs_month <- end(deposits_ts)[2]

if (last_obs_month == 12) {
  first_fc_date <- as.Date(paste(last_obs_year + 1, "01", "01", sep = "-"))
} else {
  first_fc_date <- as.Date(paste(last_obs_year, last_obs_month + 1, "01", sep = "-"))
}

forecast_months <- format(
  seq(first_fc_date, by = "month", length.out = 24), "%b %Y"
)

future_table <- data.frame(
  Month = forecast_months,
  Forecast = round(as.numeric(fc_24$mean), 0),
  Lower_80 = round(as.numeric(fc_24$lower[, 1]), 0),
  Upper_80 = round(as.numeric(fc_24$upper[, 1]), 0),
  Lower_95 = round(as.numeric(fc_24$lower[, 2]), 0),
  Upper_95 = round(as.numeric(fc_24$upper[, 2]), 0)
)

names(future_table) <- c(
  "Month", "Forecast", "Lower 80%",
  "Upper 80%", "Lower 95%", "Upper 95%"
)

print(future_table)
write.csv(future_table, "forecast_24_months_final.csv", row.names = FALSE)

# =============================================================================
# STEP 13: EXPORT FINAL MODEL SUMMARY
# =============================================================================

final_summary <- data.frame(
  Final_Model = as.character(final_model),
  Ljung_Box_Statistic = round(lb_final$statistic, 4),
  Ljung_Box_p_value = round(lb_final$p.value, 4),
  AICc = round(final_model$aicc, 3)
)

print(final_summary)
write.csv(final_summary, "final_model_summary.csv", row.names = FALSE)

cat("\nOutputs saved successfully:\n")
cat("1. adf_results.csv\n")
cat("2. accuracy_comparison.csv\n")
cat("3. validation_accuracy_ranked.csv\n")
cat("4. arima_diagnostics_d1_d2.csv\n")
cat("5. forecast_24_months_final.csv\n")
cat("6. final_model_summary.csv\n")
cat("7. Figures saved as PNG files\n")

