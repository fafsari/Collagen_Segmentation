### 1- Load the necessary libraries and read the Excel files:
#-------------------------------------------------------------------------------------# # nolint
# Install and load necessary packages
install.packages("readxl")
install.packages("dplyr")
install.packages("tidyr")
install.packages("ggplot2")
install.packages("ggcorrplot")
install.packages("lubridate")
install.packages("writexl")
library(writexl)
library(readxl)
library(dplyr)
library(tidyr)
library(ggplot2)
library(ggcorrplot)
library(minpack.lm)
library(lubridate)

data_path <- "Z:\\f.afsari\\4-DUET\\database\\eGFR Correlation\\Tx_data\\Excel_files\\TimeSeriesData-deID-"

mode <- "donors"  # "donors" or "dead"

excel_file <- paste0(data_path, mode, ".xlsx")

# Read the Excel file
df <- read_excel(excel_file, col_types = "text", sheet = "Sheet1")
head(df)
print(dim(df))

df <- df %>%
    select(-'Stain', -'Bx Reason', -'Time Post-Tx', -'Race')

# Transform the data into long format
df_long <- df %>%
  pivot_longer(
    cols = contains("SCr"),   # Selects all columns that start with "eGFR_"
    names_to = "Time",             # The new column where the names of the selected columns will be stored
    values_to = "SCr"             # The new column where the values will be stored
  ) %>%
  select(Patient_ID, SP, Age, Gender, Time, SCr)


df_long <- df_long %>%
  mutate(
    Days = case_when(
      grepl("3mo", Time) ~ 0.25,
      grepl("6mo", Time) ~ 0.5,
      grepl("1yr", Time) ~ 1.0,
      grepl("2yr", Time) ~ 2.0,
      grepl("3yr", Time) ~ 3.0,
      grepl("4yr", Time) ~ 4.0,
      grepl("5yr", Time) ~ 5.0,
      grepl("6yr", Time) ~ 6.0,
      grepl("7yr", Time) ~ 7.0,
      grepl("Latest", Time) ~ 7.25 # Assuming the latest does not have a specific day
    )
  )

head(df_long)

print(dim(df_long))

# Save the transformed data to a CSV file
output_path <- paste0("Z:\\f.afsari\\4-DUET\\database\\eGFR Correlation\\Tx_data\\Excel_files\\", mode, "\\SCr_long_data.csv")

# Save the transformed data to a excel file
write.csv(df_long, output_path)














# # Convert date columns to date format
# df <- df %>%
#   mutate(across(contains("Date"), ~ as.Date(.x, format = "%m/%d/%Y")))

# # Find the latest date for each patient

# # Calculate the difference in days for each "Latest" column
# df_latest <- df %>%
#    mutate(
#     Latest7 = time_length(difftime(`Latest Date`, `7yr Date`), "years"),
#     Latest6 = time_length(difftime(`Latest Date`, `6yr Date`), "years"),
#     Latest5 = time_length(difftime(`Latest Date`, `5yr Date`), "years"),
#     Latest4 = time_length(difftime(`Latest Date`, `4yr Date`), "years"),
#     Latest3 = time_length(difftime(`Latest Date`, `3yr Date`), "years"),
#     Latest2 = time_length(difftime(`Latest Date`, `2yr Date`), "years"),
#     Latest1 = time_length(difftime(`Latest Date`, `1yr Date`), "years")
#    )
  
#   write.csv(df_latest, "Z:\\f.afsari\\4-DUET\\database\\eGFR Correlation\\Tx_data\\Excel_files\\TimeSeriesData-deID_latest_dead.csv")
#   #  ) %>%
#   #  select(contains("Date"))

# print(df_latest[,11], n = 10)

















# Show rows where both Date and SCr are NA
SCr_long_NA <- SCr_long %>%
  filter(is.na(Days) & is.na(SCr))
head(SCr_long_NA)

# Remove the first row for each patient where Date and SCr are NA
SCr_long <- SCr_long %>%
  filter(!is.na(Days) & !is.na(SCr))
head(SCr_long)

# Convert Date to numeric (days since the first date)
SCr_long <- SCr_long %>%
  group_by(Patient_ID) %>%
  mutate(Days = as.numeric(Date - TX_Date))

# Check the transformed data
head(SCr_long)

# Save the transformed data to a CSV file
write.csv(SCr_long, file = "Z:\\f.afsari\\4-DUET\\database\\SCr Correlation\\Tx_data\\SCr_long_data.csv")


collagen_data <- read_excel(collagen_file)
head(collagen_data)

### 3- Scatter plot for each Slide Number group
#-------------------------------------------------------------------------------------#
# Create a directory to save plots if it doesn't exist
dir.create("Z:\\f.afsari\\4-DUET\\database\\SCr Correlation\\plots", showWarnings = FALSE)

# Scatter plot for each Slide Number group
plots <- lapply(unique(SCr_long$Patient_ID), function(slide_num) {
  data_subset <- SCr_long[SCr_long$Patient_ID == slide_num, ]
  
  p <- ggplot(data_subset, aes(x = Date, y = SCr)) +
    geom_point() +
    labs(title = paste("", slide_num),
         x = "Date",
         y = "SCr")
  
  # Save plot as PNG file
  plot_filename <- paste("Z:\\f.afsari\\4-DUET\\database\\SCr Correlation\\plots\\", slide_num, ".png", sep = "")
  ggsave(plot_filename, plot = p, width = 8, height = 6, units = "in")
  
  return(p)
})

# Print each plot separately
plots

# Scatter plot for all data points
p <- ggplot(SCr_long, aes(x = Date, y = SCr)) +
  geom_point() +
  labs(title = "SCr vs Date",
       x = "Date",
       y = "SCr")

# Save plot as PNG file
plot_filename <- "./Collagen_Segmentation/plots/SCr_vs_Date.png"
ggsave(plot_filename, plot = p, width = 8, height = 6, units = "in")

# Print the plot
print(p)

### 4- Fitting several models to SCr data
#-------------------------------------------------------------------------------------#
#========================================================================#
# 1. Linear Model: SCr = intercept + slope.Days
#========================================================================#
# Fit a linear regression model to each 'Slide Number' group and extract the slope
linear_model <- SCr_long %>%
  group_by(Patient_ID) %>%
  do({
    model <- lm(SCr ~ Days, data = .)
    data.frame(Patient_ID = first(.$Patient_ID),
               beta_0_lin = coef(model)[1], 
               beta_1_lin = coef(model)[2])
  })

# View the slopes for each Patient_ID
print(linear_model, n = Inf)

#========================================================================#
# 2. Polynomial Model: SCr = beta_2_poly.Days^2 + beta_1_poly.Days + beta_0_poly
#========================================================================#
SCr_poly <- SCr_long %>%
  group_by(Patient_ID) %>%
  do({
    model <- try(lm(SCr ~ poly(Days, degree = 2, raw = TRUE), data = .), silent = TRUE)
    if (inherits(model, "try-error")) {
      cat("Error in fitting model for Patient_ID:", unique(.$Patient_ID), "\n")
      print(model)  # Print the error message or model object for diagnostics
      tibble(Patient_ID = unique(.$Patient_ID), beta_0_poly = NA, beta_1_poly = NA, beta_2_poly = NA)
    } else {
      tibble(
        Patient_ID = unique(.$Patient_ID), 
        beta_0_poly = coef(model)[1],  # Intercept
        beta_1_poly = coef(model)[2],  # Coefficient for linear term
        beta_2_poly = coef(model)[3]   # Coefficient for quadratic term
      )
    }
  })

print(SCr_poly, n = Inf)
#========================================================================#
# 3. Exponential Model: SCr = a.exp(b.Days)
#========================================================================#
SCr_exponent <- SCr_long %>%
  group_by(Patient_ID) %>%
  do({
    model <- try(nls(SCr ~ a * exp(b * Days), data = ., start = list(a = 1, b = 0.001)), silent = TRUE)
    if (inherits(model, "try-error")) {
      cat("Error in fitting model for Slide Number:", unique(.$Patient_ID), "\n")
      print(model)  # Print the error message or model object for diagnostics
      tibble(a = NA, b = NA)
    } else {
      tibble(beta_0_exp = coef(model)["a"], beta_1_exp = coef(model)["b"])
    }
  })

SCr_exponent <- SCr_exponent %>%
  select(-a)
SCr_exponent <- SCr_exponent %>%
  select(-b)
print(SCr_exponent, n = Inf)
#========================================================================#
# 4. Piecewise Linear Model:
#========================================================================#
# Calculate median Days for each Patient_ID
breakpoints <- SCr_long %>%
  group_by(Patient_ID) %>%
  summarize(Breakpoint = median(Days, na.rm = TRUE))

# Join breakpoints with SCr_long to add Breakpoint column
SCr_long_breakpts <- SCr_long %>%
  left_join(breakpoints, by = "Patient_ID")

# View the first few rows of SCr_long_breakpts
head(SCr_long_breakpts)
# Fit piecewise linear model for each Slide Number group
SCr_pw_linear <- SCr_long_breakpts %>%
  group_by(Patient_ID) %>%
  do({
    model <- rlm(SCr ~ Days + I(Days > Breakpoint) * (Days - Breakpoint + 1e-10), data = .)
    
    # Extract model coefficients
    data.frame(
      Patient_ID = first(.$Patient_ID),
      intercept_pw_lin = coef(model)[1],
      slope_before_pw_lin = coef(model)[2],
      slope_after_pw_lin = coef(model)[3]
    )
  })
SCr_pw_linear <- SCr_pw_linear %>%
  rename(beta_0_pw_lin = "intercept_pw_lin", beta_1_pw_lin = "slope_before_pw_lin", beta_2_pw_lin = "slope_after_pw_lin")
# View the models for each Slide Number
print(SCr_pw_linear, n = Inf)

#========================================================================#
# 5. Logistic Growth Model: SCr_i(Days) = Asym \cdot \left( \frac{1}{1 + \exp \left( \frac{xmid - Days}{scal}\right)} \right)
#========================================================================#
# Visualize the data to check for outliers
ggplot(SCr_long, aes(x = Days, y = SCr)) +
  geom_point() +
  theme_minimal()

# Fit the model using nlsLM
# install.packages("minpack.lm")

# Scale the Days column (Normalize x = (x-m)/s)
SCr_long_scaled <- SCr_long %>%
  mutate(Days_scaled = scale(Days))

# View the first few rows
head(SCr_long_scaled)

# Define a function to fit the model for each patient with debugging
fit_logistic_model <- function(data) {
  # Calculate starting values based on data characteristics
  start_vals <- list(Asym = max(data$SCr, na.rm = TRUE), 
                     xmid = quantile(data$Days_scaled, 0.5, na.rm = TRUE), 
                     scal = 1)

  # print(paste("Patient ID:", unique(data$Patient_ID)))
  # print(paste("Starting values: Asym =", start_vals$Asym, "xmid =", start_vals$xmid, "scal =", start_vals$scal))
  # print("Data subset:")
  # print(head(data))
  
  # Fit the logistic model using scaled Days and original SCr values
  tryCatch({
    model <- nlsLM(SCr ~ SSlogis(Days_scaled, Asym, xmid, scal), 
                   data = data, 
                   start = start_vals, 
                   control = nls.lm.control(maxiter = 10000))
    # Extract model coefficients
    coef(model)
  }, error = function(e) {
    print(paste("Error fitting model for Patient ID:", unique(data$Patient_ID)))
    print(e)
    # Return NA if the model fails to converge
    return(c(Asym = NA, xmid = NA, scal = NA))
  })
}

# Apply the function to each Patient_ID group
SCr_logistic <- SCr_long_scaled %>%
  group_by(Patient_ID) %>%
  do({
    coefs <- fit_logistic_model(.)
    data.frame(Patient_ID = first(.$Patient_ID),
               Asym = coefs["Asym"],
               xmid = coefs["xmid"],
               scal = coefs["scal"])
  })

# Print all rows of the resulting data frame temporarily
print(SCr_logistic, n = Inf)

#========================================================================#
#  6. Power Model: SCr ~ \beta_0 * Days^\beta_1
#========================================================================#
SCr_power <- SCr_long %>%
  group_by(Patient_ID) %>%
  do({
    model <- try(nls(SCr ~ beta_0 * Days^beta_1, data = SCr_long,
                   start = list(beta_0 = 1, beta_1 = 0.001)), silent = TRUE)
    if (inherits(model, "try-error")) {
      cat("Error in fitting model for Slide Number:", unique(.$Patient_ID), "\n")
      print(model)  # Print the error message or model object for diagnostics
      tibble(beta_0 = NA, beta_1 = NA)
    } else {
      tibble(beta_0_pow = coef(model)["beta_0"], beta_1_pow = coef(model)["beta_1"])
    }
  })

print(SCr_power, n = Inf)

# Extract coefficients from SCr_power
coefficients <- SCr_power %>%
  mutate(Patient_ID = as.character(Patient_ID))  # Ensure Patient_ID is character for plotting

# Function to plot power law curves
plot_power_curves <- function(coefficients, data, plot_path = "./Collagen_Segmentation/plots/") {
  plots <- lapply(1:nrow(coefficients), function(i) {
    patient_id <- coefficients$Patient_ID[i]
    a <- coefficients$a[i]
    b <- coefficients$b[i]
    
    # Filter data for the current patient
    patient_data <- filter(data, Patient_ID == patient_id)
    
    # Create plot
    p <- ggplot(patient_data, aes(x = Days, y = SCr)) +
      geom_point() +
      geom_line(aes(y = a * Days^b), color = "blue") +  # Plot the power law curve
      labs(title = paste("Patient ID:", patient_id),
           x = "Days",
           y = "SCr")
    
    # Save plot as PNG file
    plot_filename <- paste(plot_path, patient_id, "_power_law.png", sep = "")
    ggsave(plot_filename, plot = p, width = 8, height = 6, units = "in")
    
    return(p)
  })
  
  return(plots)
}

# Plot power law curves for each patient
power_plots <- plot_power_curves(coefficients, SCr_long, plot_path = "./Collagen_Segmentation/plots/")

# Print each plot separately (optional)
print(power_plots)

#========================================================================#
# 7. Simple SCr average
#========================================================================#
# Calculate average SCr for each 'Slide Number'
SCr_avg <- SCr_long %>%
  group_by(Patient_ID) %>%
  summarize(average_SCr = mean(SCr, na.rm = TRUE))

print(SCr_avg, n = Inf)

# Load required libraries
library(dplyr)

#========================================================================#
# 8. Average the fitted linear models using two consecutive points
#========================================================================#
# Create a function to fit a linear model using two consecutive points and return slope and intercept
fit_consecutive_lm <- function(data) {
  results <- data.frame()
  
  for (i in 1:(nrow(data) - 1)) {
    # Subset data for two consecutive points
    subset_data <- data[i:(i+1), ]
    
    # Fit a linear model
    model <- lm(SCr ~ Days, data = subset_data)
    
    # Extract coefficients
    intercept <- coef(model)[1]
    slope <- coef(model)[2]
    
    # Store results
    results <- rbind(results, data.frame(Patient_ID = unique(data$Patient_ID), intercept = intercept, slope = slope))
  }
  
  return(results)
}

# Apply the function to each Patient_ID and calculate average slope and intercept
SCr_avg_consecutive <- SCr_long %>%
  group_by(Patient_ID) %>%
  do(fit_consecutive_lm(.)) %>%
  group_by(Patient_ID) %>%
  summarize(beta_0_lin_avg = mean(intercept, na.rm = TRUE),
            beta_1_lin_avg = mean(slope, na.rm = TRUE))

# View the results
print(SCr_avg_consecutive, n = Inf)


### 4- Filter the collagen data:
#-------------------------------------------------------------------------------------#
# Filter collagen data to include only the Slide Numbers present in the cleaned SCr data
collagen_data <- collagen_data %>%
  rename(Patient_ID = `Slide Number`)

collagen_data_filtered <- collagen_data %>%
  filter(Patient_ID %in% SCr_long$Patient_ID)

# Check the cleaned collagen data
head(collagen_data_filtered)

### 5- Merge the average SCr scores with the collagen data and compute the correlation matrix:
#-------------------------------------------------------------------------------------#
# Merge the slope of SCr scores with collagen data
merged_data <- merge(linear_model, collagen_data_filtered, by = "Patient_ID")
merged_data <- merge(SCr_poly, merged_data, by = "Patient_ID")
merged_data <- merge(SCr_exponent, merged_data, by = "Patient_ID")
merged_data <- merge(SCr_pw_linear, merged_data, by = "Patient_ID")
merged_data <- merge(SCr_logistic, merged_data, by = "Patient_ID")
merged_data <- merge(SCr_power, merged_data, by = "Patient_ID")
merged_data <- merge(SCr_avg, merged_data, by = "Patient_ID")
merged_data <- merge(SCr_avg_consecutive, merged_data, by = "Patient_ID")

# Remove the Patient_ID column for correlation computation
correlation_data <- merged_data %>%
  select(-Patient_ID)

# # Compute the correlation matrix
# cor_matrix <- cor(correlation_data, use = "complete.obs")

# Impute missing values (e.g., using mean)
correlation_data_imputed <- correlation_data %>%
  mutate_all(funs(replace(., is.na(.), mean(., na.rm = TRUE))))

# Compute the correlation matrix
cor_matrix <- cor(correlation_data_imputed)

# Display the correlation matrix
print(cor_matrix)

# Specify rows and columns to save
rows_to_save <- c(
  "beta_0_lin", "beta_1_lin",
  "beta_0_poly", "beta_1_poly", "beta_2_poly",
  "beta_0_exp", "beta_1_exp",
  "beta_0_pw_lin", "beta_1_pw_lin", "beta_2_pw_lin",
  "Asym", "xmid", "scal",
  "beta_0_pow", "beta_1_pow",
  "average_SCr",
  "beta_0_lin_avg", "beta_1_lin_avg"
)

# Select columns to save (exclude rows_to_save)
cols_to_save <- setdiff(colnames(cor_matrix), rows_to_save)

# Subset the correlation matrix
cor_matrix_subset <- cor_matrix[rows_to_save, cols_to_save]

# Save the subsetted correlation matrix to a CSV file
write.csv(cor_matrix, "./Collagen_Segmentation/plots/correlation_matrix.csv")
write.csv(cor_matrix_subset, "./Collagen_Segmentation/plots/correlation_matrix_subset.csv")

### 4- Plot the correlation matrix:
#-------------------------------------------------------------------------------------#
# Plot the correlation matrix
corr_plot <- ggcorrplot(cor_matrix, 
                        method = "square", 
                        type = "full",
                        lab = TRUE, 
                        lab_size = 1.5, 
                        colors = c("blue", "white", "red"), 
                        title = "Correlation Matrix",
                        ggtheme = theme_minimal())

# Save the plot
ggsave("./Collagen_Segmentation/plots/correlation_matrix_plot.png", plot = corr_plot, width = 10, height = 8)

# Plot the subset of the correlation matrix
# corr_plot <- ggcorrplot(cor_matrix_subset, 
#                         method = "square", 
#                         type = "full",
#                         lab = TRUE, 
#                         lab_size = 1.75, 
#                         colors = c("blue", "white", "red"), 
#                         title = "Correlation Matrix",
#                         ggtheme = theme_minimal() +
#                           theme(axis.text.x = element_text(size = 6),  # Adjust x-axis text size
#                                 axis.text.y = element_text(size = 10),  # Adjust y-axis text size
#                                 plot.title = element_text(size = 14),   # Adjust plot title size
#                           )
# )
# # Modify the square size directly in the ggplot object
# corr_plot <- corr_plot + 
#   theme(
#     rect = element_rect(size = 10)  # Adjust square size here
#   )

# corr_plot <- ggcorrplot(cor_matrix_subset,
#                         sig.level = 0.05,
#                         lab_size = 4.5, p.mat = NULL, 
#                         insig = c("pch", "blank"), pch = 1, pch.col = "black", pch.cex =1, tl.cex = 14) +
#   theme(axis.text.x = element_text(margin=margin(-2,0,0,0)),  # Order: top, right, bottom, left
#         axis.text.y = element_text(margin=margin(0,-2,0,0))) +
#   geom_vline(xintercept=1:ncol(mtcars)-0.5, colour="white", size=2) +
#   geom_hline(yintercept=1:ncol(mtcars)-0.5, colour="white", size=2) 
# install.packages("reshape2")
library(reshape2)   
# Plot with correlation values displayed in tiles
corr_plot <- ggplot(melt(cor_matrix_subset), aes(Var1, Var2, fill=value, label = round(value, 2))) +
  geom_tile(color = "white") +  # Add tiles with white borders
  geom_text(color = "black", size = 1.75, vjust = 1) +  # Add text labels for correlation values
  scale_fill_gradient2(low = "blue", mid = "white", high = "red") +  # Color gradient
  theme_minimal() +
  coord_equal() +
  labs(x = "", y = "", fill = "Correlation") +
  theme(
    axis.text.x = element_text(size = 10, angle = 90, vjust = 1, hjust = 1, margin = margin(-3, 0, 0, 0)),
    axis.text.y = element_text(size = 10, margin = margin(0, -3, 0, 0)),
    panel.grid.major = element_blank()
  )

# Save the plot
ggsave("./Collagen_Segmentation/plots/correlation_matrix_subset_plot.png", plot = corr_plot, width = 10, height = 8)