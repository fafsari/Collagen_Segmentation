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
install.packages("openxlsx")
library(openxlsx)
library(writexl)
library(readxl)
library(dplyr)
library(tidyr)
library(ggplot2)
library(ggcorrplot)
library(minpack.lm)
library(lubridate)

excel_file <- "Z:\\f.afsari\\4-DUET\\database\\Tx_data\\Excel_files\\Original files\\TimeSeriesData-deID_fatemeh_date.xlsx"

mode <- "dead"  # "donors" or "dead"

# excel_file <- paste0(data_path, mode, ".xlsx")

# Read the Excel file
df <- read_excel(excel_file, col_types = "text", sheet = mode)
head(df)
print(dim(df))

df <- df %>%
    select(-'Bx Reason', -'Time Post-Tx')

# Transform the data into long format
df_long <- df %>%
  pivot_longer(
    cols = contains("SCr"),   # Selects all columns that start with "eGFR_"
    names_to = "Time",             # The new column where the names of the selected columns will be stored
    values_to = "SCr"             # The new column where the values will be stored
  ) %>%
  select(Patient_ID, 'SP#', Age, Gender, Time, SCr)


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
output_path <- paste0("Z:\\f.afsari\\4-DUET\\database\\Tx_data\\Excel_files\\Obtained files\\SCr_long_data_1.xlsx")

# Create a new workbook
wb <- createWorkbook()

# Add a sheet and write the data
addWorksheet(wb, mode)
writeData(wb, mode, df_long)

# Save the workbook
saveWorkbook(wb, output_path, overwrite = TRUE)

# Save the transformed data to a excel file
# write.csv(df_long, output_path)
