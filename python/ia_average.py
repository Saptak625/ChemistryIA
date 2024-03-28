from pymeasurement import Measurement as M
import pandas as pd
import scipy
import numpy as np
import matplotlib.pyplot as plt

# Read the standardization curve from the IA.xlsx file.
# D5 to I10
cal_df = pd.read_excel('IA.xlsx', sheet_name='Standardization')
cal_df = cal_df.iloc[3:9, 3:9]

# Make the first row the column names.
cal_df.columns = cal_df.iloc[0]
cal_df = cal_df.iloc[1:]

# Remove the index column.
cal_df = cal_df.reset_index(drop=True)

# Cast the data to floats.
cal_df = cal_df.astype(float)

# Calculate the absorbance.
cal_df['Transmittance (± 0.00035)']=M.importColumn(cal_df['Transmittance (± 0.035%)']/100, uncertainty=0.00035, decimals=4)
cal_df['Absorbance'] = cal_df[f'Transmittance (± 0.00035)'].apply(lambda x: M.apply_func('log(1/x, 10)', x=x))

print(cal_df)

# Make a linear curve fit of the concentration and absorbance data.
line = scipy.stats.linregress(cal_df['Concentration (M)'].to_numpy(), [float(i.sample.value) for i in cal_df['Absorbance']])
print(f"y = {line.slope}x + {line.intercept}")
print(f"r^2 = {line.rvalue**2}")

beer_slope = M.fromStr(f'{line.slope} +/- 13.15 L/mol')
beer_intercept = M.fromStr(f'{line.intercept} +/- 0.003634')

# Plot the data and the linear fit.
plt.figure()
plt.errorbar(cal_df['Concentration (M)'], [float(i.sample.value) for i in cal_df['Absorbance']], fmt='.', yerr=[float(i.uncertainty.value) for i in cal_df['Absorbance']], label='original data')
plt.plot(cal_df['Concentration (M)'], line.intercept + line.slope * cal_df['Concentration (M)'], 'r--', label='fitted line')
plt.legend(loc='upper left')
plt.title('Standardization Curve: Absorbance vs. Concentration')
plt.xlabel('Concentration (M)')
plt.ylabel('Absorbance')
plt.grid()

# Read the transmittance data from the IA.xlsx file.
# E8 to S17
df = pd.read_excel('IA.xlsx', sheet_name='Main Experiment')
df = df.iloc[6:16, 4:19]

# Cast the data to floats.
df = df.astype(float)

# Name the columns of the data frame.
df.columns = ['Temperature (C)'] + [f'Transmittance % {i}' for i in range(1, 15)]

# Remove the index column.
df = df.reset_index(drop=True)

# Absorbance = log_10 (100 / Transmittance)
for i in range(1, 15):
    df[f'Transmittance % {i}'] = M.importColumn(df[f'Transmittance % {i}'], uncertainty=0.035, decimals=2)

# Take the average of the transmittance data (trial by trial).
df['Average Transmittance %'] = [M.average([df[f'Transmittance % {i}'][j] for i in range(1, 15)]) for j in range(10)]
df['Average Absorbance'] = df['Average Transmittance %'].apply(lambda x: M.apply_func('log(100/x, 10)', x=x))

# Calculate the concentration of the unknowns.
# Conversions
mL_to_L = M.fromStr('0.001c L/mL')

# Initial concentrations.
fe_conc = M.fromStr('0.025c mol/L')
fe_volume = M.fromStr('3.00a mL')
fe_moles_init = fe_conc * fe_volume * mL_to_L
print(f'Fe initial moles: {fe_moles_init}')

scn_conc = M.fromStr('0.025c mol/L')
scn_volume = M.fromStr('3.00a mL')
scn_moles_init = scn_conc * scn_volume * mL_to_L
print(f'SCN initial moles: {scn_moles_init}')

total_volume = M.fromStr('45.0 +/- 0.2 mL')

# Calculate the Natural Log of the Equilibrium Constant.
ln_equilibrium_constant = []
for absorbance in df['Average Absorbance']:
    fescn_conc = (absorbance - beer_intercept) / beer_slope
    fescn_moles = fescn_conc * total_volume * mL_to_L
    fe_moles_eq = fe_moles_init - fescn_moles
    scn_moles_eq = scn_moles_init - fescn_moles
    equilibrium_constant = (fescn_moles / total_volume / mL_to_L) / ((fe_moles_eq / total_volume / mL_to_L)*(scn_moles_eq / total_volume / mL_to_L))
    ln_equilibrium_constant.append(M.apply_func('log(x)', x=equilibrium_constant))
df[f'Log Average Equilibrium Constant'] = ln_equilibrium_constant

# Make Temperature into Measurements
df['Reciprocal Temperature (1/K)'] = [(1/M.fromStr(str(round(i+273, 1))+' +/- 1 K')).absolute() for i in df['Temperature (C)']]

# Final Data Frame
print(df)

# Van't Hoff Plot
average_x_data = df['Reciprocal Temperature (1/K)']
average_y_data = df['Log Average Equilibrium Constant']
print(average_x_data)
print([float(i.uncertainty.value) for i in average_x_data])
print(average_y_data)
print([float(i.uncertainty.value) for i in average_y_data])
plt.figure()
plt.errorbar([float(i.sample.value) for i in average_x_data], [float(i.sample.value) for i in average_y_data], fmt='.', xerr=[float(i.uncertainty.value) for i in average_x_data], yerr=[float(i.uncertainty.value) for i in average_y_data])
line_2 = scipy.stats.linregress([float(i.sample.value) for i in average_x_data], [float(i.sample.value) for i in average_y_data])
print(f"y = {line_2.slope}x + {line_2.intercept}")
print(f"r^2 = {line_2.rvalue**2}")
plt.plot([float(i.sample.value) for i in average_x_data], line_2.slope * np.array([float(i.sample.value) for i in average_x_data]) + line_2.intercept, 'r--')
plt.grid()
plt.xlabel('Reciprocal Temperature (1/K)')
plt.ylabel('Natural Log of Equilibrium Constant')
plt.title("Van't Hoff Plot")
plt.show()

# Convert the Measurements to exportColumn
for column in df.columns:
    if 'object' in str(df[column].dtype):
        M.exportColumn(df, df[column])

# Save the data frame to an excel file.
df.to_excel('IA_output_average.xlsx')