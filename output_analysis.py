# -*- coding: utf-8 -*-
"""
Created on Sun Jan 16 12:08:58 2022

@author: Andrea Vismara
"""
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import pandas as pd
import numpy as np
import seaborn as sns
import random
import os

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
print(THIS_FOLDER)

output_fol = os.path.join(THIS_FOLDER, "Segregation_plots/" )

try:
    os.mkdir(output_fol)
except OSError:
    print ("Creation of the directory %s failed" % output_fol)
    #%%

sns.set("notebook")
sns.set_style("darkgrid")

data_file = os.path.join(THIS_FOLDER, 'Core_Housing_Model poster_output-table.csv')
baseline_file = os.path.join(THIS_FOLDER, 'Core_Housing_Model baseline-table.csv')

df_1 = pd.read_csv(data_file, skiprows = 6)
df_2 = pd.read_csv(baseline_file, skiprows = 6)

df = pd.concat([df_2, df_1])

df.rename(columns = {'segregation_working_obj ;;average segregation measure for the working class based on distribution of turtles of the same class in neighboring patches' : 'segregation_working_obj',
'segregation_middle_obj ;;average segregation measure for the middle class based on distribution of turtles of the same class in neighboring patches': 'segregation_middle_obj',
'segregation_upper_obj ;;average segregation measure for the upper class based on distribution of turtles of the same class in neighboring patches': 'segregation_upper_obj',
'segregation_working_norm ;;average segregation measure for the working class based on distance to greeneries and services' : 'segregation_working_norm',
'segregation_middle_norm ;;average segregation measure for the middle class based on distance to greeneries and services' : 'segregation_middle_norm',
'segregation_upper_norm ;;average segregation measure for the upper class based on distance to greeneries and services' : 'segregation_upper_norm'}, inplace = True)

df = df.sort_values( by = '[run number]', axis = 0).reset_index(drop = True)

print( df.head(10) )

#%%

standard_columns = ['[run number]', 'reliance-on-network', 'number',
       'income-socially-influenced?', 'fixed-distr-social-housing',
       'hide-links?', 'clustered-social-housing', 'minimum-score-upper',
       'social-housing?', 'minimum-score-working',
       'different-district-colours?', 'number-socialhousing',
       'minimum-score-middle', '[step]']


#%% Baseline Scenario Plots

df_baseline = df[df['social-housing?'] == False]

df_baseline_avg = df_baseline.groupby('[step]').mean()

df_social_housing = df[df['social-housing?'] == True]

df_social_housing_avg = df_social_housing.groupby(['[step]', 'clustered-social-housing', 'number-socialhousing']).mean()

#%%create a timeline of the evolution of unhappiness and people that cannot afford their house
# for the baseline case

fig, ax = plt.subplots()

ax = sns.scatterplot(data=df_baseline, x="[step]", y="percent-unhappy")
ax.set(xlim=(-5, 105))
ax.set_xlabel('Simulation Time')
ax.set_title('Percentage unhappy families at each time step')
ax.get_figure().savefig(output_fol + "scatter_base_unhappiness.pdf",
                 bbox_inches='tight')
plt.show()
plt.close()

fig, ax = plt.subplots()

ax = sns.scatterplot(data=df_baseline, x="[step]", y="percent-cannot-afford")
ax.set(xlim=(-5, 105))
ax.set_xlabel('Simulation Time')
ax.set_title('Percentage families that cannot afford their house at each time step')
ax.get_figure().savefig(output_fol + "scatter_base_afford.pdf",
                 bbox_inches='tight')
plt.show()
plt.close()

# correlation between unhappiness and cannot afford

fig, ax = plt.subplots()

ax = sns.regplot(data=df_baseline, x="percent-cannot-afford", y="percent-unhappy", 
                 order=2, ci=90)
#ax.set(xlim=(-5, 155))
ax.set_title('Correlation between not being able to afford dwelling and unhappiness')
ax.get_figure().savefig(output_fol + "scatter_base_all_correlation.pdf",
                 bbox_inches='tight')
plt.show()
plt.close()

fig, ax = plt.subplots()

ax = sns.lineplot(data=df_baseline_avg, x="[step]", y="percent-unhappy")
ax.set(xlim=(-5, 105))
ax.set_xlabel('Simulation Time')
ax.set_title('Percentage unhappy families at each time step')
ax.get_figure().savefig(output_fol + "timeseries_base_unhappiness.pdf",
                 bbox_inches='tight')
plt.show()
plt.close()

fig, ax = plt.subplots()

ax = sns.lineplot(data=df_baseline_avg, x="[step]", y="percent-cannot-afford")
ax.set(xlim=(-5, 105))
ax.set_xlabel('Simulation Time')
ax.set_title('Percentage families that cannot afford their house at each time step')
ax.get_figure().savefig(output_fol + "timeseries_base_afford.pdf",
                 bbox_inches='tight')
plt.show()
plt.close()

# correlation between unhappiness and cannot afford

fig, ax = plt.subplots()

ax = sns.regplot(data=df_baseline_avg, x="percent-cannot-afford", y="percent-unhappy", 
                 order=1, ci=90)
#ax.set(xlim=(-5, 155))
ax.set_title('Correlation between not being able to afford dwelling and unhappiness')
ax.get_figure().savefig(output_fol + "scatter_base_correlation.pdf",
                 bbox_inches='tight')
plt.show()
plt.close()

#%%create a timeline of the evolution of unhappiness and people that cannot afford their house
# for the cases with social housing

fig, ax = plt.subplots()

ax = sns.scatterplot(data=df_social_housing, x="[step]", y="percent-unhappy", hue = 'number-socialhousing')
ax.set(xlim=(-5, 105))
ax.set_xlabel('Simulation Time')
ax.set_title('Percentage unhappy families at each time step')
ax.get_figure().savefig(output_fol + "scatter_SH_unhappiness.pdf",
                 bbox_inches='tight')
plt.show()
plt.close()

fig, ax = plt.subplots()

ax = sns.scatterplot(data=df_social_housing, x="[step]", y="percent-cannot-afford", hue = 'number-socialhousing')
ax.set(xlim=(-5, 105))
ax.set_xlabel('Simulation Time')
ax.set_title('Percentage families that cannot afford their house at each time step')
ax.get_figure().savefig(output_fol + "scatter_SH_afford.pdf",
                 bbox_inches='tight')
plt.show()
plt.close()

# correlation between unhappiness and cannot afford

fig, ax = plt.subplots()

ax = sns.regplot(data=df_social_housing, x="percent-cannot-afford", y="percent-unhappy", 
                 order=2, ci=90)
#ax.set(xlim=(-5, 155))
ax.set_title('Correlation between not being able to afford dwelling and unhappiness')
ax.get_figure().savefig(output_fol + "scatter_SH_all_correlation.pdf",
                 bbox_inches='tight')
plt.show()
plt.close()

fig, ax = plt.subplots()

ax = sns.scatterplot(data=df_social_housing_avg, x="[step]", y="percent-unhappy", size = 'number-socialhousing', hue = 'clustered-social-housing')
ax.set(xlim=(-5, 105))
ax.set_xlabel('Simulation Time')
ax.set_title('Percentage unhappy families at each time step')
ax.get_figure().savefig(output_fol + "timeseries_SH_unhappiness.pdf",
                 bbox_inches='tight')
plt.show()
plt.close()

fig, ax = plt.subplots()

ax = sns.scatterplot(data=df_social_housing_avg, x="[step]", y="percent-cannot-afford", size = 'number-socialhousing', hue = 'clustered-social-housing')
ax.set(xlim=(-5, 105))
ax.set_xlabel('Simulation Time')
ax.set_title('Percentage families that cannot afford their house at each time step')
ax.get_figure().savefig(output_fol + "timeseries_SH_afford.pdf",
                 bbox_inches='tight')
plt.show()
plt.close()

# correlation between unhappiness and cannot afford

fig, ax = plt.subplots()

ax = sns.regplot(data=df_social_housing_avg, x="percent-cannot-afford", y="percent-unhappy", 
                 order=1, ci=90)
#ax.set(xlim=(-5, 155))
ax.set_title('Correlation between not being able to afford dwelling and unhappiness')
ax.get_figure().savefig(output_fol + "scatter_SH_correlation.pdf",
                 bbox_inches='tight')
plt.show()
plt.close()

#%%

df_baseline = df[df['social-housing?'] == False]
df_baseline = df_baseline[df_baseline['[step]'] > 4]

df_baseline_avg = df_baseline.groupby('[step]').mean()

df_social_housing = df[df['social-housing?'] == True]
df_social_housing = df_social_housing[df_social_housing['[step]'] > 4]

df_social_housing_avg = df_social_housing.groupby(['[step]', 'clustered-social-housing', 'number-socialhousing']).mean()

df = df[df['[step]'] > 4]
df_avgs = df.groupby(['[step]', 'social-housing?', 'clustered-social-housing', 'number-socialhousing']).mean()
#%%create a timeline of the evolution of unhappiness and people that cannot afford their house
# for the baseline case

fig, ax = plt.subplots()

ax = sns.scatterplot(data=df_baseline, x="[step]", y="percent-unhappy")
ax.set(xlim=(-5, 105))
ax.set_xlabel('Simulation Time')
ax.set_title('Percentage unhappy families at each time step')
ax.get_figure().savefig(output_fol + "scatter_base_unhappiness_after5.pdf",
                 bbox_inches='tight')
plt.show()
plt.close()

fig, ax = plt.subplots()

ax = sns.scatterplot(data=df_baseline, x="[step]", y="percent-cannot-afford")
ax.set(xlim=(-5, 105))
ax.set_xlabel('Simulation Time')
ax.set_title('Percentage families that cannot afford their house at each time step')
ax.get_figure().savefig(output_fol + "scatter_base_afford_after5.pdf",
                 bbox_inches='tight')
plt.show()
plt.close()

# correlation between unhappiness and cannot afford

fig, ax = plt.subplots()

ax = sns.regplot(data=df_baseline, x="percent-cannot-afford", y="percent-unhappy", 
                 order=2, ci=90)
#ax.set(xlim=(-5, 155))
ax.set_title('Correlation between not being able to afford dwelling and unhappiness')
ax.get_figure().savefig(output_fol + "scatter_base_all_correlation_after5.pdf",
                 bbox_inches='tight')
plt.show()
plt.close()

#%% Nice lineplots 

fig, ax = plt.subplots()

ax = sns.lineplot(data=df_baseline_avg, x="[step]", y="percent-unhappy", color = 'r', label = '0')
ax = sns.lineplot(data =df_social_housing_avg, x = '[step]', y="percent-unhappy", hue = 'number-socialhousing', palette = 'pastel')
ax.set(xlim=(-5, 105))
ax.set_xlabel('Simulation Time')
ax.set_title('Percentage unhappy families at each time step')
ax.get_figure().savefig(output_fol + "timeseries_unhappiness_after5.pdf",
                 bbox_inches='tight')
plt.show()
plt.close()

fig, ax = plt.subplots()

ax = sns.lineplot(data=df_baseline_avg, x="[step]", y="percent-cannot-afford", color = 'r', label = '0', legend = False)
ax = sns.lineplot(data =df_social_housing_avg, x = '[step]', y="percent-cannot-afford", hue = 'number-socialhousing', palette = 'pastel', legend = False)
ax.set(xlim=(-5, 105))
ax.set_xlabel('Simulation Time')
ax.set_title('Percentage families that cannot afford their house at each time step')
ax.get_figure().savefig(output_fol + "timeseries_afford_after5.pdf",
                 bbox_inches='tight')
plt.show()
plt.close()

#%%
# correlation between unhappiness and cannot afford

fig, ax = plt.subplots()

ax = sns.regplot(data=df_baseline_avg, x="percent-cannot-afford", y="percent-unhappy", 
                 order=1, ci=90)
#ax.set(xlim=(-5, 155))
ax.set_title('Correlation between not being able to afford dwelling and unhappiness')
ax.get_figure().savefig(output_fol + "scatter_base_correlation_after5.pdf",
                 bbox_inches='tight')
plt.show()
plt.close()

#%%create a timeline of the evolution of unhappiness and people that cannot afford their house
# for the cases with social housing

fig, ax = plt.subplots()

ax = sns.scatterplot(data=df_social_housing, x="[step]", y="percent-unhappy", hue = 'number-socialhousing')
ax.set(xlim=(-5, 155))
ax.set_xlabel('Simulation Time')
ax.set_title('Percentage unhappy families at each time step')
ax.get_figure().savefig(output_fol + "scatter_SH_unhappiness_after5.pdf",
                 bbox_inches='tight')
plt.show()
plt.close()

fig, ax = plt.subplots()

ax = sns.scatterplot(data=df_social_housing, x="[step]", y="percent-cannot-afford", hue = 'number-socialhousing')
ax.set(xlim=(-5, 155))
ax.set_xlabel('Simulation Time')
ax.set_title('Percentage families that cannot afford their house at each time step')
ax.get_figure().savefig(output_fol + "scatter_SH_afford_after5.pdf",
                 bbox_inches='tight')
plt.show()
plt.close()

# correlation between unhappiness and cannot afford

fig, ax = plt.subplots()

ax = sns.regplot(data=df_social_housing, x="percent-cannot-afford", y="percent-unhappy", 
                 order=2, ci=90)
#ax.set(xlim=(-5, 155))
ax.set_title('Correlation between not being able to afford dwelling and unhappiness')
ax.get_figure().savefig(output_fol + "scatter_SH_all_correlation_after5.pdf",
                 bbox_inches='tight')
plt.show()
plt.close()

fig, ax = plt.subplots()

ax = sns.scatterplot(data=df_social_housing_avg, x="[step]", y="percent-unhappy", size = 'number-socialhousing', hue = 'clustered-social-housing')
ax.set(xlim=(-5, 155))
ax.set_xlabel('Simulation Time')
ax.set_title('Percentage unhappy families at each time step')
ax.get_figure().savefig(output_fol + "timeseries_SH_unhappiness_after5.pdf",
                 bbox_inches='tight')
plt.show()
plt.close()

fig, ax = plt.subplots()

ax = sns.scatterplot(data=df_social_housing_avg, x="[step]", y="percent-cannot-afford", size = 'number-socialhousing', hue = 'clustered-social-housing')
ax.set(xlim=(-5, 155))
ax.set_xlabel('Simulation Time')
ax.set_title('Percentage families that cannot afford their house at each time step')
ax.get_figure().savefig(output_fol + "timeseries_SH_afford_after5.pdf",
                 bbox_inches='tight')
plt.show()
plt.close()

# correlation between unhappiness and cannot afford

fig, ax = plt.subplots()

ax = sns.regplot(data=df_social_housing_avg, x="percent-cannot-afford", y="percent-unhappy", 
                 order=1, ci=90)
#ax.set(xlim=(-5, 155))
ax.set_title('Correlation between not being able to afford dwelling and unhappiness')
ax.get_figure().savefig(output_fol + "scatter_SH_correlation_after5.pdf",
                 bbox_inches='tight')
plt.show()
plt.close()

#%% Boxplots unhappiness and affordability
df_after10 = df[df['[step]'] > 10]

fig, ax = plt.subplots()

ax = sns.boxplot(data = df_after10, y = 'percent-unhappy', x = 'social-housing?',
                 hue = 'clustered-social-housing')
sns.move_legend(ax, loc="upper left", frameon=True)
ax.set_title('Boxplot unhappiness')
ax.get_figure().savefig(output_fol + "boxplot_unhappiness_after10.pdf",
                 bbox_inches='tight')
plt.show()
plt.close()

fig, ax = plt.subplots()

ax = sns.boxplot(data = df_after10, y = 'percent-cannot-afford', x = 'social-housing?',
                 hue = 'clustered-social-housing')
sns.move_legend(ax, loc="upper right", frameon=True)
ax.set_title('Boxplot households that cannot afford home')
ax.get_figure().savefig(output_fol + "boxplot_affordability_after10.pdf",
                 bbox_inches='tight')

plt.show()
plt.close()

#%% Segregation measures

l1 = df_avgs.index.get_level_values(1)

df_avgs_SH = df_avgs[(l1== True)]
df_avgs_no_SH = df_avgs[(l1== False)]

#%% objective and normative segregation
sns.set_palette(sns.color_palette(['gold', 'limegreen', 'darkred']))

clustered_SH = False

if clustered_SH:
    l2 = df_avgs_SH.index.get_level_values(2)
    df_avgs_SH = df_avgs_SH[(l2 == True)]
else :
    l2 = df_avgs_SH.index.get_level_values(2)
    df_avgs_SH = df_avgs_SH[(l2 == False)]
    
l3 =  df_avgs_SH.index.get_level_values(3)
SH_units_max = max(np.unique(l3))
SH_units_half = round(0.5 * max(np.unique(l3)))
df_avgs_SH_max = df_avgs_SH[(l3== SH_units_max)]
df_avgs_SH_half = df_avgs_SH[(l3== SH_units_half)]

fig,ax = plt.subplots(1, 3, sharey = True, figsize=(10,5))

sns.lineplot(ax = ax[0], data = df_avgs_no_SH, x = '[step]', y = 'segregation_working_obj',
                        label = 'working class', legend = False)
sns.lineplot(ax = ax[0], data = df_avgs_no_SH, x = '[step]', y = 'segregation_middle_obj',
                        label = 'middle class', legend = False)
sns.lineplot(ax = ax[0], data = df_avgs_no_SH, x = '[step]', y = 'segregation_upper_obj',
                        label ='upper class', legend = False)

sns.lineplot(ax = ax[1], data = df_avgs_SH_half, x = '[step]', y = 'segregation_working_obj',
                        label = 'working class', ci = None, legend = False)
sns.lineplot(ax = ax[1], data = df_avgs_SH_half, x = '[step]', y = 'segregation_middle_obj',
                        label = 'middle class', ci = None, legend = False)
sns.lineplot(ax = ax[1], data = df_avgs_SH_half, x = '[step]', y = 'segregation_upper_obj',
                        label ='upper class', ci = None, legend = False)

sns.lineplot(ax = ax[2], data = df_avgs_SH_max, x = '[step]', y = 'segregation_working_obj',
                        label = 'working class', ci = None)
sns.lineplot(ax = ax[2], data = df_avgs_SH_max, x = '[step]', y = 'segregation_middle_obj',
                        label = 'middle class', ci = None)
sns.lineplot(ax = ax[2], data = df_avgs_SH_max, x = '[step]', y = 'segregation_upper_obj',
                        label ='upper class', ci = None)

ax[0].set_title('No social housing')
ax[0].set_ylabel('Average number neighbors of the same class')
ax[0].set_xlabel('Simulation time')
ax[1].set_xlabel('Simulation time')
ax[1].set_title(f'With {SH_units_half} social housing units')
ax[2].set_xlabel('Simulation time')
ax[2].set_title(f'With {SH_units_max} social housing units')
    
fig.get_figure().savefig(output_fol + "objective_segregation_comparison1.pdf",
                 bbox_inches='tight')

plt.show()
plt.close()

#%%
fig,ax = plt.subplots(1, 2, sharey = True)

sns.lineplot(ax = ax[0], data = df_avgs_no_SH, x = '[step]', y = 'segregation_working_norm',
                        palette = 'viridis', label = 'working class')
sns.lineplot(ax = ax[0], data = df_avgs_no_SH, x = '[step]', y = 'segregation_middle_norm',
                        palette = 'Blues', label = 'middle class')
sns.lineplot(ax = ax[0], data = df_avgs_no_SH, x = '[step]', y = 'segregation_upper_norm',
                        palette = 'rocket', label ='upper class')

sns.lineplot(ax = ax[1], data = df_avgs_SH, x = '[step]', y = 'segregation_working_norm',
                        palette = 'viridis', label = 'working class')
sns.lineplot(ax = ax[1], data = df_avgs_SH, x = '[step]', y = 'segregation_middle_norm',
                        palette = 'Blues', label = 'middle class')
sns.lineplot(ax = ax[1], data = df_avgs_SH, x = '[step]', y = 'segregation_upper_norm',
                        palette = 'rocket', label ='upper class')

ax[0].set_title('No social housing')
ax[0].set_ylabel('Average number services and green areas as neighbours')
ax[0].set_xlabel('Simulation time')
ax[1].set_xlabel('Simulation time')
ax[1].set_title('With social housing')
    
fig.get_figure().savefig(output_fol + "normative_segregation.pdf",
                 bbox_inches='tight')

plt.show()
plt.close()

#%%

fig,ax = plt.subplots(1, 3, sharey = True, figsize = (10, 5))

sns.lineplot(ax = ax[0], data = df_avgs_no_SH, x = '[step]', y = 'working-avg-distance-services',
                        label = 'working class', legend = False)
sns.lineplot(ax = ax[0], data = df_avgs_no_SH, x = '[step]', y = 'middle-avg-distance-services',
                        label = 'middle class', legend = False)
sns.lineplot(ax = ax[0], data = df_avgs_no_SH, x = '[step]', y = 'upper-avg-distance-services',
                        label ='upper class', legend = False)

sns.lineplot(ax = ax[1], data = df_avgs_SH_half, x = '[step]', y = 'working-avg-distance-services',
                        ci = None, label = 'working class', legend = False)
sns.lineplot(ax = ax[1], data = df_avgs_SH_half, x = '[step]', y = 'middle-avg-distance-services',
                        ci = None, label = 'middle class', legend = False)
sns.lineplot(ax = ax[1], data = df_avgs_SH_half, x = '[step]', y = 'upper-avg-distance-services',
                        ci = None, label ='upper class', legend = False)

sns.lineplot(ax = ax[2], data = df_avgs_SH_max, x = '[step]', y = 'working-avg-distance-services',
                        ci = None, label = 'working class')
sns.lineplot(ax = ax[2], data = df_avgs_SH_max, x = '[step]', y = 'middle-avg-distance-services',
                        ci = None, label = 'middle class')
sns.lineplot(ax = ax[2], data = df_avgs_SH_max, x = '[step]', y = 'upper-avg-distance-services',
                        ci = None, label ='upper class')

ax[0].set_title('No social housing')
ax[0].set_ylabel('Average distance from services')
ax[0].set_xlabel('Simulation time')
ax[1].set_xlabel('Simulation time')
ax[1].set_title(f'With {SH_units_half} social housing units')
ax[2].set_xlabel('Simulation time')
ax[2].set_title(f'With {SH_units_max} social housing units')
    
fig.get_figure().savefig(output_fol + "normative_segregation_distance_services.pdf",
                 bbox_inches='tight')

plt.show()
plt.close()


fig,ax = plt.subplots(1, 3, sharey = True, figsize = (10, 5))

sns.lineplot(ax = ax[0], data = df_avgs_no_SH, x = '[step]', y = 'working-avg-distance-green',
                        label = 'working class', legend = False)
sns.lineplot(ax = ax[0], data = df_avgs_no_SH, x = '[step]', y = 'middle-avg-distance-green',
                        label = 'middle class', legend = False)
sns.lineplot(ax = ax[0], data = df_avgs_no_SH, x = '[step]', y = 'upper-avg-distance-green',
                        label ='upper class', legend = False)

sns.lineplot(ax = ax[1], data = df_avgs_SH_half, x = '[step]', y = 'working-avg-distance-green',
                        ci = None, label = 'working class', legend = False)
sns.lineplot(ax = ax[1], data = df_avgs_SH_half, x = '[step]', y = 'middle-avg-distance-green',
                        ci = None, label = 'middle class', legend = False)
sns.lineplot(ax = ax[1], data = df_avgs_SH_half, x = '[step]', y = 'upper-avg-distance-green',
                        ci = None, label ='upper class', legend = False)

sns.lineplot(ax = ax[2], data = df_avgs_SH_max, x = '[step]', y = 'working-avg-distance-green',
                        ci = None, label = 'working class')
sns.lineplot(ax = ax[2], data = df_avgs_SH_max, x = '[step]', y = 'middle-avg-distance-green',
                        ci = None, label = 'middle class')
sns.lineplot(ax = ax[2], data = df_avgs_SH_max, x = '[step]', y = 'upper-avg-distance-green',
                        ci = None, label ='upper class')

ax[0].set_title('No social housing')
ax[0].set_ylabel('Average distance from green areas')
ax[0].set_xlabel('Simulation time')
ax[1].set_xlabel('Simulation time')
ax[1].set_title(f'With {SH_units_half} social housing units')
ax[2].set_xlabel('Simulation time')
ax[2].set_title(f'With {SH_units_max} social housing units')
fig.get_figure().savefig(output_fol + "normative_segregation_distance_green.pdf",
                 bbox_inches='tight')

plt.show()
plt.close()

#%% Distribution in dsitricts

def horizontal_barplot(df_here):
    """
    Parameters
    ----------
    df : pandas dataframe from which to extrapolate the data
    """
    
    classes = ['working class', 'middle class', 'upper class']
    
    hh_distr_1 = np.mean(df_here['working-district-1'] + df_here['middle-district-1']
                                    + df_here['upper-district-1'])
    hh_distr_2 = np.mean(df_here['working-district-2'] + df_here['middle-district-2']
                                    + df_here['upper-district-2'])
    hh_distr_3 = np.mean(df_here['working-district-3'] + df_here['middle-district-3']
                                    + df_here['upper-district-3'])
    hh_distr_4 = np.mean(df_here['working-district-4'] + df_here['middle-district-4']
                                    + df_here['upper-district-4'])
    hh_centre = np.mean(df_here['working-centre'] + df_here['middle-centre']
                                    + df_here['upper-centre'])
    hh_periphery = np.mean(df_here['working-periphery'] + df_here['middle-periphery']
                                    + df_here['upper-periphery'])
    
    distribution = {
        'District 1': np.round([np.mean(df_here['working-district-1']) / hh_distr_1, np.mean(df_here['middle-district-1']) / hh_distr_1,
             np.mean(df_here['upper-district-1']) / hh_distr_1], 3),
        'District 2': np.round([ np.mean(df_here['working-district-2']) / hh_distr_2,  np.mean(df_here['middle-district-2']) / hh_distr_2,
              np.mean(df_here['upper-district-2']) / hh_distr_2], 3),
        'District 3': np.round([ np.mean(df_here['working-district-3']) / hh_distr_3,  np.mean(df_here['middle-district-3']) / hh_distr_3,
             np.mean(df_here['upper-district-3']) / hh_distr_3], 3),
        'District 4': np.round([np.mean(df_here['working-district-4']) / hh_distr_4,  np.mean(df_here['middle-district-4']) / hh_distr_4,
              np.mean(df_here['upper-district-4']) / hh_distr_4], 3),
        'Inner City': np.round([ np.mean(df_here['working-centre']) / hh_centre,  np.mean(df_here['middle-centre']) / hh_centre,
              np.mean(df_here['upper-centre']) / hh_centre], 3),
        'Periphery': np.round([ np.mean(df_here['working-periphery']) / hh_periphery,  np.mean(df_here['middle-periphery']) / hh_periphery,
              np.mean(df_here['upper-periphery']) / hh_periphery], 3)
        }
    
    labels = list(distribution.keys())
    data = np.array(list(distribution.values()))
    data_cum = data.cumsum(axis=1)
    category_colors = sns.color_palette(['gold', 'limegreen', 'darkred'])
    
    fig, ax = plt.subplots(figsize=(9.2, 5))
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data, axis=1).max())

    for i, (colname, color) in enumerate(zip(classes, category_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        rects = ax.barh(labels, widths, left=starts, height=0.5,
                        label=colname, color=color)

        # r, g, b, _ = color
        # text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
        ax.bar_label(rects, label_type='center', color= 'black')
    ax.legend(ncol=len(classes), bbox_to_anchor=(0, -0.1),
              loc='lower left', fontsize='small')

    return fig, ax

#%% Plot distribution in districts

df_SH_cluster = df_social_housing[df_social_housing['clustered-social-housing'] == True]
df_SH_No_cluster = df_social_housing[df_social_housing['clustered-social-housing'] == False]

# baseline
fig, ax = horizontal_barplot(df_baseline)
ax.set_title('Geographical segregation, baseline')
ax.set_xlabel('Percentage occupation in each area')

fig.get_figure().savefig(output_fol + "geographical_segregation_bars_baseline.pdf",
                 bbox_inches='tight')

# clustered social housing
values = df_social_housing['number-socialhousing'].unique()

for i in values:
    
    df_chosen = df_SH_cluster[df_SH_cluster['number-socialhousing'] == i]
    
    fig, ax = horizontal_barplot(df_chosen)
    ax.set_title(f'Geographical segregation with clustered social housing, n = {i}')
    ax.set_xlabel('Percentage occupation in each area')

    fig.get_figure().savefig(output_fol + f"geographical_segregation_bars_SH_{i}.pdf",
                     bbox_inches='tight')

# not clustered social housing

for i in values:
    
    df_chosen = df_SH_No_cluster[df_SH_No_cluster['number-socialhousing'] == i]
    
    fig, ax = horizontal_barplot(df_chosen)
    ax.set_title(f'Geographical segregation with non-clustered social housing, n = {i}')
    ax.set_xlabel('Percentage occupation in each area')

    fig.get_figure().savefig(output_fol + f"geographical_segregation_bars_SH_noCluster_{i}.pdf",
                     bbox_inches='tight')




