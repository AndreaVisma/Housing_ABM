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

data_file = os.path.join(THIS_FOLDER, 'Core_Housing_Model experiment_social_housing-table.csv')

df = pd.read_csv(data_file, skiprows = 6)

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
ax.set(xlim=(-5, 155))
ax.set_xlabel('Simulation Time')
ax.set_title('Percentage unhappy families at each time step')
ax.get_figure().savefig(output_fol + "scatter_base_unhappiness.pdf",
                 bbox_inches='tight')
plt.show()
plt.close()

fig, ax = plt.subplots()

ax = sns.scatterplot(data=df_baseline, x="[step]", y="percent-cannot-afford")
ax.set(xlim=(-5, 155))
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
ax.set(xlim=(-5, 155))
ax.set_xlabel('Simulation Time')
ax.set_title('Percentage unhappy families at each time step')
ax.get_figure().savefig(output_fol + "timeseries_base_unhappiness.pdf",
                 bbox_inches='tight')
plt.show()
plt.close()

fig, ax = plt.subplots()

ax = sns.lineplot(data=df_baseline_avg, x="[step]", y="percent-cannot-afford")
ax.set(xlim=(-5, 155))
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
ax.set(xlim=(-5, 155))
ax.set_xlabel('Simulation Time')
ax.set_title('Percentage unhappy families at each time step')
ax.get_figure().savefig(output_fol + "scatter_SH_unhappiness.pdf",
                 bbox_inches='tight')
plt.show()
plt.close()

fig, ax = plt.subplots()

ax = sns.scatterplot(data=df_social_housing, x="[step]", y="percent-cannot-afford", hue = 'number-socialhousing')
ax.set(xlim=(-5, 155))
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
ax.set(xlim=(-5, 155))
ax.set_xlabel('Simulation Time')
ax.set_title('Percentage unhappy families at each time step')
ax.get_figure().savefig(output_fol + "timeseries_SH_unhappiness.pdf",
                 bbox_inches='tight')
plt.show()
plt.close()

fig, ax = plt.subplots()

ax = sns.scatterplot(data=df_social_housing_avg, x="[step]", y="percent-cannot-afford", size = 'number-socialhousing', hue = 'clustered-social-housing')
ax.set(xlim=(-5, 155))
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


#%%create a timeline of the evolution of unhappiness and people that cannot afford their house
# for the baseline case

fig, ax = plt.subplots()

ax = sns.scatterplot(data=df_baseline, x="[step]", y="percent-unhappy")
ax.set(xlim=(-5, 155))
ax.set_xlabel('Simulation Time')
ax.set_title('Percentage unhappy families at each time step')
ax.get_figure().savefig(output_fol + "scatter_base_unhappiness_after5.pdf",
                 bbox_inches='tight')
plt.show()
plt.close()

fig, ax = plt.subplots()

ax = sns.scatterplot(data=df_baseline, x="[step]", y="percent-cannot-afford")
ax.set(xlim=(-5, 155))
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

fig, ax = plt.subplots()

ax = sns.lineplot(data=df_baseline_avg, x="[step]", y="percent-unhappy")
ax.set(xlim=(-5, 155))
ax.set_xlabel('Simulation Time')
ax.set_title('Percentage unhappy families at each time step')
ax.get_figure().savefig(output_fol + "timeseries_base_unhappiness_after5.pdf",
                 bbox_inches='tight')
plt.show()
plt.close()

fig, ax = plt.subplots()

ax = sns.lineplot(data=df_baseline_avg, x="[step]", y="percent-cannot-afford")
ax.set(xlim=(-5, 155))
ax.set_xlabel('Simulation Time')
ax.set_title('Percentage families that cannot afford their house at each time step')
ax.get_figure().savefig(output_fol + "timeseries_base_afford_after5.pdf",
                 bbox_inches='tight')
plt.show()
plt.close()

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

df_avgs = df.groupby(['[step]', 'social-housing?', 'clustered-social-housing', 'number-socialhousing']).mean()
l1 = df_avgs.index.get_level_values(1)
l2 = df_avgs.index.get_level_values(2)
df_avgs_SH = df_avgs[(l1== True)]
df_avgs_no_SH = df_avgs[(l1== False)]

#%% objective and normative segregation
fig,ax = plt.subplots(1, 2, sharey = True)

sns.lineplot(ax = ax[0], data = df_avgs_no_SH, x = '[step]', y = 'segregation_working_obj',
                        palette = 'viridis', label = 'working class')
sns.lineplot(ax = ax[0], data = df_avgs_no_SH, x = '[step]', y = 'segregation_middle_obj',
                        palette = 'Blues', label = 'middle class')
sns.lineplot(ax = ax[0], data = df_avgs_no_SH, x = '[step]', y = 'segregation_upper_obj',
                        palette = 'rocket', label ='upper class')

sns.lineplot(ax = ax[1], data = df_avgs_SH, x = '[step]', y = 'segregation_working_obj',
                        palette = 'viridis', label = 'working class')
sns.lineplot(ax = ax[1], data = df_avgs_SH, x = '[step]', y = 'segregation_middle_obj',
                        palette = 'Blues', label = 'middle class')
sns.lineplot(ax = ax[1], data = df_avgs_SH, x = '[step]', y = 'segregation_upper_obj',
                        palette = 'rocket', label ='upper class')

ax[0].set_title('No social housing')
ax[0].set_ylabel('Average number neighbors of the same class')
ax[0].set_xlabel('Simulation time')
ax[1].set_xlabel('Simulation time')
ax[1].set_title('With social housing')
    
fig.get_figure().savefig(output_fol + "objective_segregation_comparison1.pdf",
                 bbox_inches='tight')

plt.show()
plt.close()

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
ax[0].set_ylabel('Average number neighbors of the same class')
ax[0].set_xlabel('Simulation time')
ax[1].set_xlabel('Simulation time')
ax[1].set_title('With social housing')
    
fig.get_figure().savefig(output_fol + "normative_segregation.pdf",
                 bbox_inches='tight')

plt.show()
plt.close()

#%%

fig,ax = plt.subplots(1, 2, sharey = True)

sns.lineplot(ax = ax[0], data = df_avgs_no_SH, x = '[step]', y = 'working-avg-distance-services',
                        palette = 'viridis', label = 'working class')
sns.lineplot(ax = ax[0], data = df_avgs_no_SH, x = '[step]', y = 'middle-avg-distance-services',
                        palette = 'Blues', label = 'middle class')
sns.lineplot(ax = ax[0], data = df_avgs_no_SH, x = '[step]', y = 'upper-avg-distance-services',
                        palette = 'rocket', label ='upper class')

sns.lineplot(ax = ax[1], data = df_avgs_SH, x = '[step]', y = 'working-avg-distance-services',
                        palette = 'viridis', label = 'working class')
sns.lineplot(ax = ax[1], data = df_avgs_SH, x = '[step]', y = 'middle-avg-distance-services',
                        palette = 'Blues', label = 'middle class')
sns.lineplot(ax = ax[1], data = df_avgs_SH, x = '[step]', y = 'upper-avg-distance-services',
                        palette = 'rocket', label ='upper class')

ax[0].set_title('No social housing')
ax[0].set_ylabel('Average number neighbors of the same class')
ax[0].set_xlabel('Simulation time')
ax[1].set_xlabel('Simulation time')
ax[1].set_title('With social housing')
    
fig.get_figure().savefig(output_fol + "normative_segregation_distance_services.pdf",
                 bbox_inches='tight')

plt.show()
plt.close()

fig,ax = plt.subplots(1, 2, sharey = True)

sns.lineplot(ax = ax[0], data = df_avgs_no_SH, x = '[step]', y = 'working-avg-distance-green',
                        palette = 'viridis', label = 'working class')
sns.lineplot(ax = ax[0], data = df_avgs_no_SH, x = '[step]', y = 'middle-avg-distance-green',
                        palette = 'Blues', label = 'middle class')
sns.lineplot(ax = ax[0], data = df_avgs_no_SH, x = '[step]', y = 'upper-avg-distance-green',
                        palette = 'rocket', label ='upper class')

sns.lineplot(ax = ax[1], data = df_avgs_SH, x = '[step]', y = 'working-avg-distance-green',
                        palette = 'viridis', label = 'working class')
sns.lineplot(ax = ax[1], data = df_avgs_SH, x = '[step]', y = 'middle-avg-distance-green',
                        palette = 'Blues', label = 'middle class')
sns.lineplot(ax = ax[1], data = df_avgs_SH, x = '[step]', y = 'upper-avg-distance-green',
                        palette = 'rocket', label ='upper class')

ax[0].set_title('No social housing')
ax[0].set_ylabel('Average number neighbors of the same class')
ax[0].set_xlabel('Simulation time')
ax[1].set_xlabel('Simulation time')
ax[1].set_title('With social housing')
    
fig.get_figure().savefig(output_fol + "normative_segregation_distance_green.pdf",
                 bbox_inches='tight')

plt.show()
plt.close()








