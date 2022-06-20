#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 12:08:10 2022

@author: Timo Sommer

Plots a lineplot with custom specified data.
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme()


plot_data = {
                'goodness': [1, 2, 3, 4, 5],
                'precision': [0.9, 0.9, np.nan, 0.96, 1.0],
                'sensitivity': [1, 0.5, 0, 0.5, 0.5],
                'NPV': [np.nan, 0.1, 0.1, 0.15, 0.18],
                'specificity': [0., 0.5, 1, 0.8, 1]
    }

save_file = '/home/timo/Masterarbeit/Images/assessment_of_extrapolation/subjective_assessment.png'


x = 'goodness'
y = ['precision', 'sensitivity', 'NPV', 'specificity']

ylabel = 'value'
huelabel = 'metric'



# Plot data

df = pd.DataFrame(plot_data)
df_wide = pd.melt(df, [x]).rename(columns={'value': ylabel, 'variable': huelabel})

sns.lineplot(data=df_wide, x=x, y=ylabel, hue=huelabel)

plt.xticks([1, 2, 3, 4, 5], ['very bad', 'bad', 'ok', 'good', 'very good'])  # Set labels
plt.xlabel('Subjective assessment of situation')
plt.ylabel('metric value')
# plt.legend(loc='lower left')


plt.tight_layout()
plt.savefig(save_file, dpi=400)

plt.show()





