#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 23:36:16 2021

@author: Timo Sommer

This script is for doing some tests and plots of my implementation of DisorderedSOAP.
"""
import os
soap_dir = '/home/timo/Masterarbeit/Rechnungen/Datasets/Experiments/Chem_formula_matched/Sc_ICSD_chem_formula_manipulated/features/SOAP'

csv_distances = 'Doping_distances.csv'
csv_svd = 'SOAP_SVD_{n_components}.csv'
soap_feats = 'SOAP_sparse.npz'
svd_models = 'SVD_{n_components}.z'

save_figs_dir = '/home/timo/Masterarbeit/Analysen/Features/DisorderedSOAP'

# all_n_svd_components = [50, 100, 200, 500, 1000]
all_n_svd_components = [1000]

csv_distances = os.path.join(soap_dir, csv_distances)
csv_svd = os.path.join(soap_dir, csv_svd)
soap_feats = os.path.join(soap_dir, soap_feats)
svd_models = os.path.join(soap_dir, svd_models)


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from Dataset_preparation.Calc_similarities import get_formula_diff
from Dataset_preparation.Check_dataset import get_chem_dict
from sparse import load_npz
import joblib
from DisorderedSOAP import distance
from scipy.spatial.distance import pdist, squareform, cdist
from sklearn.manifold import TSNE
import itertools
from Own_functions import movecol





def avg_dist(el, df):
    entries_with_el = df['chem_comp'].str.contains(f'-{el}-')
    distances = df.loc[entries_with_el]['L2_distance']
    avg = distances.mean()
    return(avg)

def L2_distance(point, coords):
    dist = np.sqrt(np.sum((point - coords)**2, axis=0))
    return(dist)
    
def label_point(x, y, val, ax, add_distance={}):
    for x_loc, y_loc, el in zip(x, y, val):
        x_loc *= 1.1
        if el in add_distance:
            dist = add_distance[el]
            x_loc += dist[0]
            y_loc += dist[1]
        ax.text(x_loc, y_loc, el, fontsize=9)
        
def reconstruction_error(soap_features, svd, df_svd):
    """Returns a df with the MSE between the original SOAP features and the reconstruction of the SVD for each element.
    """
    batch_size = 300    # For compromise of runtime and memory.
    data = []
    for i in range(0, len(soap_features), batch_size):
        start = i
        end = i + batch_size
        svd_batch = svd_features[start:end, :]
        inverse_soap_batch = svd.inverse_transform(svd_batch)
        soap_batch = soap_features[start:end, :].todense()
        l2dist = distance(inverse_soap_batch, soap_batch, axis=1)
        n_zeros = np.sum(soap_batch == 0, axis=1)
        inverse_n_zeros = np.sum(inverse_soap_batch == 0, axis=1)
        frac_zeros = inverse_n_zeros / n_zeros
        data.extend([{
                        'L2_distance': l2dist,
                        'frac_zeros_recognized': frac_zeros                        
                    } for l2dist, frac_zeros in zip(l2dist, frac_zeros)])
    df_values = pd.DataFrame(data)
    df_svd = pd.concat([df_svd, df_values], axis='columns')
    
    df_svd['elements'] = df_svd['formula_sc'].apply(lambda formula: list(get_chem_dict(formula).keys()))
    elements = [el for els in df_svd['elements'] for el in els]    # flatten
    elements = pd.Series(data=elements, name='Count').value_counts()
    elements = elements.reset_index().rename(columns={'index': 'Element', 'Count': 'Occurrences'})
    df_svd['chem_comp'] = df_svd['elements'].apply(lambda els: '-' + '-'.join(els) + '-')
    elements['MSE'] = elements['Element'].apply(lambda el: avg_dist(el, df_svd))
    return(df_svd, elements)
        
        

df_distances = pd.read_csv(csv_distances, header=1)
soap_features = load_npz(soap_feats)



#%%
# Get a table with comparisons of the formula `compare` with all other data points.
compare = 'Ba2Cu3Y1O6'
metric = 'cosine'
frac_random_entries = 1
n_svd_components = 1000
filepath = csv_svd.replace('{n_components}', f'{n_svd_components}')
df = pd.read_csv(filepath, header=1).sample(frac=frac_random_entries)
svd_cols = [col for col in df.columns if col.startswith('SVD')]
ybco = df[df['formula_sc'] == compare].iloc[0].to_frame().T
df[f'distance_{compare}'] = cdist(ybco[svd_cols].to_numpy(), df[svd_cols].to_numpy(), metric=metric).T
df = movecol(df, [f'distance_{compare}'], 'formula_sc')
df1 = df[[col for col in df.columns if not col in svd_cols]]

#%%
# Get a TSNE of the final compressed SOAP features.
frac_random_entries = 1
n_svd_components = 1000
filepath = csv_svd.replace('{n_components}', f'{n_svd_components}')
df = pd.read_csv(filepath, header=1).sample(frac=frac_random_entries)
svd_cols = [col for col in df.columns if col.startswith('SVD')]
plt.figure()
df[['embedding 0', 'embedding 1']] = TSNE().fit_transform(df[svd_cols].to_numpy())
sns.scatterplot(data=df, x='embedding 0', y='embedding 1', hue='Class1_sc')
# plt.title(f'TSNE ({frac_random_entries*100:.0f}% of data, $n_{{SVD}} = {n_svd_components}$)')
plt.title(f'TSNE ($n_{{SVD}} = {n_svd_components}$)')
plt.gca().legend().set_title('')
fig_name = 'TSNE.png'
plt.savefig(os.path.join(save_figs_dir, fig_name), dpi=300)
plt.show()


#%%
# Plot a scatterplot of cosine distance vs rel. Delta_chem_formula for a subset of all final data points.
frac_random_entries = 0.05
df = pd.read_csv(filepath, header=1).sample(frac=frac_random_entries)
# df = df[svd_cols].astype('float32')
mutual_distances = pd.DataFrame(itertools.combinations(df['formula_sc'], 2), columns=['i','j'])
mutual_distances[['cifname_i', 'cifname_j']] = list(itertools.combinations(df['cifname'], 2))
mutual_distances[['Class_i', 'Class_j']] = list(itertools.combinations(df['Class1_sc'], 2))
mutual_distances['Cosine distance'] = pdist(df[svd_cols].to_numpy(), metric)
mutual_distances['Classes'] = [d1 if d1 == d2 else 'Not same' for d1, d2 in zip(mutual_distances['Class_i'], mutual_distances['Class_j'])]
mutual_distances['Delta_chem_formula'] = mutual_distances.apply(lambda row: get_formula_diff(row['i'], row['j'], mode='rel'), axis=1) / 2
mutual_distances_sq = pd.DataFrame(data=squareform(mutual_distances['Cosine distance']), index=df['formula_sc'], columns=df['formula_sc'])
print(f'Size mutual_distances_sq: {mutual_distances_sq.memory_usage(index=True, deep=True).sum()/1e9} GB')
plt.figure()
sns.scatterplot(data=mutual_distances, x='Delta_chem_formula', y='Cosine distance', s=20, alpha=0.1)
plt.xlabel('$\Delta_{chem. formula}$')
plt.title(f'Comparison of final features ({frac_random_entries*100:.0f}% of data, $n_{{SVD}}$ = {n_svd_components})')
fig_name = 'Scatterplot_final_features_cosine_dist.png'
plt.savefig(os.path.join(save_figs_dir, fig_name), dpi=300)
plt.show()




#%%
# Get reconstruction error of SVD for each model with n_components.
all_elements = {}
explained_variances = []
frac_recognized_zeros = []
for idx, n_svd_components in enumerate(all_n_svd_components):
    print(f'SVD components: {n_svd_components}')
    # Read in dataframe with SVD features, the original SOAP features and the SVD model.
    filepath = csv_svd.replace('{n_components}', f'{n_svd_components}')
    modelpath = svd_models.replace('{n_components}', f'{n_svd_components}')
    df_svd = pd.read_csv(filepath, header=1)
    svd_cols = [col for col in df_svd.columns if col.startswith('SVD')]
    svd_features = df_svd[svd_cols].to_numpy()
    svd = joblib.load(modelpath)

    df_svd, mse_elements = reconstruction_error(soap_features, svd, df_svd)
    all_elements[n_svd_components] = mse_elements
    total_explained_variance = sum(svd.explained_variance_ratio_)
    explained_variances.append(total_explained_variance)
    recognized_zeros = df_svd['frac_zeros_recognized'].mean()
    frac_recognized_zeros.append(recognized_zeros)
    

#%%
# Artificially shift point labels in data coordinates.
shift_point_labels = {
                   50:  {'Pu': (-2.2, 0), 'Pa': (0, -0.06), 'I': (0, -0.01)},
                   100: {'Pu': (-2.2, 0), 'Pa': (0, -0.06), 'Cm': (0, -0.03)},
                   200: {'Np': (0, 0.03)},
                   500: {'Cm': (0, -0.03), 'Am': (0, 0.02)},
                   1000:{'Cm': (0, -0.025), 'Po': (-2.3, -0.003), 'Pu': (-2.1, 0), 'Np': (-1, 0.02)}
                }
plot_max_MSE = 1.7
for explained_variance, n_svd_components, recognized_zeros in zip(explained_variances, all_n_svd_components, frac_recognized_zeros):
    elements = all_elements[n_svd_components]
    plt.figure()
    sns.scatterplot(data=elements, x='Occurrences', y='MSE')
    plt.title(f'Reconstruction error of SVD (n={n_svd_components}) per element')
    plt.xscale('log')
    plt.ylim(0, plot_max_MSE)
    textbox = f'Explained variance: {explained_variance:.4f}\nRecognized zeros: {recognized_zeros:.3f}'
    plt.annotate(textbox, xy=(0.56, 0.88), xycoords='axes fraction', bbox={'boxstyle': 'round', 'facecolor': 'none'})
    # Label some data points.
    label_elements = (elements['Occurrences'] < 40) | (elements['MSE'] > 0.62)
    label_elements = elements[label_elements]
    label_point(label_elements['Occurrences'], label_elements['MSE'], label_elements['Element'], plt.gca(), shift_point_labels[n_svd_components])
    fig_name = f'Scatterplot_reconstruction_error_{n_svd_components}.png'
    plt.savefig(os.path.join(save_figs_dir, fig_name), dpi=300)
    plt.show()

# Plot last plot without fixed y axis for more details.
explained_variance = explained_variances[-1]
n_svd_components = all_n_svd_components[-1]
recognized_zeros = frac_recognized_zeros[-1]
shift_point_labels = {'Po': (-2.2, 0.002)}
elements = all_elements[n_svd_components]
plt.figure()
sns.scatterplot(data=elements, x='Occurrences', y='MSE')
plt.title(f'Reconstruction error of SVD (n={n_svd_components}) per element')
plt.xscale('log')
# plt.ylim(0, plot_max_MSE)
textbox = f'Explained variance: {explained_variance:.4f}\nRecognized zeros: {recognized_zeros:.3f}'
plt.annotate(textbox, xy=(0.56, 0.88), xycoords='axes fraction', bbox={'boxstyle': 'round', 'facecolor': 'none'})
# Label some data points.
label_elements = (elements['Occurrences'] < 40) | (elements['MSE'] > 0.62)
label_elements = elements[label_elements]
label_point(label_elements['Occurrences'], label_elements['MSE'], label_elements['Element'], plt.gca(), shift_point_labels)
fig_name = f'Scatterplot_reconstruction_error_{n_svd_components}_details.png'
plt.savefig(os.path.join(save_figs_dir, fig_name), dpi=300)
plt.show()

#%%
# Plot the reconstruction error per domain.
plt.figure()
order = df_svd.groupby('Class1_sc')['L2_distance'].apply(np.mean).sort_values().index.tolist()[::-1]
sns.barplot(data=df_svd, x='Class1_sc', y='L2_distance', order=order)
plt.title('Reconstruction error per domain ($n_{SVD} = 1000$)')
plt.xlabel('')
plt.ylabel('MSE')
fig_name = 'Barplot_reconstruction_error_per_domain.png'
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(save_figs_dir, fig_name), dpi=300)
plt.show()
    
    
#%%
        
        
# Get relative distances of chemical formulas
df_distances['form_absdiff'] = df_distances.apply(lambda row: get_formula_diff(row['original_doped_formula'], row['ordered_formula'], mode='rel'), axis=1)

df_distances = df_distances.rename(columns={'L2_distance': 'MSE'})
df_distances['MAE_linear'] = 2.1 - 2.1 * df_distances['weight'] - df_distances['MSE']
df_distances['linear'] = df_distances['MAE_linear'].abs() < 0.1

df_distances['Num. elements'] = df_distances['ordered_formula'].apply(lambda formula: len(list(get_chem_dict(formula))))

df_distances['Occupancy'] = df_distances.apply(lambda row: sum(list(get_chem_dict(row['ordered_formula']).values())) != sum(list(get_chem_dict(row['original_doped_formula']).values())), axis=1)
df_distances['Occupancy'] = df_distances['Occupancy'].apply(lambda upscaled: 'Vacancies' if upscaled else 'Full')

df_distances = df_distances.sample(frac=1)


#%%

# Plot weights over MAE
plt.figure()
sns.scatterplot(data=df_distances, x='weight', y='MSE', hue='Num. elements', palette='flare', style='Occupancy', markers=['o', '^'], alpha=0.8)
plt.title('Comparison of original and proxy structures')
fig_name = 'Scatterplot_MSE_weight.png'
plt.savefig(os.path.join(save_figs_dir, fig_name), dpi=300)
plt.show()

#%%
# Plot form_absdiff over MAE
plt.figure()
sns.scatterplot(data=df_distances, x='form_absdiff', y='MSE', hue='Num. elements', palette='flare', style='Occupancy', markers=['o', '^'], alpha=0.8)
# plt.xscale('log')
# plt.yscale('log')
# plt.xlim(0, 60)
# plt.ylim(0, 3)
plt.title('Comparison of original and proxy structures')
plt.xlabel('$\Delta_{chem. formula}$')
fig_name = 'Scatterplot_MSE_form_reldiff.png'
plt.savefig(os.path.join(save_figs_dir, fig_name), dpi=300)
plt.show()




