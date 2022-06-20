"""
Created on Fri Jul 23 15:42:24 2021

@author: Timo Sommer

This script generates MAGPIE features, graphs and SOAP features of possibly disorderd structures. Additionally to the original SOAP features the same features applied to a PCA are added.
"""
from superconductors_3D.utils.projectpaths import projectpath
import os
import argparse
import pandas as pd
import numpy as np
import sys
from pymatgen.core.structure import Structure
from datetime import datetime
from sklearn.decomposition import PCA, TruncatedSVD
from superconductors_3D.dataset_preparation.utils.DisorderedSOAP.DisorderedSOAP import DisorderedSOAP, get_all_elements, distance
from superconductors_3D.machine_learning.own_libraries.own_functions import write_to_csv
from scipy.sparse import save_npz
from scipy.spatial.distance import pdist, euclidean
import joblib
import sparse as sps
from pymatgen.io.cif import CifParser
from superconductors_3D.machine_learning.own_libraries.models.GNN.GNN_utils import save_graph, get_graph_from_structure
from pathlib import Path
from joblib import Parallel, delayed, cpu_count
import shutil
from superconductors_3D.dataset_preparation.utils.lib_generate_datasets import MAGPIE_features
from megnet.data.crystal import CrystalGraphDisordered, get_elemental_embeddings
import megnet
from megnet.models import MEGNetModel
import warnings


def get_MAGPIE_features(df):
    
    print('Calculate MAGPIE features.')
    
    formulas = df['formula_sc']
    magpie_features = MAGPIE_features(formulas)
    
    magpie_names = magpie_features.columns
    magpie_names = [f'MAGPIE_{name}' for name in magpie_names]
    
    df[magpie_names] = magpie_features.to_numpy()
    
    assert df[magpie_names].notna().all().all()
    
    return df

def get_graph_from_row(row, output_graph_dir, model):
    
    cif = row.cif_2
    try:
        # Get graph from structure
        graph = get_graph_from_structure(cif_relpath=cif, model=model)
        
        # Save graph
        graph_name = Path(cif).stem + '.json'
        graph_rel_path = os.path.join(output_graph_dir, graph_name)
        save_graph(graph_name=graph_rel_path, graph=graph)
        
    except Exception as e:
        # Issue with graph, skip this entry.
        print('Exception raised in getting graph from structure:', e)
        graph_rel_path = np.nan
        
    result_dict = row.to_dict()
    result_dict['graph'] = graph_rel_path
    
    return result_dict

def get_graphs(df, output_graph_dir, n_cpus, nfeat_bond=100, r_cutoff=4, gaussian_width=0.5):
    """Calculate graphs of each structure. The nodes will be represented by a dict of elements and occupancies so that the numerical features can be calculated later.
    """    
    # Make graph directory and delete previous one.
    abs_output_graph_dir = projectpath(output_graph_dir)
    if os.path.exists(abs_output_graph_dir):
        shutil.rmtree(abs_output_graph_dir)
    os.mkdir(abs_output_graph_dir)
    
    # Create class for creating graphs from cif.
    gaussian_centers = np.linspace(0, r_cutoff + 1, nfeat_bond)    
    graph_converter = CrystalGraphDisordered(cutoff=r_cutoff)
    model = MEGNetModel(graph_converter=graph_converter, centers=gaussian_centers, width=gaussian_width)
    
    # MEGNet model cannot be pickled, therefore only n_jobs=1 possible.
    with Parallel(n_jobs=1, verbose=1) as parallel:
        data = parallel(delayed(get_graph_from_row)(row, output_graph_dir, model) for _, row in df.iterrows())
    
    # Get data with and without graphs.
    df_correct = pd.DataFrame(data)
    graph_exists = df_correct['graph'].notna()
    df_excluded = df_correct[~ graph_exists]
    df_excluded['Reason for exclusion'] = 'No graph'
    df_correct = df_correct[graph_exists]
    
    print(f'Exclude {len(df_excluded)} entries because they have isolated atoms in their graphs.')
    
    return df_correct, df_excluded

def calculate_SOAP_features(df, rcut, nmax, lmax, sigma, crossover, n_pca_components, n_cpus, symprec=0.01):
    """Calculates SOAP features for all structures with the given hyperparameters.
    """

    print(f'Start generating SOAP features with {n_cpus} cpus.')
    
    elements = get_all_elements(df['formula_sc'])
    print('Number of elements:', len(elements))
    soap = DisorderedSOAP(
                            rcut=rcut,
                            nmax=nmax,
                            lmax=lmax,
                            sigma=sigma,
                            species=elements,
                            sparse=True,
                            crossover=crossover,
                            periodic=True,
                            dtype='float32',
                            symprec=symprec
                            )
    
    print(f'Number of SOAP features: {soap.soap.get_number_of_features()}')
    print('Start calculating disordered SOAP features.')
    all_structs = [Structure.from_file(projectpath(cif)) for cif in df['cif_2']]
    soap_features = soap.create(all_structs, n_jobs=n_cpus, return_distances=True)
    print('Size of soap_features in sparse:', soap_features.data.nbytes)
    print('Shape of soap_features:', soap_features.shape)
    
    
    # Reduce dimension with SVD and PCA.
    
    # TruncatedSVD takes only csr and csc sparse format.
    soap_features = soap_features.tocsc()
    
    # Add pure SOAP features to df.
    soap_names = [f'SOAP_{i}' for i in range(soap_features.shape[1])]
    df[soap_names] = soap_features.todense()
    
    # Make PCA of SOAP features.
    if n_pca_components > 0:
        
        n_data_points = len(df)
        if n_pca_components > n_data_points:
            warnings.warn(f'Cannot make a PCA if the number of specified PCA components (here n_pca_components={n_pca_components}) is greater than the number of data points (here n_data_points={n_data_points}). Skipping PCA calculation.')
            return df
        
        starttime = datetime.now()

        print(f'Reduce number of features using PCA to {n_pca_components}.')
        soap_features = soap_features.todense()
        pca = PCA(n_components=n_pca_components, copy=False)
        pca_features = pca.fit_transform(soap_features)
        
        explained_variance = pca.explained_variance_ratio_
        total_explained_variance = sum(explained_variance)
        print(f'Total explained variance of PCA: {total_explained_variance:.4f}')
           
        # Make dataframe with pca features.
        pca_names = [f'PCA_SOAP_{i}' for i in range(n_pca_components)]
        df[pca_names] = pca_features
    
        duration = datetime.now() - starttime
        print(f'Done PCA with {n_pca_components} components. Duration: {duration}')
    
    return df

def generate_features(input_csv_data, output_graph_dir, output_csv_data, excluded_output_csv_data, rcut, nmax, lmax, sigma, crossover, n_pca_components, n_cpus, symprec=0.01):

    df = pd.read_csv(input_csv_data, header=1)  
    
    # graphs
    df_correct, df_excluded = get_graphs(df, output_graph_dir, n_cpus)    
    
    # MAGPIE
    df_correct = get_MAGPIE_features(df_correct)

    # SOAP
    df_correct = calculate_SOAP_features(df_correct, rcut, nmax, lmax, sigma, crossover, n_pca_components, n_cpus, symprec)
    
    
    # Save output df.
    comment = f'Added MAGPIE and SOAP features and graphs.'
    excluded_comment = f'All entries where the structure to graph conversion didn\'t work.'
    write_to_csv(df_correct, output_csv_data, comment)
    write_to_csv(df_excluded, excluded_output_csv_data, excluded_comment)





if __name__ == '__main__':
        
    database = 'MP'         # change this or parse from cmd
    n_cpus = 1             # change this or parse from cmd
    
    
    
    # Parse input and overwrite.
    parser = argparse.ArgumentParser()
    parser.add_argument('--database', '-d', type=str)
    parser.add_argument('--n-cpus', '-n', dest='n_cpus', type=int)
    args = parser.parse_args()
    database = args.database if not args.database is None else database
    n_cpus = args.n_cpus if not args.n_cpus is None else n_cpus
    
    
    
    input_csv_data = projectpath('data', 'intermediate', database, f'4_SC_{database}_synthetically_doped.csv')
    
    output_graph_dir = os.path.join('data', 'intermediate', database, 'graphs')
    output_csv_data = projectpath('data', 'intermediate', database, f'5_features_SC_{database}.csv')
    excluded_output_csv_data = projectpath('data', 'intermediate', database, f'excluded_5_features_SC_{database}.csv')
    
    
    n_pca_components = 100
    
    # Hyperparameters for SOAP from 2021 Fung
    rcut = 4.270
    nmax = 6
    lmax = 4
    sigma = 0.336
    crossover = False

    generate_features(input_csv_data, output_graph_dir, output_csv_data, excluded_output_csv_data, rcut, nmax, lmax, sigma, crossover, n_pca_components)
    
