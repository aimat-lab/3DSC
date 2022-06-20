#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 15:48:20 2021

@author: Timo Sommer

This script contains my own implementation of a class that generates SOAP features for disordered structures.
"""
import pandas as pd
import numpy as np
from dscribe.descriptors import SOAP
from superconductors_3D.dataset_preparation.utils.check_dataset import get_chem_dict, standardise_chem_formula
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.transformations.site_transformations import ReplaceSiteSpeciesTransformation
from itertools import product
from pymatgen.io.ase import AseAtomsAdaptor
from scipy.spatial.distance import pdist
from joblib import Parallel, delayed, cpu_count
from scipy.sparse import coo_matrix, lil_matrix
import sparse as sps
from warnings import warn
from datetime import datetime

def get_elements(formula):
    """Returns a list of the elements in a string of a chemical formula.
    """
    chemdict = get_chem_dict(formula)
    elements = list(chemdict.keys())
    return(elements)

def get_all_elements(formulas):
    """Returns a unique list of all elements from a list of chemical formulas as strings.
    """
    formulas = pd.Series(formulas)
    element_lists = formulas.apply(get_elements)
    all_elements = list(element_lists.explode().unique())
    return(all_elements)

def distance(v1, v2, axis=0):
    """Calculates the L2 distance for two vectors. Also works for sparse vectors."""
    distance = np.sqrt(np.sum((v1-v2)**2, axis=axis))
    return(distance)



class DisorderedSOAP():
    """Class for generating SOAP features even for disordered structures with doping. Uses the Dscribe library to generate SOAP features for ordered structures close to the doped structure and then does a weighted average of these structures to get a final vector of SOAP features.
    """
    
    def __init__(self, average='on', sparse=False, symprec=0.01, **kwargs):
        """Keywords work exactly as in the Dscribe implementation of SOAP. The only difference are `average` and `sparse`.
        `average: Inner averaging is not possible anymore. No averaging would be possible but is currently not supported. Currently this option is always set to doing a weighted average of sites, weighted by the total occupancy.
        `sparse`: Calculate with and output a sparse matrix.
        `symprec` (float) – Tolerance for symmetry finding. Defaults to 0.01, which is fairly strict and works well for properly refined structures with atoms in the proper symmetry coordinates. For structures with slight deviations from their proper atomic positions (e.g., structures relaxed with electronic structure codes), a looser tolerance of 0.1 (the value used in Materials Project) is often needed.
        """
        self.soap = SOAP(average='off', sparse=sparse, **kwargs)
        self.sparse = sparse
        self.symprec = symprec
    
    def if_disordered(self, site):
        disordered = len(site.species) > 1
        return(disordered)
    
    def total_occupancy(self, site):
        """Returns the total occupancy of a site of a pymatgen structure.
        """
        species_and_occu = site.species.as_dict()
        total_occupancy = sum(species_and_occu.values())
        return(total_occupancy)
    
    def coo_to_lil(self, vector):
        """Transforms `sparse.core.coo` to `lil` matrix."""
        length = len(vector)
        vector = vector.broadcast_to((1, length))
        vector = vector.to_scipy_sparse().tolil()
        return(vector)

    def normalise_total_occupancy_of_site(self, struct):
        """If a site has a total occupancy of less than 1, upscale the site to having a total occupancy of 1 and record the previous total occupancy as weight for this site.
        """
        site_weights = []
        upscale = {}
        eps = 1e-8
        for i, site in enumerate(struct):
            species_and_occu = site.species.as_dict()
            total_occupancy = sum(species_and_occu.values())
            site_weights.append(total_occupancy)
            if total_occupancy > 1+eps:
                warn(f'Total occupancy > 1: {total_occupancy}')
            if total_occupancy != 1:
                if len(species_and_occu) == 1:
                    upscale[i] = list(species_and_occu.keys())[0]
                elif len(species_and_occu) > 1:
                    factor = 1 / total_occupancy
                    upscale[i] = {species: occu*factor for species, occu in species_and_occu.items()}
        make_sites_ordered = ReplaceSiteSpeciesTransformation(upscale)
        upscaled_struct = make_sites_ordered.apply_transformation(struct)
        ordered_or_doped = [site.is_ordered if len(site.species) == 1 else True for site in upscaled_struct]
        assert all(ordered_or_doped)
        return(upscaled_struct, site_weights)
    
    def get_disordered_sites(self, symm_struct):
        sites = [equiv_sites[0] for equiv_sites in symm_struct.equivalent_sites]
        all_indices = symm_struct.equivalent_indices
        disordered_sites = {key: [] for key in ['elements', 'occupancies', 'indices']}
        for indices, site in zip(all_indices, sites):
            if self.if_disordered(site):
                disordered_sites['elements'].append(list(site.species.as_dict().keys()))
                disordered_sites['occupancies'].append(list(site.species.as_dict().values()))
                disordered_sites['indices'].append(indices)
        return(disordered_sites)
    
    def get_all_atoms(self, ordered_struct):
        """Depending on whether the structure has pseudo valencies or not, ordered_struct.species will be either a species or already an element.
        """
        try:
            pymatgen_atoms = [species.element.name for species in ordered_struct.species]
        except AttributeError:
            pymatgen_atoms = [species.name for species in ordered_struct.species]
        return pymatgen_atoms
    
    def get_ordered_ase_structures_and_weights(self, pymatgen_struct):
        """Takes a list of paths to cifs, possibly with disordered atom sites. Returns:
        `ase_structures`: A list of all possible combinations of ordered ase structures for each doped site.
        `doping_weights`: The weight (the probability) of this ordered structure given the doping occupancy.
        `site_weights:` The initial total occupancy of each site. Sites with total occupancy less than 1 should be weighted less in the average of the sites.
        """
        ase_structures = []
        doping_weights = []
    
        symm_analyzer = SpacegroupAnalyzer(pymatgen_struct, symprec=self.symprec)
        symm_struct = symm_analyzer.get_symmetrized_structure()
        # print(symm_struct)
        disordered_sites = self.get_disordered_sites(symm_struct)
        
        # Return only one structure if it already is fully ordered.
        structure_ordered = len(disordered_sites['elements']) == 0
        assert pymatgen_struct.is_ordered == structure_ordered
        if structure_ordered:
            ase_struct = AseAtomsAdaptor().get_atoms(pymatgen_struct)
            weight = 1
            ase_structures.append(ase_struct)
            doping_weights.append(weight)
            return(ase_structures, doping_weights)
        
        assert np.allclose([sum(occus) for occus in disordered_sites['occupancies']], 1)
        element_combs = product(*disordered_sites['elements'])
        occupancy_combs = product(*disordered_sites['occupancies'])
        for all_elements, all_occupancies in zip(element_combs, occupancy_combs):
            # Get all combinations of ordered structures that would make up the disordered structure and use the probability of occurence of this structure as weight.
            weight = np.product(all_occupancies)
            replace_sites = {}
            for element, indices in zip(all_elements, disordered_sites['indices']):
                for idx in indices:
                    replace_sites[idx] = element
            make_sites_ordered = ReplaceSiteSpeciesTransformation(replace_sites)
            ordered_struct = make_sites_ordered.apply_transformation(pymatgen_struct)
            ase_struct = AseAtomsAdaptor().get_atoms(ordered_struct)
            
            pymatgen_atoms = self.get_all_atoms(ordered_struct)
            ase_atoms = ase_struct.get_chemical_symbols()
            assert pymatgen_atoms == ase_atoms  # Make sure that atomic sites are the same
            ase_structures.append(ase_struct)
            doping_weights.append(weight)
        # print(pymatgen_struct)
        # print(ordered_struct)
        # print(symm_struct)
        # print(f'Chemical formulas: {[atom.get_chemical_formula() for atom in ase_structures]}')
        return(ase_structures, doping_weights)
    
    def split_in_soap_vectors_and_distances(self, results):
        """Assign the two dimensions of results (soap vectors and distances) to own variables.
        """
        doping_distances = []
        n_soap_features = self.soap.get_number_of_features()
        shape = (self.n_structs, n_soap_features)
        # Use lil matrix to construct sparse matrix because this is the fastest version.
        soap_vectors = lil_matrix(shape) if self.sparse else np.empty(shape)
        for i, (soap_vector, distances) in enumerate(results):        
            # Construct soap array.
            soap_vector = self.coo_to_lil(soap_vector) if self.sparse else soap_vector
            soap_vectors[i,:] = soap_vector
            # Construct doping distances.
            for dist in distances:
                dist['idx'] = i
                doping_distances.append(dist)
        if self.sparse:
            soap_vectors = sps.COO.from_scipy_sparse(soap_vectors)
        return(soap_vectors, doping_distances)
    
    def weighted_average(self, array, weights, axis=0):
        """Computes the weighted average. Can deal with sparse input unlike np.average().
        """
        # setup weights to broadcast along axis
        wgt = np.asanyarray(weights)
        wgt = np.broadcast_to(wgt, (array.ndim-1)*(1,) + wgt.shape)
        wgt = wgt.swapaxes(-1, axis)
        
        total = np.sum(wgt)
        avg = np.sum(array * wgt, axis=axis) / total
    
        # Sanity check            
        if type(array).__module__ == 'numpy':
            avg2 = np.average(array, axis, weights)
            assert np.all(avg == avg2)
        return(avg)
    
    def get_soap_vector(self, raw_pymatgen_struct):
        """Returns a vector with SOAP features for each cif structure in cif_paths. In contrast to the standard SOAP features this is also possible for doped structures.
        """
        # If a site has a total occupancy of less than 1 we need to use the total occupancy as weighting factor for this atom site before averaging the sites.
        pymatgen_struct, site_weights = self.normalise_total_occupancy_of_site(raw_pymatgen_struct)
        
        # For doped structures get all combinations of ordered structures with the regarding probabilities (weights).
        ase_structures, doping_weights = self.get_ordered_ase_structures_and_weights(pymatgen_struct)
        
        # Generate the SOAP features for each site of each ordered structure.
        site_features = self.soap.create(ase_structures)
        # Get weighted average of occupancy of each site.
        structure_features = self.weighted_average(site_features, site_weights, axis=1)            
        # Get weighted average of probability of each ordered structure (given the doped structure).
        soap_features = self.weighted_average(structure_features, doping_weights, axis=0)
        
        doping_distances = []
        if self.return_distances:
            # Get the L2 distances of the soap vector of the original structure and the soap vector of the ordered structure for each doping possibility.
            orig_formula = standardise_chem_formula(raw_pymatgen_struct.formula)
            for struct, vec, weight in zip(ase_structures, structure_features, doping_weights):
                ordered_formula = standardise_chem_formula(struct.get_chemical_formula())
                dist = distance(soap_features, vec)
                doping_distances.append({
                                            'original_doped_formula': orig_formula,
                                            'ordered_formula': ordered_formula,
                                            'L2_distance': dist,
                                            'weight': weight
                                        })
        return(soap_features, doping_distances)
    
    def create(self, system, n_jobs=1, return_distances=False):#, only_physical_cores=False, verbose=False):
        """Create DSOAP features for the given systems. 
        
        `system`: list of pymatgen structures, possibly disordered and with doping.
        """
        self.return_distances = return_distances
        self.n_structs = len(system)
        assert type(system) == list
        
        with Parallel(n_jobs=n_jobs, verbose=1) as parallel:
            results = parallel(delayed(self.get_soap_vector)(structure) for structure in system)
        now = datetime.now()
        soap_vectors, doping_distances = self.split_in_soap_vectors_and_distances(results)
        print(f'Splitting done in {datetime.now() - now}')
        
        if self.return_distances:
            self.df_distances = pd.DataFrame(doping_distances)
        return(soap_vectors)

