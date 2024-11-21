<p align="center">
  <img src="resources/3DSC_logo_small.png">
</p>

# The 3DSC database
This repository contains data and code of the 3DSC database, the first extensive database of superconductors with their critical temperature *T*<sub>c</sub> and their three-dimensional crystal structure. We describe the database and the algorithm to generate it in our paper https://www.nature.com/articles/s41597-023-02721-y.

## Using the 3DSC database

The 3DSC database can be found on figshare under https://doi.org/10.6084/m9.figshare.c.6914407.v1. The zip file can be extracted via the command `tar -xzvf 3DSC_MP.tar.gz`.

Alternatively, the 3DSC<sub>MP</sub> database can be found under `superconductors_3D/data/final/MP`. The file `3DSC_MP.csv` contains the 3DSC<sub>MP</sub> in tabular form. The three most important columns are the following:

- `formula_sc`: The chemical formula of the material, which is exactly the original chemical formula of the SuperCon entry. Note that due to the normalization of chemical formulas in the matching algorithm, this chemical formula might deviate by a constant factor from the chemical formula of the primitive unit cell of the structure.
- `tc`: The critical temperature in Kelvin. Non-superconductors have a *T*<sub>c</sub> of 0.
- `cif`: The path to the cif file of the 3DSC<sub>MP</sub> crystal structure relative to the directory `3DSC/superconductors_3D/`. If the name contains `synth_doped` this means that this structure was artificially doped.

Additionally to these three basic columns of the 3DSC<sub>MP</sub> database, there are a lot of columns which were important in the matching and adaptation algorithm, which are from the original Materials Project database or which were important for the analysis in our paper. These columns are less important if you just want to use the 3DSC<sub>MP</sub>, but they might be interesting for everyone looking to improve the 3DSC<sub>MP</sub> or reproduce the results in our paper.

- `MAGPIE.*`: MAGPIE feature vectors of the chemical formula of this material. Missing in the github version (see note below).
- `SOAP.*`: DSOAP feature vectors of the structure. Missing in the github version (see note below).
- `.*_2`: All columns ending with `_2` are the columns from the original structure from the Materials Project or columns added in the process of cleaning the initial Materials Project database.
- `totreldiff`: The $\Delta_\mathrm{totrel}$ from our paper[1], a measure of the difference between the original chemical formula of the SuperCon and of the Materials Project.
- `formula_frac`: The normalization factor of the chemical formulas.
- `sc_class`: The superconductor group (either 'Other', 'Heavy_fermion', 'Chevrel', 'Oxide', 'Cuprate', 'Ferrite', 'Carbon'). Some of the entries also have 'OxideHeavy_fermion' or 'Heavy_fermionChevrel', which means that the algorithm could not uniquely attribute this material into one group.
- `weight`: The sample weight which was used for the XGB model and the calculcation of the scores. This is just the inverse of the number of crystal structures per SuperCon entry in the database.
- `cif_before_synthetic_doping`: The path to the original cif file of the Materials Project before artificial doping.
- `crystal_temp_2`: The crystal temperature. Non-zero only for the 3DSC<sub>ICSD</sub>.
- `no_crystal_temp_given_2`: If the crystal temperature was not explicitly given. Always True in the 3DSC<sub>MP</sub>. In the 3DSC<sub>ICSD</sub>, this is True if no crystal temperature was given and 293K was assumed.
- `cubic`, `hexagonal`, `monoclinic`, `orthorhombic`, `tetragonal`, `triclinic`, `trigonal`, `primitive`, `base-centered`, `body-centered`, `face-centered`: The symmetry features as described in the supporting information of our paper.

Note that in the github version of this dataset, we have removed the `SOAP.*` and the `MAGPIE.*` columns due to memory constraints. You can get these columns by executing the matching and adaptation algorithm as described below.

## How to cite the 3DSC
Please cite our paper as given in [1].

## Generating the 3DSC database

### Installation

The following installation instructions have been tested on Mac, but they should work also for Linux and Windows. If you have any problem with this, please open an issue.
1. Clone the 3DSC repository
```sh
git clone https://github.com/aimat-lab/3DSC.git
```
2. Change into the cloned directory
```sh
cd 3DSC
```
3. Install the package `superconductors_3D` into the current environment via
 ```sh
pip install .
 ```
Sometimes, this can throw an error because of an issue with `setuptools>58.0.0`. In this case, install `setuptools=58` before installing the package, for example like this:
```sh
conda create --name 3DSC python=3.9 setuptools=58 pip
conda activate 3DSC
pip install .
```

### Generate the 3DSC<sub>MP</sub> database
You can run the code to reproduce the 3DSC database, based on structures either from the Materials Project (MP) or the ICSD. The following command will generate the 3DSC<sub>MP</sub> using one core with data stored at `superconductors_3D/data`, i.e. this command works if you are in the cloned directory. 
```sh
make_3DSC -d MP -n 1 -dd superconductors_3D/data
```
The command `make_3DSC` automatically runs through all stages of the matching and adaptation algorithm described in the paper: In step 0-2, the cif files, the Materials Project database and the SuperCon database provided under `superconductors_3D/data/source/MP` are cleaned. In step 3, the SuperCon entries and the crystal structures are matched based on their chemical composition. In step 4, artificial doping is performed for all matches where the relative chemical formula doesn't match perfectly. In step 5, the chemical composition and the three-dimensional crystal structure are featurized using the MAGPIE and the Disordered SOAP algorithm. The latter is an extension of the SOAP algorithm which is described in the SI of our paper. Finally, matches are ranked and only the best matches are kept. Note that in general multiple matches can be ranked equally and will all end up in the final 3DSC dataset. Also note that step 5 (featurization) is usually skipped, except if you manually install the needed python packages(see requirements.txt).

The intermediate data will be saved under `superconductors_3D/data/intermediate/MP` and the final data will be saved under `superconductors_3D/data/final/MP`.

#### The 3DSC<sub>ICSD</sub>
The 3DSC<sub>ICSD</sub> is generated in the same way as the 3DSC<sub>MP</sub>, but with the flag `-d ICSD`. However, due to licensing reasons, the full data of the ICSD is not provided in this directory. Instead, we provide 13 ICSD structures in this repository to show the general structure of the data. Also, we provide the ICSD IDs of all structures in the final big 3DSC<sub>ICSD</sub> dataset from the paper under `superconductors_3D/data/final/ICSD/3DSC_ICSD_only_IDs.csv`. Thus, it should be easy for you to recreate the dataset and check if you get the same data.

If you have access to the ICSD, your first step is to download all the cif files. Then, you need to extract the following properties from the cif file and save them in a file called `ICSD_subset.csv`: 
- `_database_code_icsd`
- `_chemical_formula_sum`
- `_cell_measurement_temperature`
- `_diffrn_ambient_temperature`
- `_chemical_name_structure_type`
- `_exptl_crystal_density_diffrn`
- `_chemical_formula_weight`
- `_cell_length_a`
- `_cell_length_b`
- `_cell_length_c`
- `_cell_angle_alpha`
- `_cell_angle_beta`
- `_cell_angle_gamma`
- `_cell_volume`
- `_cell_formula_units_z`
- `_symmetry_space_group_name_H-M`
- `_space_group_IT_number`
- `_diffrn_ambient_pressure`

Additionally, you can optionally add a column `type` which can specify each structure to be either 'experimental' or 'theoretical'. Structures marked as theoretical will simpy be excluded by the code.

It is up to you if and how you want to make use of this option. In the original 3DSC<sub>ICSD</sub> dataset, all structures that were marked in the ICSD to be theoretical were excluded. If you would like to reproduce the 3DSC<sub>ICSD</sub> from the paper but you can't find this information, you can simply include only IDs which are listed in the file `3DSC_ICSD_only_IDs.csv`. However, please note that you cannot simply match the ICSD IDs from this file with their ICSD structures since many of them have not undergone artificial doping. Also please note that the ICSD IDs differ between the API and the website and the ICSD IDs specified in this file are the one from the API.

After downloading the cifs and saving the properties in `ICSD_subset.csv`, you can simply put them into a folder structure identical to the one in `superconductors_3D/data/` and run the `make_3DSC` command, providing the path to the new data directory. Note that the output are 1035 structures encompassing 544 superconductors, even though we only provide 13 structures in this repository. This is due to the artificial doping that is performed.

## Reproducing the results from the 3DSC paper

First of all, let's reproduce the exact versions of the conda environment. The exact conda environment was generated and tested using Linux with conda 4.13.0 and git 2.38.1. For Mac and Windows, we have not been able to reproduce the exact conda environment because of several difficult to install packages (such as chemml), that are necessary for the machine learning part. However, you can still generate the 3DSC, which should work on all systems, as explained  in [Generating the 3DSC database](#generating-the-3dsc-database).

In the cloned directory, run the following commands:
1. Setup the conda environment with name 3DSC. First, check your ~/.condarc file and temporarily set
   ```sh
   channel_priority: false
   ```
   After installing the conda environment you can set this parameter back to its previous value. 
   Note: If you have a very old conda version < 4.6.0, this parameter might throw errors for you, in this case try to leave it out. 
2. Now read in the provided conda environment file to generate the correct conda environment:
   ```sh
   conda env create -f ./environment.yaml --name 3DSC
   ```
   Note: It is important that this is done once for each directory in which this repo will be installed. If you clone the repo into another local directory, do this step again, don't skip it. The conda environment will be linked to the absolute path to the cloned version of this repo.
3. Activate the conda environment:
   ```sh
   conda activate 3DSC
   ```
Now, we can run the exact steps to reproduce most of the paper: The generation of the 3DSC<sub>MP</sub>, most of the statistical plots shown in the paper and the most important machine learning results: 
```sh
python superconductors_3D/run_everything.py -d MP -n N_CPUS
```
For the 3DSC<sub>ICSD</sub>, see above.

By the way: If you want to see the github repo in the state that it was for publication, please use 
```sh
git checkout 2471dd51a298a854cb4f365ebd39e72c7cbf3634
```

## License
The 3DSC<sub>MP</sub> database is subject to the Creative Commons Attribution 4.0 License, implying that the content may be copied, distributed, transmitted, and adapted, without obtaining specific permission from the repository owner, provided proper attribution is given to the repository owner. All software in this repository is subject to the MIT license. See `LICENSE.md` for more information.

## Origin of data

We are grateful to the provider of different databases which have made the 3DSC possible:

- The superconductor data is freely accessible provided by Stanev et al.[2] under a CC BY 4.0 license. This data is a subset of the SuperCon database, which originally was accessible at [3] and has recently been moved to [6] under a CC BY 4.0 license.
- The crystal structures as input for the 3DSC<sub>MP</sub> are freely accessible and provided by the Materials Project database[4] under a CC BY 4.0 license.
- The crystal structures as input for the 3DSC<sub>ICSD</sub> are provided by the Inorganic Crystal Structure Database (ICSD)[5], which is a commercial database. Access to the ICSD is possible by buying a license.

## References
1. Sommer, T., Willa, R., Schmalian, J. et al. 3DSC - a dataset of superconductors including crystal structures. Sci Data 10, 816 (2023). https://doi.org/10.1038/s41597-023-02721-y
2. Stanev, V. et al. Machine learning modeling of superconducting critical temperature. npj Comput. Mater. 4, 29, 10.1038/s41524-018-0085-8 (2018). ArXiv: 1709.02727
3. SuperCon, http://supercon.nims.go.jp/indexen.html (2020).
4. Materials Project, https://materialsproject.org/.
5. ICSD, https://icsd.products.fiz-karlsruhe.de/.
6. SuperCon, https://doi.org/10.48505/nims.3739