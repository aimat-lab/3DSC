## The 3DSC database
This repository contains the code and data used to create the 3DSC database. We describe the algorithm and the database in our paper [1].

The 3DSC<sub>MP</sub> database can be found under `superconductors_3D/data/final/MP`. The file `3DSC_MP.csv` contains the 3DSC<sub>MP</sub> in tabular form. The three most important columns are the following:

- `formula_sc`: The chemical formula of the material, which is exactly the original chemical formula of the SuperCon entry. Note that due to the normalization of chemical formulas in the matching algorithm, this chemical formula might deviate by a constant factor from the chemical formula of the primitive unit cell of the structure.
- `tc`: The critical temperature in Kelvin. Non-superconductors have a *T*<sub>c</sub> of 0.
- `cif`: The path to the cif file from the directory `superconductors_3D/superconductors_3D/` on. If the name contains `synth_doped` this means that this structure was artificially doped.

Additionally to these three basic columns of the 3DSC<sub>MP</sub> database, there are a lot of columns which were important in the matching and adaptation algorithm, which are from the original Materials Project database or which were important for the anaylsis in our paper. These columns are less important if you just want to use the 3DSC<sub>MP</sub>, but they might be interesting for everyone looking to improve the 3DSC<sub>MP</sub> or reproduce the results in our paper.

- `MAGPIE.*`: MAGPIE feature vectors of the chemical formula of this material. Missing in the github version (see note below).
- `SOAP.*`: DSOAP feature vectors of the structure. Missing in the github version (see note below).
- `.*_2`: All columns ending with `_2` are the columns from the original structure from the Materials Project or columns added in the process of cleaning the initial Materials Project database.
`totreldiff`: The $\Delta_\mathrm{totrel}$ from the paper, a measure of the difference between the original chemical formula of the SuperCon and of the Materials Project.
- `formula_frac`: The normalization factor of the chemical formulas.
- `sc_class`: The superconductor group (either 'Other', 'Heavy_fermion', 'Chevrel', 'Oxide', 'Cuprate', 'Ferrite', 'Carbon'). Some of the entries also have 'OxideHeavy_fermion' or 'Heavy_fermionChevrel', which means that the algorithm could not uniquely attribute this material into one group.
- `weight`: The sample weight which was used for the XGB model and the calculcation of the scores. This is just the inverse of the number of crystal structures per SuperCon entry.
- `cif_before_synthetic_doping`: The path to the original cif file of the Materials Project.
- `crystal_temp_2`: The crystal temperature. Non-zero only for the 3DSC<sub>ICSD</sub>.
- `no_crystal_temp_given_2`: If the crystal temperature was not explicitly given. Always True in the 3DSC<sub>MP</sub>. In the 3DSC<sub>ICSD</sub>, this is True if no crystal temperature was given and 293K was assumed.
`cubic`, `hexagonal`, `monoclinic`, `orthorhombic`, `tetragonal`, `triclinic`, `trigonal`, `primitive`, `base-centered`, `body-centered`, `face-centered`: The symmetry features as described in the appendix of our paper.

Note that in the github version of this dataset, we have removed the `SOAP.*` and the `MAGPIE.*` columns due to memory constraints. You can get these columns by executing the matching and adaptation algorithm as described below.


<!-- GETTING STARTED -->
## Reproducing the paper

### Prerequisites

This code was developed and tested with and for Linux. Most likely it will throw errors for other OS. To install the Python packages we use conda 4.11.0.

### Installation


1. Download the 3DSC repository into the current directory
   ```sh
   git clone git@github.com:TimoSommer/superconductors_3D.git
   ```
2. Change into this directory
   ```sh
   cd superconductors_3D
   ```
3. Setup the conda environment with the name superconductors_3D
   ```sh
   conda env create -f ./environment.yaml --name superconductors_3D
   ```
   Note: It is important that this is done once for each directory in which this repo will be installed. If you clone the repo into another local directory, do this step again, don't skip it. The conda environment will be linked to the path to the cloned version of this repo.
4. Activate the conda environment:
   ```sh
   conda activate superconductors_3D
   ```

### Reproduction
In the following we will describe the exact steps to reproduce most of the paper: The generation of the 3DSC<sub>MP</sub>, the statistical plots shown in the paper and the most important machine learning results. For the 3DSC<sub>ICSD</sub> see the section `The 3DSC<sub>ICSD</sub>`. If you want to automatically perform all of these steps please run
```sh
python superconductors_3D/run_everything.py -d MP -n N_CPUS
```
Please replace N_CPUS with the number of cores that you want to use in parallel, e.g. `1`. The flag `-d MP` means that we create the 3DSC using crystal structures from the Materials Project. If you want to use the crystal structures from the ICSD you need to change this flag to `-d ICSD`.

#### Generating the 3DSC dataset
To generate the 3DSC dataset, run the command
```sh
python superconductors_3D/generate_3DSC.py -d MP -n N_CPUS
```
The script `generate_3DSC.py` automatically runs through all stages of the matching and adaptation algorithm described in the paper: In step 0-2, the cif files, the Materials Project database and the SuperCon database are cleaned. In step 3, the SuperCon entries and the crystal structures are matched based on their chemical composition. In step 4, artificial doping is performed for all matches where the relative chemical formula doesn't match perfectly. In step 5, the chemical composition and the three-dimensional crystal structure are featurized using the MAGPIE and the Disordered SOAP algorithm. The latter is an extension of the SOAP algorithm which is described in the SI of our paper. Finally, matches are ranked and only the best matches are kept. Note that in general multiple matches can be ranked equally and will all end up in the final 3DSC dataset.

#### Statistical plots
To generate the statistical plots shown in the paper please run the command
```sh
python superconductors_3D/plot_dataset_statistics.py -d MP
```
The plots will be saved in the directory `../results/dataset_statistics/SC_MP_matches`.

#### Machine learning results
To reproduce the most important machine learning results shown in the paper please run the command
```sh
python superconductors_3D/train_ML_models.py
```

#### The 3DSC<sub>ICSD</sub>
Above we have focused on the 3DSC<sub>MP</sub> which is based on crystal structures from the Materials Project. We have also created another 3DSC, based on crystal structures from the ICSD, the 3DSC<sub>ICSD</sub>. However, because the crystal structures from the ICSD are not available freely, we cannot provide the source files here. Instead, we provide the 3DSC<sub>ICSD</sub> only with the ICSD IDs of the matched crystal structures. This file can be found under `superconductors_3D/data/final/ICSD/3DSC_ICSD_only_IDs.csv`. The ICSD ID can be found in the column `database_id_2` and is prepended by 'ICSD-'. Note that this is the ICSD ID from the API, not from the website, therefore you cannot find the corresponding structure by searching for the ID on the ICSD website.

If you have access to the ICSD, you can download it and run it through the matching and adaptation algorithm yourself. We do not recommend to somehow try to match the IDs to the crystal structures since many of the structures are artificially doped.



<p align="right">(<a href="#top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Bibliography
[1] TODO: Link to our paper

<p align="right">(<a href="#top">back to top</a>)</p>


<!-- LICENSE -->
## License
????????     TODO     ??????????
Distributed under the MIT License. See `LICENSE.txt` for more information.
????????????
<p align="right">(<a href="#top">back to top</a>)</p>



