<img src="resources/3DSC_logo_small.png" />


# The 3DSC database
This repository contains the code and data used to create the 3DSC database, the first extensive database of superconductors with their critical temperature *T*<sub>c</sub> and their three-dimensional crystal structure. We describe the  database and the algorithm to generate it in our paper TODO INSERT_LINK[1].


## Using the 3DSC database

The 3DSC<sub>MP</sub> database can be found under `superconductors_3D/data/final/MP`. The file `3DSC_MP.csv` contains the 3DSC<sub>MP</sub> in tabular form. The three most important columns are the following:

- `formula_sc`: The chemical formula of the material, which is exactly the original chemical formula of the SuperCon entry. Note that due to the normalization of chemical formulas in the matching algorithm, this chemical formula might deviate by a constant factor from the chemical formula of the primitive unit cell of the structure.
- `tc`: The critical temperature in Kelvin. Non-superconductors have a *T*<sub>c</sub> of 0.
- `cif`: The path to the cif file from the directory `superconductors_3D/superconductors_3D/` on. If the name contains `synth_doped` this means that this structure was artificially doped.

Additionally to these three basic columns of the 3DSC<sub>MP</sub> database, there are a lot of columns which were important in the matching and adaptation algorithm, which are from the original Materials Project database or which were important for the anaylsis in our paper. These columns are less important if you just want to use the 3DSC<sub>MP</sub>, but they might be interesting for everyone looking to improve the 3DSC<sub>MP</sub> or reproduce the results in our paper.

- `MAGPIE.*`: MAGPIE feature vectors of the chemical formula of this material. Missing in the github version (see note below).
- `SOAP.*`: DSOAP feature vectors of the structure. Missing in the github version (see note below).
- `.*_2`: All columns ending with `_2` are the columns from the original structure from the Materials Project or columns added in the process of cleaning the initial Materials Project database.
- `totreldiff`: The $\Delta_\mathrm{totrel}$ from the paper, a measure of the difference between the original chemical formula of the SuperCon and of the Materials Project.
- `formula_frac`: The normalization factor of the chemical formulas.
- `sc_class`: The superconductor group (either 'Other', 'Heavy_fermion', 'Chevrel', 'Oxide', 'Cuprate', 'Ferrite', 'Carbon'). Some of the entries also have 'OxideHeavy_fermion' or 'Heavy_fermionChevrel', which means that the algorithm could not uniquely attribute this material into one group.
- `weight`: The sample weight which was used for the XGB model and the calculcation of the scores. This is just the inverse of the number of crystal structures per SuperCon entry.
- `cif_before_synthetic_doping`: The path to the original cif file of the Materials Project.
- `crystal_temp_2`: The crystal temperature. Non-zero only for the 3DSC<sub>ICSD</sub>.
- `no_crystal_temp_given_2`: If the crystal temperature was not explicitly given. Always True in the 3DSC<sub>MP</sub>. In the 3DSC<sub>ICSD</sub>, this is True if no crystal temperature was given and 293K was assumed.
- `cubic`, `hexagonal`, `monoclinic`, `orthorhombic`, `tetragonal`, `triclinic`, `trigonal`, `primitive`, `base-centered`, `body-centered`, `face-centered`: The symmetry features as described in the appendix of our paper.

Note that in the github version of this dataset, we have removed the `SOAP.*` and the `MAGPIE.*` columns due to memory constraints. You can get these columns by executing the matching and adaptation algorithm as described below.


### How to cite the 3DSC
Please cite our paper TODO INSERT_LINK[1].


## Reproducing the 3DSC paper


### Prerequisites

This code was developed and tested with and for Linux. Most likely it will throw errors for other OS. To install the Python packages we used conda 4.11.0.


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
In the following we will describe the exact steps to reproduce most of the paper: The generation of the 3DSC<sub>MP</sub>, most of the statistical plots shown in the paper and the most important machine learning results. If you want to automatically perform all of these steps please run
```sh
python superconductors_3D/run_everything.py -d MP -n N_CPUS
```
Please replace N_CPUS with the number of cores that you want to use in parallel, e.g. `1`. The flag `-d MP` means that we create the 3DSC using crystal structures from the Materials Project.

If you want to use the crystal structures from the ICSD you need to change this flag to `-d ICSD`. For how to deal with the 3DSC<sub>ICSD</sub>, please see the section later.


#### Generating the 3DSC dataset
To generate the 3DSC dataset, run the command
```sh
python superconductors_3D/generate_3DSC.py -d MP -n N_CPUS
```
The script `generate_3DSC.py` automatically runs through all stages of the matching and adaptation algorithm described in the paper: In step 0-2, the cif files, the Materials Project database and the SuperCon database are cleaned. In step 3, the SuperCon entries and the crystal structures are matched based on their chemical composition. In step 4, artificial doping is performed for all matches where the relative chemical formula doesn't match perfectly. In step 5, the chemical composition and the three-dimensional crystal structure are featurized using the MAGPIE and the Disordered SOAP algorithm. The latter is an extension of the SOAP algorithm which is described in the SI of our paper. Finally, matches are ranked and only the best matches are kept. Note that in general multiple matches can be ranked equally and will all end up in the final 3DSC dataset.
The intermediate data will be saved under `superconductors_3D/data/intermediate/MP` and the final data will be saved under `superconductors_3D/data/final/MP`. Running this command with 1 core needs about 0.5h.


#### Statistical plots
To generate the statistical plots shown in the paper please run the command
```sh
python superconductors_3D/plot_dataset_statistics.py -d MP
```
The results will be saved under `results/dataset_statistics/SC_MP_matches`. Running this command with 1 core needs few minutes.


#### Machine learning results
To reproduce the most important machine learning results shown in the paper please run the command
```sh
python superconductors_3D/train_ML_models.py
```
The results will be saved under `results/machine_learning`.

Warning: Please note that because we removed the `SOAP` and `MAGPIE` columns from the github version of the 3DSC<sub>MP</sub>, you need to first run the command above to generate the 3DSC<sub>MP</sub> before running this command. Additionally, please note that this command needs a couple of hours and several GB of disc space to run, because per default it trains 100 models (and 25 for the 3DSC<sub>ICSD</sub>) for 10 different train fractions in order to reprodude the results of the paper. If you want to modify these numbers you should be able to quickly identify them in the source code.


#### The 3DSC<sub>ICSD</sub>
Above we have focused on the 3DSC<sub>MP</sub> which is based on crystal structures from the Materials Project. We have also created another 3DSC, based on crystal structures from the ICSD, the 3DSC<sub>ICSD</sub>. However, because the crystal structures from the ICSD are not available freely, we cannot provide the source files here. Instead, we provide the 3DSC<sub>ICSD</sub> only with the ICSD IDs of the matched crystal structures. This file can be found under `superconductors_3D/data/final/ICSD/3DSC_ICSD_only_IDs.csv`. The ICSD ID can be found in the column `database_id_2` and is prepended by 'ICSD-'. Note that this is the ICSD ID from the API, not from the website, therefore you cannot find the corresponding structure by searching for the ID on the ICSD website.

If you have access to the ICSD, you can download it and run it through the matching and adaptation algorithm yourself. We do not recommend to somehow try to match the IDs to the crystal structures since many of the structures are artificially doped.


## License
The 3DSC database is subject to the Creative Commons Attribution 4.0 License, as are it's sources, the superconductor data provided by Stanev et al.[2] which is a subset of the SuperCon[3] and the Materials Project database[4]. All the software in this repository is subject to the MIT license. See `LICENSE.md` for more information.


## References
[1] TODO: Link to our paper
[2] Stanev, V. et al. Machine learning modeling of superconducting critical temperature. npj Comput. Mater. 4, 29, 10.1038/s41524-018-0085-8 (2018). ArXiv: 1709.02727
[3] SuperCon, http://supercon.nims.go.jp/indexen.html (2020).
[4] Materials Project, https://materialsproject.org/.


