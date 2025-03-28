## Overview
This repository accompanies the manuscript **TBD** and contains all data and code used in the analysis.

**Abstract:** Recent research has highlighted the electrical properties of fungi and their potential applications in unconventional computing systems. Nearly all studies to date have focused on "spikes," i.e. transient electrical impulses analogous to neural action potentials. Here, we describe a novel electrophysiological biomarker in fungi, the spectral exponent, that can be leveraged in the development of fungal electronics. We demonstrate for the first time that electrophysiological signals recorded from fungi exhibit power law scaling, a property of electroencephalography (EEG) that is indicative of long-range temporal correlations. We confirm this finding across three species of fungi as well as an independently-collected, open-access dataset. We hypothesize that the spectral exponent is a universal biomarker of network state across eukaryotic kingdoms

## Dataset
This project uses recordings from three species of fungi:
- *Hericium erinaceus* (lion’s mane)
- *Lentinula edodes* (shiitake)
- *Pleurotus djamor* (pink oyster)

The data from these recordings is stored locally in [`data/recordings.csv`](data/recordings.csv)

This project also leverages an independently collected, open-access dataset (**licensed under Creative Commons Attribution 4.0 International**):  
Mishra, A. K., Kim, J., Baghdadi, H., & Shepherd, R. (2024). *Mycelium bioelectric native and light stimulated signal and robot control codes* [Data set]. Zenodo. [https://doi.org/10.5281/zenodo.12812074](https://doi.org/10.5281/zenodo.12812074)


## Requirements
Install required Python packages with:
   ```bash
   pip install -r requirements.txt
   ```

Dependencies include:
- numpy  
- scipy  
- pandas  
- matplotlib  
- seaborn  
- neurodsp  
- specparam  

## Usage
1. **Download the additional dataset** from [Zenodo](https://zenodo.org/records/12812074) and update the `DATASET_PATH` variable in [`code/info.py`](code/info.py) *(required for Figure 3 only)*
2. **Run the pipeline** by navigating to the base directory and executing `make all`. The scripts can also be executed sequentially in the order listed in the `Makefile`.
3. **Generated figures** will be saved in the `figures/` directory.

## License
This project is licensed under the GNU General Public License v3.0 – see the [LICENSE](LICENSE) file for details.

## Citation
If you use this code or data in your work, please cite:  
**TBD**