

----------------------------------------
GENERAL

Clustering and Peak finding Code.

Documentation:
Below is a general description of the various folders and contents.
Each folder also has a dedicated README.txt file to document each script.
Scripts all have headers describing functionality / use.
All scripts are in python 3.6.8 (any python 3 version should be compatible)

NOTE: Many scripts have variables/constants in the code and also near the top.
These can be adjusted to modify functionality.



Running Scripts:
Most scripts are run from the top directory with the following command

python3 -m folder.file_name
ex. python3 -m algorithms.PSG [arguments]

(top directory is where this file is)

NOTE: SOME SCRIPTS USE DIRECTORIES OUTSIDE THE REPOSITORY. The save commands can
either be commented out or the directories can be adjusted.





----------------------------------------
FOLDERS

algorithms - methods for finding optimal next measurement locations
  similarity_metrics - similarity metrics used to compare spectra

clustering - various visualizations and clustering techniques. Identifying
             specific clusters automatically.

data_loading - Loading experimental data for algorithms and clustering

GUI - Stand Alone GUI for clustering algorithms / data analysis

jupyter_notebooks - various documentation and jupyter_notebooks

logs - outputs of algorithms (running statistics) (algorithms)

peak_clustering - Clustering based on peak information

peak_error - csv file to store peak errors in peak finding algorithm
           - (peak_clustering/peak_error_plot.py)

peak_fitting - code for fitting profiles to diffraction data

testing - various tests / temporary code

utils - utility objects and methods

videos - saved experimental video (algorithms)
