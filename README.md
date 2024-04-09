# EBPI
* Note: This source code was developed in Linux, and has been tested in Ubuntu 20.04 with Python 3.8 and CUDA version 11.7

## Source 

1. Clone the repository 
```
git clone https://github.com/kaist-sbml/EBPI.git
```
2. Download four large files from Google Drive and replace them in the repository
   * arrow/model/checkpoint.pickle
   * pmc/oa_comm_use_file_list.csv
   * pmc/oa_non_comm_use_pdf.csv
   * text/model/text_classifier_model.pickle
```
https://drive.google.com/drive/folders/1yr96rPF2y6Oc96mLIE-E8hQRp6aZCt_e?usp=sharing
```
3. Create and activate a conda environment
```
conda env create -f environment.yml
conda activate EBPI
```

## Example
* Run EBPI
```
python run_EBPI.py -i "./example/input" -o "./example/output"
python run_EBPI.py -m "Glycerone;Succinic acid" -e [An email for PMC search] -o "output"
```

* For more information:
```
python run_EBPI.py -h
```
