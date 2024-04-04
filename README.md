# EBPI
* Note: This source code was developed in Linux, and has been tested in Ubuntu 20.04 with Python 3.8 and CUDA version 11.7

## Source 

1. Clone the repository 
```
git clone https://github.com/kaist-sbml/EBPI.git
```

2. Create and activate a conda environment
```
conda env create -f environment.yml
conda activate EBPI
```

## Example
* Run EBPI
```
python run_EBPI.py -i "./example/input" -o "./example/output"
python run_EBPI.py -m "Glycerone; Succinic acid" -e [An email for PMC search] -o "./output"
```

* For more information:
```
python run_EBPI.py -h
```
