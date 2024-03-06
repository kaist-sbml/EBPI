# EBPI
* Note: This source code was developed in Linux, and has been tested with Python 3.8

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
python run_EBPI.py -m "Glycerone; Succinic acid" -o "./output"
```

* For more information:
```
python run_EBPI.py -h
```
