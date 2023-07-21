Universal script for making GCB style output from any model. Requires no special python environment, just the standard anaconda module

Usage syntax:


`module purge`
`module add python/anaconda/2019.10/3.7`

`python createGCBstyleOutput.py {inputNumber} {modelversion} {yrFrom} {yrTo} {eraPressure}`
eg:
`python createGCBstyleOutput.py -1 TOM12_TJ_1ASA 1959 2023 True`

Here `-1` is the input number. A lookup table of input numbers is provided in:
`/gpfs/data/greenocean/GCB/GCB_universal/inputNumberReference.txt`


Can also be run as a batch submission script (from login node)
An example is given here
`sbatch < runGCB.bsub`