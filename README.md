### `createGCBstyleOutput.py` is a universal script for making GCB style output from any model. 

- Requires no special python environment, just the standard anaconda module
- It assumes that the model output can be found here: `/gpfs/data/greenocean/software/runs`
- Results are by default written to: `/gpfs/data/greenocean/GCB/GCB_universal/GCBstyleOutput/{modelversion}`
By default, it takes the following command-line variables:
    {inputNumber} -- corresponding to variables we are producing. see `/gpfs/data/greenocean/GCB/GCB_universal/inputNumberReference.txt` for lookup table
    {modelversion} 
    {yrFrom} --
    {yrTo} -- 
    {eraPressure} -- era formulation for pressure (True) or NCEP (False)

Usage syntax (from an interactive session):

`module purge`
`module add python/anaconda/2019.10/3.7`

`python createGCBstyleOutput.py {inputNumber} {modelversion} {yrFrom} {yrTo} {eraPressure}`
eg:
`python createGCBstyleOutput.py -1 TOM12_TJ_1ASA 1959 2020 True`

Can also be run as a batch submission script (from login node)
An example is given here:
`sbatch < runGCB.bsub`