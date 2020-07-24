# Clarification on tasks
There are so many experiments associated with this project, so its quite easy to run experiments with the
wrong code. So we make a clarification here. 

## Examples
We provide some script examples in the 'run_temp.sh', including scripts of:
1. cloded set uda
2. partial da
3. opent set da
4. SymmNets-V2 Strengthened for Closed Set UDA

We also provide the corresponding log file in the ./experiments. More specifically,
1. The results of McDalNets based on the c2i of ImageCLEF and Visda are stored in ./experiments/ckpt
2. The results of partial da and open set da are stored in ./experiments/Partial and ./experiments/Open, respectively.
3. The reuslts of Symmnets and Symmnets-SC are strored in ./experiments/SymmNets

## Configs
We provide configs for example experiments in the ./experiments/configs. 
The rule of name is:
1. 'dataset' _ 
2. 'train' _ 
3. 'source domain' 2 'target domain' 
4. _ 'task settings (closed set for default | partial | open)'
5. _cfg
6. SC (only for Symmnets-SC)

Note that the configs for different tasks settings may be different.

## Solver
The solvers for all task settints are provided in the ./solver
