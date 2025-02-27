# Gluing KKL observer

Implement the Gluing KKL methodology for hybrid systems with unknown jump times, presented in https://hal.science/hal-04685195/document, and illustrate it on 2 examples : the bouncing ball and dry friction parameter estimation in friction oscillator exhibiting stick-slip.

## Requirements
Requires MATLAB 2024b or higher, the [Statistics and Machine Learning Toolbox](https://mathworks.com/products/statistics.html) package, the [Deep Learning Toolbox](https://mathworks.com/products/deep-learning.html) package and the [Hybrid Equations Toolbox](https://mathworks.com/matlabcentral/fileexchange/41372-hybrid-equations-toolbox) package.

## Content
The /utils folder contains mutliple class definition.
- ObservedHybridSystem is used to add an observation function $h$ to an object of the class hybrid system.
- AugmentedSystem is used to simulate the hybrid dynamic of the observer $z$ and encapsulate the generation and labelling of data points ($x$, $z$, labels) into a single method.
- T_InvPredctor encapsulate the inference of $x$ from $z$ using the adequate regressors and classifier.

The /ObserverModels folder contains pre-trained models used in the examples.  
The /Data folder contains data used to train those models.  
The /Examples folder contains one subfolder for each examples.

## How to run Examples
Each examples consists of
- A SystemClass file that defines the HybridSystem with : the continuous dynamic $f$ and the jump map $g$ alongside the flow set $C$ and the jump set $D$.
- A DataGeneration script that generate a dataset ($x$, $z$, labels) for this system.
- A Training scripts that import the dataset and train the after_jump/before_jump regressors and the classifier.
- A RunObserver script that illustrate the end result of the observer using those models.

Any of these scripts can be run independantly from the others thanks to pre-loaded datasets and pre-trained models. If you want to observe the effect of changing eigenvalues of the $z$ dynamic you must re-run all the scripts in the correct order : DataGeneration &rarr; Training &rarr; RunObserver while making sure that you correctly modified the loaded files at each step. 
