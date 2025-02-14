## GluingKKL

Implement the Gluing KKL methodology for hybrid systems introduced in https://hal.science/hal-04685195/document, illustrate it on 2 examples : the bouncing ball and the stick slip phenomenon.

# Requirements
Requires MATLAB 2024b or higher, the [Statistics and Machine Learning Toolbox](https://mathworks.com/products/statistics.html) package, the [Deep Learning Toolbox](https://mathworks.com/products/deep-learning.html) package and the [Hybrid Equations Toolbox](https://mathworks.com/matlabcentral/fileexchange/41372-hybrid-equations-toolbox) package.

# Content
The /utils folder contains mutliple class definition.
- ObservedHybridSystem is used to add an observation function h to an object of the class hybrid system.
- AugmentedSystem is used to simulate the hybrid dynamic of the observer z and encapsulate the generation and labelling of data points (x, z, labels) into a single method.
- T_InvPredctor encapsulate the inference of x from z using the adequate regressors and classifier.

The /ObserverModels folder contains pre-trained models used in the examples.
The /Data folder contains data used to train those models.
The /Examples folder contains one subfolder for each examples.
- A DataGeneration script
