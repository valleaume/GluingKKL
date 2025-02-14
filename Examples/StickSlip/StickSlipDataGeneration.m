%% Example: StickSlip using HybridSystem
% In this example, we illustrate the Gluing methodology. We create
% a StickSlip object using the adequate HybridSystem class.
% This system is the modelization of an harmonic oscillator subject to coulomb friction law on a treadmill.
% We then generate and save a (x, z, labels) dataset using the AugmentedSystem class. 

addpath('utils', 'Examples/StickSlip');

% Create a StickSlip object.
sys = StickSlipSystemClass();


% Create an observed system

% Define the observation function y = h(x, t)
h = @(x, t) (x(1));

% Create the associated BouncingBall object
obs_sys = ObservedHybridSystem(sys, 1, h);

% Define the z dynamic :  z' = Az + Bh(x)
B = [1; 0; 1; 0; 1; 1];
eps = 0.1;
w_1 = 2*pi/3;
w_2 = 2*pi/4;
% We chose an A matrix with 2 "fast" real eigenvalues and 4 complex conjugate eigenvalues with pulsation similar to the pulsations observed in the z signal
A = [-eps, -w_1, 0, 0, 0, 0 ;  w_1, -eps, 0, 0, 0, 0; 0, 0, -eps, -w_2, 0, 0; 0, 0, w_2, -eps, 0, 0; 0, 0, 0, 0, -3, 0; 0, 0, 0, 0, 0, -5];
%  Define the AugmentedSystem with state [x, z], specify dim z = 6
aug_sys = AugmentedSystem(obs_sys, 6, A, B);


%% Generate a labeled dataset of (x,z) pair

% Random initial conditions sampled uniformly inside a specific box
Init_conditions = aug_sys.generateRandomConditions([-1, 1 ; -3, 2; 0.05, 1; 0.05, 1; -1, 1], 30000); % Take more points to account for some points being discarded
Init_conditions = Init_conditions(:, Init_conditions(3, :) > Init_conditions(4, :)); % We must have \mu_s > \mu_d, transform the rectangle into a triangle
Init_conditions(5,:) = 1-2*(Init_conditions(2,:) < sys.v_t); % overwrite q in order to have phyisically plausible initial conditions

% Choose a time after which the z dynamic is in stationnary state
t_take = 5/min(abs(real(eig(A))));  
% Generate the labeled dataset : 1000 initial conditions, 200 points per trajectories chosen between t_take and t_take + 20s
data_3 = aug_sys.generateData(Init_conditions, t_take, t_take + 20, 40, 12000);
fprintf( ' Proportion of q = 1 data points : %.2f%%', nnz(data_3(5,:)==1)/nnz(~isnan(data_3(aug_sys.state_dimension + 1,:))) );  % q=1 could be considered an outlier


%% Save dataset
today = string(datetime("today"));
datas_filename = strcat('Data/raw-stick-slip-', today);
save(datas_filename, "data_3", "A", "B")  % save labelled dataset and the corresponding z dynamic used to generate it