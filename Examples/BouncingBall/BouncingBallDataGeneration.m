%% Example: Bouncing Ball 
% In this example, we illustrate the Gluing methodology. We create
% a BouncingBall subject to friction using the HybridSystem class.
% We then generate and save a (x, z, labels) dataset using the AugmentedSystem class
% in view of training a model of the inverse gluing transformation giving x
% from z.

% Create a BouncingBall object.
addpath('utils', 'Examples/BouncingBall');
sys = BouncingBallSystemClass();
sys.mu = 2; % Additional velocity at each impact

% Create an observed system

% Define the observation function y = h(x, t)
h = @(x, t) (x(1));

% Create the associated BouncingBall object
obs_sys = ObservedHybridSystem(sys, 1, h);

% Define the z dynamic :  z' = Az + Bh(x)
A = diag([-1, -2, -3]); % choose 3 real eigenvalues for the z dynamic 
B = [1; 1; 1];
aug_sys = AugmentedSystem(obs_sys, 3, A, B);

%% Generate a labeled dataset of (x,z) pair

% Random initial conditions sampled uniformly inside a specific rectangle
Init_conditions = aug_sys.generateRandomConditions([0, 5; -12, 12], 400);

% Choose a time after which the z dynamic is in stationnary state
t_take = 5/min(abs(real(eig(A))));
% Generate the dataset from 400 initial conditions, with 200 points stored per trajectory, chosen between t_take and t_take + 15s, max_dt of ODE solver : 0.001s
data = aug_sys.generateData(Init_conditions, t_take, t_take + 15, 200, 400, 0.001);

%% Save dataset
today = string(datetime("today"));
datas_filename = strcat('Data/raw-bouncing-ball-', today);
save(datas_filename, "data", "A", "B")  % save labelled dataset and the corresponding z dynamic used to generate it