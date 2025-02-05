%% Example: StickSlip using HybridSystem
% In this example, we illustrate the Gluing methodology. We create and solve
% a StickSlip object using the adequate HybridSystem class.
% This system is the modelization of an harmonic oscillator subject to coulomb friction law.
% We then generate a dataset it using the AugmentedSystem class. Using the
% splitting methodology we train 2 neural networks and a classifier to
% learn T_inv. We then illustrate the observer.
% For a full description of this example, find the following page in the 
% HyEQ Toolbox Help: Hybrid Equations MATLAB Library > Creating and Simulating Hybrid Systems

%% Create a StickSlip object.
sys = StickSlip();


%% Create an observed system

% Define the observation function y = h(x)
h = @(x, t) (x(1));

% Create the associated BouncingBall object
obs_sys = ObservedHybridSystem(sys, 1, h);

% Define the AugmentedSystem
eps = 0.1;
w_1 = 2*pi/3;
w_2 = 2/4*pi;
A = [-eps, -w_1, 0, 0, 0, 0 ;  w_1, -eps, 0, 0, 0, 0; 0, 0, -eps, -w_2, 0, 0; 0, 0, w_2, -eps, 0, 0; 0, 0, 0, 0, -3, 0; 0, 0, 0, 0, 0, -5];
B = [1; 0; 1; 0; 1; 1];
aug_sys = AugmentedSystem(obs_sys, 6, A, B);


%% Compute a solution

% Initial condition for the initial system
x0 =  [0; sys.v_t; 0.45; 0.1; -1];

% Time spans
tspan = [0, 100];
jspan = [0, 15];

% Specify solver options.
config = HybridSolverConfig('AbsTol', 1e-3, 'RelTol', 1e-7);

% Compute solutions
sol = sys.solve(x0, tspan, jspan, config);

%% Plot the solution

figure(1)
clf
hpb = HybridPlotBuilder();
hpb.title('Ball')...
    .subplots('on')...
    .legend()...
    .plotFlows(sol)

%% Compute an augmented solution

% Initial condition for augmented system
X0 =  [0; sys.v_t; 0.45; 0.1 ; 0; 0; 0; 0; 0; 0; 0];

% Time spans
tspan = [0, 16];
jspan = [0, 20];

% Specify solver options.
config = HybridSolverConfig('AbsTol', 1e-3, 'RelTol', 1e-7);

% Compute solutions
sol_aug = aug_sys.solve(X0, tspan, jspan, config);


%% Plot Augmented solution
figure(2)
clf
hpb = HybridPlotBuilder();
hpb.title('Ball')...
    .subplots('on')...
    .legend()...
    .plotFlows(sol_aug)

%% Generate a labeled dataset of (x,z) pair

% Random initial conditions sampled uniformly inside a specific rectangle
Init_conditions = aug_sys.generateRandomConditions([-1, 1 ; -3, 2; 0.05, 1; 0.05, 1; -1, 1], 30000); % take more points to account for some points being
Init_conditions = Init_conditions(:, Init_conditions(3, :) > Init_conditions(4, :)); % We must have \mu_s > \mu_d
Init_conditions(5,:) = 1-2*(Init_conditions(2,:)<sys.v_t); % overwrite the adapted q

% Generate the dataset
t_take = 5/min(abs(real(eig(A))));
data_3 = aug_sys.generateData(Init_conditions, t_take, t_take + 20, 200, 1000);
fprintf( ' Proportion of q = 1 data points : %.2f%%', nnz(data_3(5,:)==1)/nnz(~isnan(data_3(aug_sys.state_dimension + 1,:))) );  % q=1 outlier?


%% Save dataset
today = string(datetime("today"));
datas_filename = strcat('Data/raw-stick-slip-', today);
save(datas_filename, "data_3", "A", "B")