%% Example: Bouncing Ball using HybridSystem
% In this example, we illustrate the Gluing methodology. We create and solve
% a BouncingBall suject to friction using the HybridSystem class.
% We then generate a dataset it using the AugmentedSystem class. Using the
% splitting methodology we train 2 neural networks and a classifier to
% learn T_inv. We then illustrate the observer.
% For a full description of this example, find the following page in the 
% HyEQ Toolbox Help: Hybrid Equations MATLAB Library > Creating and Simulating Hybrid Systems

%% Create a BouncingBall object.

sys = BouncingBallFriction();
sys.mu = 2; % Additional velocity at each impact

%% Create an observed system

% Define the observation function y = h(x)
h = @(x, t) (x(1));

% Create the associated BouncingBall object
obs_sys = ObservedHybridSystem(sys, 1, h);

% Define the AugmentedSystem
A = diag([-1, -2, -3]);
B = [1; 1; 1];
aug_sys = AugmentedSystem(obs_sys, 3, A, B);


%% Compute a solution

% Initial condition for the initial system
x0 =  [4; 0];

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
X0 =  [4; 1; 0; 0; 0];

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
Init_conditions = aug_sys.generateRandomConditions([0, 5; -12, 12], 400);

% Generate the dataset
data = aug_sys.generateData(Init_conditions, 7, 20, 200, 400, 0.001);

%% Save dataset
today = string(datetime("today"));
datas_filename = strcat('Data/raw-bouncing-ball-', today);
data_3 = data;   %save datas as "data_3" for compatibility purposes
save(datas_filename, "data_3", "A", "B")