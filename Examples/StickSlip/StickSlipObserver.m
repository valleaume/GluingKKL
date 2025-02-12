%% Test performances of the fully fledged observer

addpath('/utils', '/Examples/StickSlip'); 
%% Create an observed system

% Create a StickSlip object.
sys = StickSlipSystemClass();


%%%%%%%% CAN BE CHANGED %%%%%%%%
perturbation_amp = 0.00;                     % Amplitude of unmodeled perturbation, default :  no perturbation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Define the observation function y = h(x, t) with an unmodeled sinusoidal perturbation 
h = @(x, t) (x(1) + perturbation_amp*sin(t));

% Create the associated BouncingBall object
obs_sys = ObservedHybridSystem(sys, 1, h);


% load the z dynamic
data = load("Data/raw-stick-slip.mat");
A = data.A;
B = data.B;

% Define the AugmentedSystem [x, z] and the z dynamic
aug_sys = AugmentedSystem(obs_sys, 6, A, B);

% Generate a ground truth x signal and its corresponding z signal

% Initial condition
X0 = [0; sys.v_t; 0.7; 0.1; 0]; % start from a sticking state (q = 0, v = v_t), with unexcited spring (x = 0), mu_s = 0.7, mu_d = 0.1 
Z0 = [ 0; 0; 0; 0; 0; 0];

% Time spans
tspan = [0, 26];
jspan = [0, 28];

% Specify solver options.
config = HybridSolverConfig('AbsTol', 1e-3, 'RelTol', 1e-7);

% Compute solutions
sol_test = aug_sys.solve([X0; Z0], tspan, jspan, config);

% Extract z and y from simulation result
y = sol_test.x(:,1);
z = sol_test.x(:, aug_sys.nx + 1 : aug_sys.nx + aug_sys.nz);

% Plot z signal

close all; % close all previously opened figures

figure(1);
clf;
plot(sol_test.t, z);

title("Z signal");
legend_strings = "z_" + string(1:aug_sys.nz);
legend(legend_strings);

%% Reconstruct the observer result

pretrained_model = "ObserverModels/stick-slip-predictor.mat";
models = load(pretrained_model);

% Use the T_InvPredictor utility to easily reconstruct x from z
T_inv = T_InvPredictor(models);
x_pred = T_inv.predict(z);

%% Plot observer result and ground truth

figure(2);
clf;
plot(sol_test.x(:,1), sol_test.x(:,2));
hold on;
plot(x_pred(:,1), x_pred(:,2));
title("Phase plot");
legend("Ground Truth", "Observer");

figure(3);
clf;
plot(sol_test.t(:), sol_test.x(:,1));
hold on;
plot(sol_test.t(:), x_pred(:,1));
title("position");
legend("$x_1$", "$\hat{x_1}$",'Interpreter', 'latex');

figure(4);
clf;
plot(sol_test.t(:), sol_test.x(:,2));
hold on;
plot(sol_test.t(:), x_pred(:,2));
title("velocity");
legend('$ x_2 $', '$\hat{x_2}$', 'Interpreter', 'latex');

figure(5);
clf;
plot(sol_test.t(:), sol_test.x(:,3));
hold on;
plot(sol_test.t(:), x_pred(:,3));
title("Static friction coefficient");
legend('$ \mu_s $', '$\hat{\mu_s}$', 'Interpreter', 'latex');

figure(6);
clf;
plot(sol_test.t(:), sol_test.x(:,4));
hold on;
plot(sol_test.t(:), x_pred(:,4));
title("Dynamic friction coefficient");
legend('$ \mu_d $', '$\hat{\mu_d}$', 'Interpreter', 'latex');

