%% Test performances of the fully fledged observer

addpath('/utils', '/Examples/BoucingBall');
% ReCreate the BouncingBall object.
sys = BouncingBallSystemClass();
sys.mu = 2; % Additional velocity at each impact

% Create the observed augmented system

%%%%%%%% CAN BE CHANGED %%%%%%%%
perturbation_amp = 0.00;                     % Amplitude of unmodeled perturbation, default :  no perturbation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Define the observation function y = h(x, t) with an unmodeled sinusoidal perturbation 
h = @(x, t) (x(1) + perturbation_amp*sin(t));

% Create the associated BouncingBall object
obs_sys = ObservedHybridSystem(sys, 1, h);

% Define the AugmentedSystem
A = diag([-1, -2, -3]);
B = [1; 1; 1];
aug_sys = AugmentedSystem(obs_sys, 3, A, B);



% Generate a ground truth x signal and its corresponding z signal

% Initial condition
X0 = [5; 2];
Z0 = [0; 0; 0];

% Time spans
tspan = [0, 26];
jspan = [0, 28];

% Specify solver options.
config = HybridSolverConfig('AbsTol', 1e-3, 'RelTol', 1e-7);

% Compute solutions
sol_test = aug_sys.solve([X0; Z0], tspan, jspan, config);

y = sol_test.x(:, 1);
z = sol_test.x(:, 3:5);

% Plot z signal

close all; % close all previously opened figures

figure(1);
clf;
plot(sol_test.t, z);

title("Z signal");
legend_strings = "z_" + string(1:aug_sys.nz);
legend(legend_strings);

%% Reconstruct the observer result
pretrained_model = "ObserverModels/bouncing-ball-predictor.mat";
models = load(pretrained_model);
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


