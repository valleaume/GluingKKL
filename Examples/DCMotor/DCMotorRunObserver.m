%% DCMotorRunObserver.m
% Test and visualize observer reconstruction using the DC motor augmented system
%
% This script recreates the augmented observer, simulates it from a chosen
% initial condition, then attempts to reconstruct the plant state `x` from
% the observer features `z` using a trained predictor.

addpath('utils', 'Examples/DCMotor');
close all;

%% Build DC motor model and observed-system wrapper
% Recreate the DC motor object (hybrid model with the sine-input version).
sys = DCMotorHybridSystemSineClass();

% Observation map `h(x,t)`. The ordering here must match the observation
% used when datasets and predictors were created.
h = @(x, t) [x(1), x(2)]; % observe rotor angle and angular velocity
obs_sys = ObservedHybridSystem(sys, 2, h);

%% Load latest dataset to recover z-dynamics (A,B)
files = dir('Data/raw-dc-motor-*.mat');
assert(~isempty(files), 'No raw DC motor dataset found in Data/');
[~, idx] = max([files.datenum]);
loaded = load(fullfile('Data', files(idx).name));
A = loaded.A;
B = loaded.B;

disp('Loaded z-dynamics A and B:');
disp(A);
disp(B);

% Create augmented system. Use the saved A dimension to set nz for clarity.
nz_saved = size(A, 1);
aug_sys = AugmentedSystem(obs_sys, nz_saved, A, B);

%% Initial conditions for simulation
% System initial state `X0` ordering must match model convention. We set
% an input sine wave with pulsation and nonzero derivatives to stress the
% observer dynamics.
u_0 = 0;            % initial input value
u_dot_0 = 4;       % initial derivative of input
U_pulsation = 0.6*pi; % input sine pulsation (kept constant here)
X0 = [-0.; 0; 0; 0; u_0; u_dot_0; 0.0816*pi/180; 0.021*pi/180; U_pulsation];
Z0 = zeros(aug_sys.nz, 1); % observer (z) initial condition

% Time and jump spans for hybrid solver
tspan = [0, 50];
jspan = [0, 200];

% Solver tolerances
config = HybridSolverConfig('AbsTol', 1e-3, 'RelTol', 1e-7);

%% Simulate the augmented observer
sol = aug_sys.solve([X0; Z0], tspan, jspan, config);

% Extract system state `x(t)` and observer features `z(t)` from solution
x = sol.x(:, 1:aug_sys.nx);
z = sol.x(:, aug_sys.nx + 1 : aug_sys.nx + aug_sys.nz);

% Append inputs (u, u_dot) and pulsation to z for predictor input
% (predictor was trained with these appended features)
z = cat(2, z, x(:,5:6), x(:,9));

t = sol.t(:);

%% (Optional) Build reference signals for different modes and compare
% The following constructs linear mappings `T_0`, `T_1`, `T__1` that were
% used for analytical comparison in the original script. Kept as-is for
% diagnostic plotting; comments clarify their purpose.
lambda = -15;
G = [1/sys.L; 0; 0];
F_1 = [-sys.R/sys.L, -sys.Ke/sys.L, 0; sys.Kt/sys.J, 0, 0; 0, 1, 0];
H = [0, 1, 0];
M_1 = H/(F_1 - lambda * eye(3));
N_1 = M_1*G;

i_star = -1/sys.Kt*X0(8);
A__1 = [0; X0(8)/sys.J; 0];
A_1 = [0; -X0(8)/sys.J; 0];
B_1 = M_1*A_1/lambda;
B__1 = M_1*A__1/lambda;

F_0 = [-sys.R/sys.L, -sys.Ke/sys.L, 0; 0, 0, 0; 0, 1, 0];
M_0 = H/(F_0 - lambda * eye(3));
N_0 = M_0*G;

ampl = sqrt((u_dot_0/U_pulsation)^2+u_0^2); % amplitude of the input sine
w = ampl/(lambda^2+U_pulsation^2)*(lambda*cos(U_pulsation*t)-U_pulsation*sin(U_pulsation*t)); 

T_0 = M_0*x(:,1:3)' + N_0*w'; 
T_1 = M_1*x(:,1:3)' + N_1*w' + B_1; 
T__1 = M_1*x(:,1:3)' + N_1*w' + B__1; 

%% Plot observer trajectory in z-space
figure(3);
plot(t, z);
title('Observer trajectory in z-coordinates', 'Interpreter', 'latex');
legend(arrayfun(@(k) ['$z_{' num2str(k) '}$'], 1:aug_sys.nz, 'UniformOutput', false), 'Interpreter', 'latex');
xlabel('Time', 'Interpreter', 'latex');
grid on;

figure;
plot(t, T_0, 'LineWidth', 1.5);
hold on;    
plot(t, T_1, 'LineWidth', 1.5);
plot(t, T__1, '--', 'LineWidth', 1.5);
plot(t, z(:,8)', 'LineWidth', 1.5);
legend('T mode 0', 'T mode 1', 'T mode -1', '$z_5$', 'Interpreter', 'latex');
title('Observer in z space', 'Interpreter', 'latex');

%% Load trained predictor (state estimator) and reconstruct x from z
model_files = dir('ObserverModels/dc-motor-predictor-state*.mat');
assert(~isempty(model_files), 'No DC motor predictor model found in ObserverModels/');
[~, idx_model] = max([model_files.datenum]);
models = load(fullfile('ObserverModels', model_files(idx_model).name));

% Verify that the loaded model corresponds to the same z-dynamics
assert(isequal(A, models.A) && isequal(B, models.B), 'Loaded model does not match current z dynamic.');

% Normalize z using saved training statistics and predict
X_pred = predict(models.mdl_s, (z - models.mu) ./ models.sigma);

%% Plot ground truth vs observer reconstruction
figure;
plot(x(:, 1), x(:, 2), 'LineWidth', 1.5);
hold on;
plot(X_pred(:, 1), X_pred(:, 2), '--', 'LineWidth', 1.5);
legend('Ground truth', 'Observer estimate', 'Interpreter', 'latex');
title('DC Motor state reconstruction', 'Interpreter', 'latex');
xlabel('$x_1$', 'Interpreter', 'latex');
ylabel('$x_2$', 'Interpreter', 'latex');
grid on;

figure;
plot(t, x(:, 1), 'LineWidth', 1.5);
hold on;
plot(t, X_pred(:, 1), '--', 'LineWidth', 1.5);
legend('Ground truth', 'Observer estimate', 'Interpreter', 'latex');
title('DC Motor current reconstruction', 'Interpreter', 'latex');
xlabel('Time', 'Interpreter', 'latex');
ylabel('$x_1$ (current)', 'Interpreter', 'latex');
grid on;

figure;
plot(t, x(:, 2), 'LineWidth', 1.5);
hold on;
plot(t, X_pred(:, 2), '--', 'LineWidth', 1.5);
legend('Ground truth', 'Observer estimate', 'Interpreter', 'latex');
title('DC Motor angular velocity reconstruction', 'Interpreter', 'latex');
xlabel('Time', 'Interpreter', 'latex');
ylabel('$x_2$ (angular velocity)', 'Interpreter', 'latex');
grid on;

figure;
plot(t, x(:, 7), 'LineWidth', 1.5);
hold on;
plot(t, X_pred(:, 1), '--', 'LineWidth', 1.5);
hold on;
plot(t, x(:, 8), 'LineWidth', 1.5);
plot(t, X_pred(:, 2), '--', 'LineWidth', 1.5);
legend('Ground truth F_s', 'Observer estimate F_s', 'Ground truth F_d', 'Observer estimate F_d', 'Interpreter', 'latex');
title('DC Motor friction reconstruction', 'Interpreter', 'latex');
xlabel('Time', 'Interpreter', 'latex');
ylabel('Friction (N)', 'Interpreter', 'latex');
grid on;

