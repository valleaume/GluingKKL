%% Test observer reconstruction using the DC motor augmented system

addpath('utils', 'Examples/DCMotor');
close all;

% Recreate the DC motor object.
sys = DCMotorHybridSystemClass();

% Define the observation function y = h(x, t)
h = @(x, t) [x(2), x(3), x(5), x(6)]; % observe rotor angle, angular velocity and input
obs_sys = ObservedHybridSystem(sys, 4, h);

% Load the latest dataset to recover the z dynamic.
files = dir('Data/raw-dc-motor-*.mat');
assert(~isempty(files), 'No raw DC motor dataset found in Data/');
[~, idx] = max([files.datenum]);
loaded = load(strcat('Data/', files(idx).name));
A = loaded.A;
B = loaded.B;

disp(A);
disp(B);

% Define the augmented system.
aug_sys = AugmentedSystem(obs_sys, 8*4, A, B);

% Choose an initial condition for the DC motor state and the observer state.
X0 = [-0.; 0; 0; 0; 1; 0; 0.066; 0.005];
Z0 = zeros(aug_sys.nz, 1);

% Time spans.
tspan = [0, 30];
jspan = [0, 200];

% Solver options.
config = HybridSolverConfig('AbsTol', 1e-3, 'RelTol', 1e-7);

% Compute solution of the augmented observer system.
sol = aug_sys.solve([X0; Z0], tspan, jspan, config);

% Extract system and observer trajectories.
x = sol.x(:, 1:aug_sys.nx);
z = sol.x(:, aug_sys.nx + 1 : aug_sys.nx + aug_sys.nz);

t = sol.t(:);

figure;
plot(t, z);
title('Observer trajectory in z-coordinates', 'Interpreter', 'latex');
legend(arrayfun(@(k) ['$z_{' num2str(k) '}$'], 1:aug_sys.nz, 'UniformOutput', false), 'Interpreter', 'latex');
xlabel('Time', 'Interpreter', 'latex');
grid on;

% Load a trained predictor if available.
model_files = dir('ObserverModels/dc-motor-predictor-*.mat');
assert(~isempty(model_files), 'No DC motor predictor model found in ObserverModels/');
[~, idx_model] = max([model_files.datenum]);
models = load(strcat('ObserverModels/', model_files(idx_model).name));

assert(isequal(A, models.A) && isequal(B, models.B), 'Loaded model does not match current z dynamic.');

% Reconstruct x from z.

z = cat(2, z, x(:,5:6)); % we also input u and u_dot to the predictor
X_pred = predict(models.mdl, (z - models.mu) ./ models.sigma);

figure;
plot(x(:, 1), x(:, 2), 'LineWidth', 1.5);
hold on;
plot(X_pred(:, 1), X_pred(:, 2), '--', 'LineWidth', 1.5);
legend('Ground truth', 'Observer estimate', 'Interpreter', 'latex');
title('DC Motor state reconstruction', 'Interpreter', 'latex');
xlabel('$x_1$', 'Interpreter', 'latex');
ylabel('$x_2$', 'Interpreter', 'latex');
grid on;

figure

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
plot(t, X_pred(:, 7), '--', 'LineWidth', 1.5);
legend('Ground truth', 'Observer estimate', 'Interpreter', 'latex');
title('DC Motor static friction reconstruction', 'Interpreter', 'latex');       
xlabel('Time', 'Interpreter', 'latex');
ylabel('$x_7$ (static friction)', 'Interpreter', 'latex');
grid on;
hold on;
plot(t, x(:, 8), 'LineWidth', 1.5);
plot(t, X_pred(:, 8), '--', 'LineWidth', 1.5);
legend('Ground truth F_s', 'Observer estimate F_s', 'Ground truth F_d', 'Observer estimate F_d', 'Interpreter', 'latex');

