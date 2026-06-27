%% Test observer reconstruction using the DC motor augmented system

addpath('utils', 'Examples/DCMotor');
close all;

% Recreate the DC motor object.
sys = DCMotorHybridSystemSineClass();

% Define the observation function y = h(x, t)
h = @(x, t) [x(2)]; %, x(3), x(5), x(6)]; % observe rotor angle, angular velocity and input
obs_sys = ObservedHybridSystem(sys, 4, h);

% Load the latest dataset to recover the z dynamic.
files = dir('Data/raw-dc-motor-*.mat');
assert(~isempty(files), 'No raw DC motor dataset found in Data/');
[~, idx] = max([files.datenum]);
loaded = load(strcat('Data/', files(idx).name));
A = loaded.A;
B = loaded.B;

% disp(A);
% disp(B);
% A = A(1:8, 1:8); % keep only the part of A corresponding to the omega dynamic
% B = B(1:8, 1); % keep only the part of B corresponding to the omega dynamic
% A(8,8 ) = -15;

eps = 2*pi/50;
w_1 = 2*pi;
w_2 = pi/3;
w_3 = 5*pi;
w_4 = pi;
A = diag([-eps, -0.3*eps, -w_3, -w_2, -w_1, -w_4]);
B = ones(6, 1);
disp(A);
disp(B);

% Define the augmented system.
aug_sys = AugmentedSystem(obs_sys, 6, A, B);

% Choose an initial condition for the DC motor state and the observer state.
u_0 = 0; % initial input
u_dot_0 = 4; % initial input derivative
U_pulsation = 0.6*pi; % pulsation of the input sine wave, we keep it constant for the test
X0 = [-0.; 0; 0; 0; u_0; u_dot_0; 0.0816*pi/180; 0.021*pi/180; U_pulsation]; % initial state of the system, we start with q = 1 to see a mode transition in the trajectory
Z0 = zeros(aug_sys.nz, 1);

% Time spans.
tspan = [0, 50];
jspan = [0, 200];

% Solver options.
config = HybridSolverConfig('AbsTol', 1e-3, 'RelTol', 1e-7);

% Compute solution of the augmented observer system.
sol = aug_sys.solve([X0; Z0], tspan, jspan, config);

% Extract system and observer trajectories.
x = sol.x(:, 1:aug_sys.nx);
z = sol.x(:, aug_sys.nx + 1 : aug_sys.nx + aug_sys.nz);

z = cat(2, z, x(:,5:6), x(:,9)); % we also add input u and u_dot to the predictor, and pulsation

t = sol.t(:);

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
disp(size(M_0));
disp(size(N_0));

ampl = sqrt((u_dot_0/U_pulsation)^2+u_0^2); % amplitude of the input sine wave
w = ampl/(lambda^2+U_pulsation^2)*(lambda*cos(U_pulsation*t)-U_pulsation*sin(U_pulsation*t)); 
disp(size(w));

T_0 = M_0*x(:,1:3)' + N_0*w'; 
T_1 = M_1*x(:,1:3)' + N_1*w' + B_1; 
T__1 = M_1*x(:,1:3)' + N_1*w' + B__1; 

figure(3);
plot(t, z);
title('Observer trajectory in z-coordinates', 'Interpreter', 'latex');
legend(arrayfun(@(k) ['$z_{' num2str(k) '}$'], 1:aug_sys.nz, 'UniformOutput', false), 'Interpreter', 'latex');
xlabel('Time', 'Interpreter', 'latex');
grid on;

disp(size(T_0));
disp(size(T_1));    
disp(size(T__1));    
disp(size(z));

figure
plot(t, T_0, 'LineWidth', 1.5);
hold on;    
plot(t, T_1, 'LineWidth', 1.5);
hold on;
plot(t, T__1, '--', 'LineWidth', 1.5);
hold on;
plot(t, z(:,8)', 'LineWidth', 1.5);

legend('T mode 0', 'T mode 1', 'T mode -1', '$z_5$', 'Interpreter', 'latex');
title('Observer in z space', 'Interpreter', 'latex');

% Load a trained predictor if available.
model_files = dir('ObserverModels/dc-motor-predictor-*.mat');
assert(~isempty(model_files), 'No DC motor predictor model found in ObserverModels/');
[~, idx_model] = max([model_files.datenum]);
models = load(strcat('ObserverModels/', model_files(idx_model).name));

assert(isequal(A, models.A) && isequal(B, models.B), 'Loaded model does not match current z dynamic.');

% Reconstruct x from z.
X_pred = zeros(size(x));
P = 5;
for i = 1:P
    X_pred(:, i) = predict(models.mdl{i}, z);
end

%z = cat(2, z, x(:,5:6)); % we also input u and u_dot to the predictor
%X_pred = predict(models.mdl, (z - models.mu) ./ models.sigma);

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
plot(t, X_pred(:, 4), '--', 'LineWidth', 1.5);
legend('Ground truth', 'Observer estimate', 'Interpreter', 'latex');
title('DC Motor static friction reconstruction', 'Interpreter', 'latex');       
xlabel('Time', 'Interpreter', 'latex');
ylabel('$x_4$ (static friction)', 'Interpreter', 'latex');
grid on;
hold on;
plot(t, x(:, 8), 'LineWidth', 1.5);
plot(t, X_pred(:, 5), '--', 'LineWidth', 1.5);
legend('Ground truth F_s', 'Observer estimate F_s', 'Ground truth F_d', 'Observer estimate F_d', 'Interpreter', 'latex');

