%% Test performances of the fully fledged observer

addpath('utils', 'Examples/BouncingBall');
close all; % close all previously opened figures

% ReCreate the BouncingBall object.
sys = BouncingBallSystemClass();
sys.mu = 2; % Additional velocity at each impact

% Create the observed augmented system

%%%%%%%% CAN BE CHANGED %%%%%%%%
perturbation_amp = 0;                     % Amplitude of unmodeled perturbation, default :  no perturbation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Define the observation function y = h(x, t) with an unmodeled sinusoidal perturbation 
h = @(x, t) (x(1) + perturbation_amp*sin(t));

% Create the associated BouncingBall object
obs_sys = ObservedHybridSystem(sys, 1, h);

% Define the AugmentedSystem
A = diag([-1, -2, -3]);
B = [1; 1; 1];
aug_sys = AugmentedSystem(obs_sys, 3, A, B);

% Generate a ground truth system trajectory x and a corresponding observer
% trajectory z

% Initial condition
X0 = [5; 2];  % system initial condition
Z0 = [0; 0; 0]; % observer initial condition

% Time spans
tspan = [0, 20];
jspan = [0, 28];

% Specify solver options.
config = HybridSolverConfig('AbsTol', 1e-3, 'RelTol', 1e-7);

% Compute solution of the cascade system - observer
sol_test = aug_sys.solve([X0; Z0], tspan, jspan, config);

x = sol_test.x(:, 1:2); % system trajectory
z = sol_test.x(:, 3:5); % observer trajectory

% Plot observer trajectory
figure(1);
clf;
plot(sol_test.t, z);
title("Observer trajectory in $z$-coordinates", 'Interpreter', 'latex');
legend_strings = "$z_" + string(1:aug_sys.nz) + "$";
legz = legend(legend_strings, 'Interpreter', 'latex');
xlabel('Time', Interpreter='latex')
grid on

%% Reconstruct the observer estimate in x-coordinates
pretrained_model = "ObserverModels/bouncing-ball-predictor.mat";
models = load(pretrained_model);

% Verify that the models were trained on the same z dynamic
assert( isequal(A, models.A), "wrong z dynamic"); 
assert( isequal(B, models.B), 'wrong z dynamic');

T_inv = T_InvPredictor(models); % learned model of the inverse of the gluing transformation
x_pred = T_inv.predict(z); % estimate \hat x of the system state x
t = sol_test.t(:); % time array of the solution

%% Plot observer result and ground truth
figure(2);
clf;
plot(x(:,1), x(:,2));
hold on;
plot(x_pred(:,1), x_pred(:,2));
title("Phase plot", 'Interpreter', 'latex');
legend("Ground Truth", "Observer estimate", 'Interpreter', 'latex');
xlabel('$x_1$', Interpreter='latex')
ylabel('$x_2$', Interpreter='latex')
grid on

figure(3);
subplot(221)
% Figure 3.1
plot(t, x(:,1));
hold on;
plot(t, x_pred(:,1));
title("Position", 'Interpreter', 'latex');
legx1 = legend("$x_1$", "$\hat{x}_1$",'Interpreter', 'latex');
xlabel('Time', Interpreter='latex')
grid on

%figure 3.2
subplot(222)
plot(t, x(:,2));
hold on;
plot(t, x_pred(:,2));
title("Velocity", 'Interpreter', 'latex');
legx2 = legend('$x_2$', '$\hat{x}_2$', 'Interpreter', 'latex');
xlabel('Time', Interpreter='latex')
grid on

%figure 3.3
subplot(223)
plot(t, x(:,1)-x_pred(:,1));
title("Position estimation error", 'Interpreter', 'latex');
xlabel('Time', Interpreter='latex')
grid on

%figure 3.4
subplot(224)
plot(t, x(:,2)-x_pred(:,2));
title("Velocity estimation error", 'Interpreter', 'latex');
xlabel('Time', Interpreter='latex')
grid on
