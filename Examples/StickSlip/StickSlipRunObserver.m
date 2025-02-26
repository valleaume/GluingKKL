%% Test performances of the fully fledged observer

addpath('utils', 'Examples/StickSlip'); 
close all; % close all previously opened figures

% ReCreate the StickSlip object.
sys = StickSlipSystemClass();
% Create the observed augmented system

%%%%%%%% CAN BE CHANGED %%%%%%%%
perturbation_amp = 0;                     % Amplitude of unmodeled perturbation, default :  no perturbation
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

% Generate a ground truth system trajectory x and a corresponding observer
% trajectory z

% Initial condition
X0 = [0; sys.v_t; 0.7; 0.1; 0]; % start from a sticking state (q = 0, v = v_t), with unexcited spring (x = 0), mu_s = 0.7, mu_d = 0.1 
Z0 = [ 0; 0; 0; 0; 0; 0];

% Time spans
tspan = [0, 76];
jspan = [0, 280];

% Specify solver options.
config = HybridSolverConfig('AbsTol', 1e-3, 'RelTol', 1e-7);

% Compute solution of the cascade system - observer
sol_test = aug_sys.solve([X0; Z0], tspan, jspan, config);

% Extract z and y from simulation result
x = sol_test.x(:, 1:aug_sys.nx); % system trajectory
z = sol_test.x(:, aug_sys.nx + 1 : aug_sys.nx + aug_sys.nz); % observer trajectory

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

pretrained_model = "ObserverModels/stick-slip-predictor.mat";
models = load(pretrained_model);

% Verify that the models were trained on the same z dynamic
assert( isequal(A, models.A), "wrong z dynamic");
assert( isequal(B, models.B), 'wrong z dynamic');

% Use the T_InvPredictor utility to easily reconstruct x from z
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

figure(4);

subplot(221)
% Figure 4.1
plot(t, x(:,3));
hold on;
plot(t, x_pred(:,3));
title("Static friction coefficient", Interpreter = 'latex');
xlabel('Time', Interpreter='latex')
legx3 = legend('$ \mu_s $', '$\hat{\mu_s}$', 'Interpreter', 'latex');
grid on 

subplot(223)
% Figure 4.2
plot(t, x(:,3) - x_pred(:,3));
title("$\mu_s$ estimation error", Interpreter = 'latex');
xlabel('Time', Interpreter='latex')
grid on 

subplot(222)
% Figure 4.3
plot(t, sol_test.x(:,4));
hold on;
plot(t, x_pred(:,4));
title("Dynamic friction coefficient", Interpreter = 'latex');
xlabel('Time', Interpreter='latex')
legend('$ \mu_d $', '$\hat{\mu_d}$', 'Interpreter', 'latex');

subplot(224)
% Figure 4.2
plot(t, x(:, 4) - x_pred(:, 4));
title("$\mu_d$ estimation error", Interpreter = 'latex');
xlabel('Time', Interpreter='latex')
grid on 

figure(5);
subplot(211)
plot(t, x(:, 5));
hold on;
plot(t, x_pred(:, 5));
title("Mode", Interpreter = 'latex');
xlabel('Time', Interpreter='latex')
legx4 = legend('$ q $', '$\hat{q}$', 'Interpreter', 'latex');
grid on 

subplot(212)
plot(t, x(:, 5) - x_pred(:, 5));
title("Mode error", Interpreter = 'latex');
xlabel('Time', Interpreter='latex')
grid on 

