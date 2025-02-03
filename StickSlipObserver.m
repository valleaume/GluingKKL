%% Test performances of the fully fledged observer


%% Create an observed system

% Create a StickSlip object.
sys = StickSlip();

% Define the observation function y = h(x)
h = @(x, t) (x(1));

% Create the associated BouncingBall object
obs_sys = ObservedHybridSystem(sys, 1, h);

% Define the AugmentedSystem

data = load("Data/raw-stick-slip.mat");
A = data.A;
B = data.B;
aug_sys = AugmentedSystem(obs_sys, 6, A, B);

%% Generate a ground truth x signal and its corresponding z signal

% Initial condition
X1 = [0; sys.v_t; 0.7; 0.1; 0; 0; 0; 0; 0; 0; 0];

% Time spans
tspan = [0, 26];
jspan = [0, 28];

% Specify solver options.
config = HybridSolverConfig('AbsTol', 1e-3, 'RelTol', 1e-7);

% Compute solutions
sol_test = aug_sys.solve(X1, tspan, jspan, config);

y = sol_test.x(:,1);
z = sol_test.x(:, aug_sys.nx + 1 : aug_sys.nx + aug_sys.nz);


%% Reconstruct the observer result
pretrained_model = "ObserverModels/stick-slip-predictor.mat";
models = load(pretrained_model);
T_inv = Predictor(models);
x_pred = T_inv.predict(z);

%% Plot observer result and ground truth
figure(7);
clf;
plot(sol_test.x(:,1), sol_test.x(:,2));
hold on;
plot(x_pred(:,1), x_pred(:,2));
title("Phase plot");
legend("Ground Truth", "Observer");

figure(8);
clf;
plot(sol_test.t(:), sol_test.x(:,2));
hold on;
plot(sol_test.t(:), x_pred(:,2));
title("velocity");
legend('$ x_2 $', '$\hat{x_2}$', 'Interpreter', 'latex');

figure(9);
clf;
plot(sol_test.t(:), sol_test.x(:,1));
hold on;
plot(sol_test.t(:), x_pred(:,1));
title("position");
legend("$x_1$", "$\hat{x_1}$",'Interpreter', 'latex');


%% Noisy data

sig = 0.01;

% Define the observation function y = h(x) + noise
h_noise = @(x, t) (x(1) + sig*sin(t));

% Create the associated BouncingBall object
obs_sys_noise = ObservedHybridSystem(sys, 1, h_noise);

aug_sys_noise = AugmentedSystem(obs_sys_noise, 6, A, B);

sol_test_noise = aug_sys_noise.solve(X1, tspan, jspan, config);


z_noise = sol_test_noise.x(:,aug_sys_noise.nx+1:aug_sys_noise.nx + aug_sys_noise.nz);

%% Reconstruct the observer result

x_pred_noise = T_inv.predict(z_noise);

%% Plot observer result and ground truth
figure(4);
clf;
plot(sol_test_noise.x(:,1), sol_test_noise.x(:,2));
hold on;
plot(x_pred_noise(:,1), x_pred_noise(:,2));
title("Phase plot, with noise");
legend("Ground Truth", "Observer");

figure(5);
clf;
plot(sol_test_noise.t(:), sol_test_noise.x(:,2));
hold on;
plot(sol_test_noise.t(:), x_pred_noise(:,2));
title("velocity, with noise");
legend("$x_2$", "$\hat{x_2}$", 'Interpreter', 'latex');

figure(6);
clf;
plot(sol_test_noise.t(:), sol_test_noise.x(:,1));
hold on;
plot(sol_test_noise.t(:), x_pred_noise(:,1));
title("position, with noise");
legend("$x_1$", "$\hat{x_1}$", 'Interpreter', 'latex');