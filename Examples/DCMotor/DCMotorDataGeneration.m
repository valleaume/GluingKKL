%% Example: DC Motor using HybridSubsystem
% In this example, we illustrate the Gluing methodology for a DC motor with
% Coulomb friction. We create the DC motor object, build an observed system,
% generate an augmented system, then collect a labeled dataset.

addpath('utils', 'Examples/DCMotor');

% Create a DC Motor object.
sys = DCMotorHybridSystemSineClass();

% Define the observation function y = h(x, t)
h = @(x, t) [x(2), x(3), x(5), x(6)]; % observe, angular velocity, rotor angle and input

% Create the observed system
obs_sys = ObservedHybridSystem(sys, 4, h);

% Define the z dynamic: z' = Az + Bh(x)
eps = 0.8;
w_1 = 2*pi/5;
w_2 = 2*pi/6;
A = [
    -eps, -w_1, 0, 0, 0, 0, 0, 0;
     w_1, -eps, 0, 0, 0, 0, 0, 0;
     0, 0, -eps, -w_2, 0, 0, 0, 0;
     0, 0, w_2, -eps, 0, 0, 0, 0;
     0, 0, 0, 0, -3, 0, 0, 0;
     0, 0, 0, 0, 0, -4, 0, 0;
     0, 0, 0, 0, 0, 0, -5, 0;
     0, 0, 0, 0, 0, 0, 0, -6];
A = kron(eye(4), A); % take a block diagonal of A
B = ones(8, 1);
B(2) = 0;
B(4) = 0;
B = kron(eye(4), B); % take a block diagonal of B
disp(A);
disp(B);
aug_sys = AugmentedSystem(obs_sys, 8*4, A, B);

%% Generate a labeled dataset of (x, z) pairs

% Random initial conditions sampled uniformly inside a specific rectangle.
Init_conditions = aug_sys.generateRandomConditions([
    -2,   2;   % current
    -50,  50;   % angular velocity
    -pi,  pi;   % angular position
    0,    1;    % mode
    -12,  12;   % u
    -1,   1;    % u_dot
    0.002, 0.2; % F_s
    0.001, 0.1;  % F_d
    2*pi/10, 4*pi/3 % pulsation
], 40000);

% Ensure physically realistic friction parameters: F_s > F_d
Init_conditions = Init_conditions(:, Init_conditions(7, :) > Init_conditions(8, :)); % We must have \mu_s > \mu_d, transform the rectangle into a triangle
Init_conditions(4,:) = (abs(Init_conditions(2,:)) > sys.omega_threshold); % overwrite q in order to have phyisically plausible initial conditions
cv_static = cvpartition(size(Init_conditions, 2), 'HoldOut', 0.5);

Init_conditions(4, test(cv_static)) = 0; % Set static mode for test set, as static mode is less represented when sampling uniformly in the rectangle
Init_conditions(2, test(cv_static)) = 0; % Set angular velocity to 0 for static mode in the test set

% Choose a time after which the z dynamic has reached stationarity
t_take = 3/min(abs(real(eig(A))));

disp('Generating dataset... This may take a few minutes.');
data = aug_sys.generateUnlabbelledData(Init_conditions, t_take, t_take + 30, 200, 10000, 0.01);
disp(size(data));

%% Save dataset

today = string(datetime("today"));
datas_filename = strcat('Data/raw-dc-motor-', today);
save(datas_filename, "data", "A", "B");
