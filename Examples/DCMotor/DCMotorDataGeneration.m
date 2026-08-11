%% Example: DC Motor using HybridSubsystem
% In this example, we illustrate the Gluing methodology for a DC motor with
% Coulomb friction. We create the DC motor object, build an observed system,
% generate an augmented system, then collect a labeled dataset.

addpath('utils', 'Examples/DCMotor');

% Create a DC Motor object.
sys = DCMotorHybridSystemSineClass();

% Define the observation function y = h(x, t)
h = @(x, t) [x(2), x(3), x(5), x(6)]; % observe, angular velocity, rotor angle and input
%h = @(x, t) [x(2), x(1)]; % observe angular velocity
ny = 4;

% Create the observed system
obs_sys = ObservedHybridSystem(sys, ny, h);

% Define the z dynamic: z' = Az + Bh(x)
eps = 2*pi/50;
w_1 = 2*pi;
w_2 = 13*pi;
w_3 = 5*pi;
w_4 = pi;
w_5 = pi/3;
A = diag([-eps, -0.3*eps, -w_3, -w_2, -w_1, -w_4, -w_5]);

%A = kron(eye(4), A); % take a block diagonal of A
B = ones(7, 1);
%B(2) = 0;
%B(4) = 0;
%% DCMotorDataGeneration.m
% Generate datasets for a DC motor with Coulomb friction using the
% "Gluing" methodology and the HybridSubsystem framework.
%
% This script performs the following high-level steps:
%  1. Create a DC motor hybrid system object.
%  2. Define an observation map y = h(x,t) for the measured signals.
%  3. Build an augmented linear dynamic (z) driven by the observed outputs.
%  4. Generate many random initial conditions and simulate the augmented
%     system long enough for the z-dynamics to reach stationarity.
%  5. Save full and PCA-reduced datasets for later predictor training.
%
% Notes / variables:
%  - `obs_sys` : observed hybrid system wrapping the true dynamics `sys`.
%  - `A`, `B`   : matrices for the z-dynamics (continuous linear dynamics).
%  - `nz`, `ny` : number of z-states per observed output and number of outputs.
%  - `aug_sys`  : AugmentedSystem that couples the observed system and z-dynamics.
%  - `generateUnlabbelledData` is used to collect (x,z) trajectories; the
%    function name contains the existing spelling used in the toolbox.

% Add utility paths used by the examples and toolbox
%B = kron(eye(4), B); % take a block diagonal of B
disp(A);
disp(B);

% Define the z dynamic :  z' = Az + Bh(x)
halfnz = 35;
nz = 2 * halfnz;
maxReal = 10;
maxImag = 4;
realParts = -maxReal * rand(halfnz, 1);
imagParts = maxImag * (rand(halfnz, 1) - 1/2) * 2;
poles = [realParts + 1i * imagParts; realParts - 1i * imagParts];
plot(real(poles), imag(poles), 'o')

A = [realParts(1), -imagParts(1); imagParts(1), realParts(1)];
for k = 2:halfnz
    A = [A zeros(2 * (k-1), 2);zeros(2, 2* (k-1)) [realParts(k), -imagParts(k);imagParts(k), realParts(k)]];
end

B = kron(eye(ny), ones(nz, 1));
A = kron(eye(ny), A);


aug_sys = AugmentedSystem(obs_sys, ny*nz, A, B);

%% Generate a labeled dataset of (x, z) pairs

bounds = [
    %-2,   2;   % current              
    %-50,  50;   % angular velocity
    %-pi,  pi;   % angular position
    %0,    1;    % mode
    -12,  12;   % u
    %-1,   1;    % u_dot
    2e-4, 8e-3; % F_s
    1e-4, 2e-3;  % F_d
    2*pi/20, 4*pi/3 % pulsation
];

% Random initial conditions sampled uniformly inside a specific rectangle. Take advantage of omega_limit
n_points = 40000;
seed = RandStream('mlfg6331_64');
Init_Conditions_varying = rand(seed, 4, n_points) .* (bounds(:, 2) - bounds(:, 1)) + bounds(:, 1);

Init_Conditions = zeros(9, n_points);
Init_Conditions(7:9, :) = Init_Conditions_varying(2:4, :);    
Init_Conditions(5,:) = Init_Conditions_varying(1, :); % we set the initial input u to be the first component of the random sample, and the initial input derivative u_dot to 0 for all trajectories, as we want to test the predictor on a wide range of inputs but we want to keep the same pulsation for all trajectories in order to be able to compare them

%Init_Conditions = aug_sys.generateRandomConditions(bound, 40000);

% Ensure physically realistic friction parameters: F_s > F_d
Init_Conditions = Init_Conditions(:, Init_Conditions(7, :) > Init_Conditions(8, :)); % We must have \F_s > \F_d, transform the rectangle into a triangle
%Init_Conditions(4,:) = (abs(Init_Conditions(2,:)) > sys.omega_threshold); % overwrite q in order to have phyisically plausible initial conditions
%cv_static = cvpartition(size(Init_Conditions, 2), 'HoldOut', 0.5);

%Init_Conditions(4, test(cv_static)) = 0; % Set static mode for test set, as static mode is less represented when sampling uniformly in the rectangle
%Init_Conditions(2, test(cv_static)) = 0; % Set angular velocity to 0 for static mode in the test set

% Choose a time after which the z dynamic has reached stationarity
t_take = 5/min(abs(real(eig(A)))) + 0.1;

disp('Generating dataset... This may take a few minutes.');
data = aug_sys.generateUnlabbelledData(Init_Conditions, t_take, t_take + 45, 70, 700, 0.1);
disp(size(data));

%% Save dataset

today = string(datetime("today"));
datas_filename = strcat('Data/raw-dc-motor-full-', today);
save(datas_filename, "data", "A", "B");

%% Do PCA on dataset
nx = aug_sys.nx;
z = data(aug_sys.nx+1:nx+aug_sys.nz,:);
z(:, any(isnan(z), 1)) = [];
m = mean(z, 1);
cov = (z-m)*(z-m)';
[U, S, V] = svd(cov);
[coeff, score, latent, tsquared, explained] = pca(z);

disp(explained);
[r_, index] = max(cumsum(explained)>0.95);
disp(index)
nz_embed = index;

P = U(:, 1:nz_embed)';

A_svd = P*A*P';
B_svd = P*B;

aug_sys_svd = AugmentedSystem(obs_sys, nz_embed, A_svd, B_svd);
data = aug_sys_svd.generateUnlabbelledData(Init_Conditions, t_take, t_take + 45, 400, 7000, 0.1);
disp(size(data));

%% Save dataset
A = A_svd;
B = B_svd;

today = string(datetime("today"));
datas_filename = strcat('Data/raw-dc-motor-svd-', today);
save(datas_filename, "data", "A", "B");

