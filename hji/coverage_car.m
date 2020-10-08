addpath(genpath('~/proejcts/helperOC'))
addpath(genpath('~/proejcts/ToolboxLS'))
addpath(genpath('~/projects/stlcg_ctrl_syn'))

% 1. Run Backward Reachable Set (BRS) with a goal
%     uMode = 'min' <-- goal
%     minWith = 'none' <-- Set (not tube)
%     compTraj = false <-- no trajectory
% 2. Run BRS with goal, then optimal trajectory
%     uMode = 'min' <-- goal
%     minWith = 'none' <-- Set (not tube)
%     compTraj = true <-- compute optimal trajectory
% 3. Run Backward Reachable Tube (BRT) with a goal, then optimal trajectory
%     uMode = 'min' <-- goal
%     minWith = 'minVOverTime' <-- Tube (not set)
%     compTraj = true <-- compute optimal trajectory
% 4. Add disturbance
%     dStep1: define a dMax (dMax = [.25, .25, 0];)
%     dStep2: define a dMode (opposite of uMode)
%     dStep3: input dMax when creating your DubinsCar
%     dStep4: add dMode to schemeData
% 5. Change to an avoid BRT rather than a goal BRT
%     uMode = 'max' <-- avoid
%     dMode = 'min' <-- opposite of uMode
%     minWith = 'minVOverTime' <-- Tube (not set)
%     compTraj = false <-- no trajectory
% 6. Change to a Forward Reachable Tube (FRT)
%     add schemeData.tMode = 'forward'
%     note: now having uMode = 'max' essentially says "see how far I can
%     reach"
% 7. Add obstacles
%     add the following code:
%     obstacles = shapeCylinder(g, 3, [-1.5; 1.5; 0], 0.75);
%     HJIextraArgs.obstacles = obstacles;
% 8. Add random disturbance (white noise)
%     add the following code:
%     HJIextraArgs.addGaussianNoiseStandardDeviation = [0; 0; 0.5];
%%
% Grid
grid_min = [-5; -5; -pi; -1] ; % Lower corner of computation domain
grid_max = [5; 5; pi; 5];    % Upper corner of computation domain
pDim = [3];
N = [21; 21; 21; 21];         % Number of grid points per dimension
g = createGrid(grid_min, grid_max, N, pDim);
% Use "g = createGrid(grid_min, grid_max, N);" if there are no periodic
% state space dimensions

% problem parameters
% input bounds
% uMin = -2.0;
% uMax = 2.0;
% obj = Quad4D([0,0,0,0], uMin, uMax);
uMin = [-3, -0.344];
uMax = [3, 0.344];
dMin = [0, 0];
dMax = [0, 0];
obj = KinematicBicycle([0, 0, 0, 0], uMin, uMax, dMin, dMax);
% wMax = 0.5;
% aRange = [-3, 3];
% dMax = [0,0];
% obj =  Plane4D([0,0,0,0], wMax, aRange, dMax);
% time vector
t0 = 0;
dt = 0.05;


tau_cov1 = t0:dt:0.2;
tau_cov2 = t0:dt:0.7;
tau_goal = t0:dt:1.0;
tau_until = t0:dt:1.0;

%% environment
R = 1.5;
cov_center = [-0.25, -0.25]*10;
goal_center = [0.05, 0.05]*10;
obstacle_center = [-0.2, 0]*10;

X = g.xs{1};
Y = g.xs{2};
circle_cov = (X - cov_center(1)).^2 + (Y - cov_center(2)).^2 - R^2;
% Z = reshape(Z, [g.N(1), g.N(2), 1, 1]);
% circle_cov = repmat(Z, [1, 1, g.N(3), g.N(4)]);
% circle_cov = shapeCylinder(g, [3,4], [cov_center,0,0], R);

circle_goal = (X - goal_center(1)).^2 + (Y - goal_center(2)).^2 - R^2;
% circle_goal = reshape(Z, [g.N(1), g.N(2), 1, 1]);
% circle_goal = repmat(Z, [1, 1, g.N(3), g.N(4)]);
% circle_goal = shapeCylinder(g, [3,4], [goal_center,0,0], R);

circle_obs = (X - obstacle_center(1)).^2 + (Y - obstacle_center(2)).^2 - R^2;
% Z = reshape(Z, [g.N(1), g.N(2), 1, 1]);
% circle_obs = repmat(Z, [1, 1, g.N(3), g.N(4)]);

figure(2)
clf;
viscircles(goal_center, R, 'Color', 'g');
viscircles(cov_center, R, 'Color', 'b');
viscircles(obstacle_center, R/2, 'Color', 'r');
xlim([grid_min(1), grid_max(1)])
ylim([grid_min(2), grid_max(2)])



%% always avoid obstacle circle
% negative inside the circle
[data_avoid_obs, tau_obs] = always(g, obj, tau_until, circle_obs);
%%
figure(3);
clf;
hold on
hideDims = [0,0,1,1];
hideVals = [pi, 5];

for i = 1:length(tau_obs)-1
    subplot(4, 5, i)
    [g2D, data2D] = proj(g, data_avoid_obs(:,:,:,:,i), hideDims, hideVals);
    contourf(g2D.xs{1}, g2D.xs{2}, data2D, -5:5);
    viscircles(obstacle_center, R/2, 'Color', 'b');
    title(tau_obs(i));
    axis equal
end
%% always_[0, 0.2] in coverage circle
% positive inside the circle
target_cov1 = repmat(reshape(circle_cov, [g.N(1),g.N(2),g.N(3),g.N(4),1]), 1, 1, 1, 1, length(tau_cov1));
[data_always_cov, tau_cov1] = always(g, obj, tau_cov1, -circle_cov, -target_cov1);
% positive values are where we want to stay inside of

% negative values are where you want to be.
data_always_cov = -data_always_cov;
%%
figure(4);
clf;
hold on
hideDims = [0,0,1,1];
hideVals = [0, 5];

for i = 1:length(tau_cov1)
    subplot(2, 3, i)
    [g2D, data2D] = proj(g, data_always_cov(:,:,:,:,i), hideDims, hideVals);
    contourf(g2D.xs{1}, g2D.xs{2}, data2D, -5:5);
    viscircles(cov_center, R, 'Color', 'b');
    title(tau_cov1(i));
    axis equal
end

%% eventually always_[0,0.2] in coverage circle
% negative values are where we want to reach (opposite of the always operator) positive ones, since there is nowhere that we want to reach during those times.
target_cov2 = -10*ones(g.N(1), g.N(2), g.N(3), g.N(4), length(tau_cov2));
target_cov2(:,:,:,:,1:length(tau_cov1)) = data_always_cov;
[data_eventually_cov, tau_cov2] = eventually(g, obj, tau_cov2, target_cov2(:,:,:,:,1), target_cov2);
%%
figure(5);
clf;
hold on
hideDims = [0,0,1,1];
hideVals = [-3*pi/4, 5];
for i = 1:length(tau_cov2)
    subplot(4, 4, i)
    [g2D, data2D] = proj(g, data_eventually_cov(:,:,:,:,i), hideDims, hideVals);
    contourf(g2D.xs{1}, g2D.xs{2}, data2D, 0:10);
    viscircles(cov_center, R, 'Color', 'b');
    title(tau_cov2(i));
    axis equal
end


%% eventually inside goal
target_goal = circle_goal;
[data_eventually_goal, tau_cov3] = eventually(g, obj, tau_goal, target_goal);
%%

figure(6);
clf;
hold on
hideDims = [0,0,1,1];
hideVals = [pi/2, 5];
for i = 1:length(tau_cov3)-1
    subplot(4, 5, i)
    [g2D, data2D] = proj(g, data_eventually_goal(:,:,:,:,i), hideDims, hideVals);
    contourf(g2D.xs{1}, g2D.xs{2}, data2D, -5:5);
    viscircles(goal_center, R, 'Color', 'b');
    viscircles(cov_center, R, 'Color', 'b');
    title(tau_cov3(i));
    axis equal
end



%% coverage until goal
% target = ones(N(1), N(2), N(3), N(4), length(tau_until));
% target(:,:,:,:,1:15) = data_eventually_goal(:,:,:,:,1:15);
target = data_eventually_goal;
obstacle = ones(N(1), N(2), N(3), N(4), length(tau_until));
obstacle(:,:,:,:,14:end) = -data_eventually_cov(:,:,:,:,1:8);

HJIextraArgs.visualize = false;
[data_until, tau_until] = until(g, obj, tau_until, target(:,:,:,:,1), obstacle, target);

%%
figure(7);
clf;
hold on
hideDims = [0,0,1,1];
hideVals = [pi/2, 5];
for i = 1:length(tau_until)
    subplot(4, 6, i)
    [g2D, data2D] = proj(g, data_until(:,:,:,:,i), hideDims, hideVals);
    contourf(g2D.xs{1}, g2D.xs{2}, data2D, 0:4:20);
    viscircles(cov_center, R, 'Color', 'b');
    viscircles(goal_center, R, 'Color', 'b');
    title(tau_until(i));
    axis equal
end

%% reach avoid with obstacle
HJIextraArgs.visualize = false;
target = data_until;
obstacle = circle_obs;
% obstacle = data_avoid_obs;
% obstacle(obstacle > 0) = 1;
[data_until_obs, tau_until_obs] = until(g, obj, tau_until, target(:,:,:,:,1), obstacle, target);

%%
figure(8);
clf;
hold on
hideDims = [0,0,1,1];
hideVals = [0, 1];
for i = 1:length(tau_until)
    subplot(4, 6, i)
    [g2D, data2D] = proj(g, data_until_obs(:,:,:,:,i), hideDims, hideVals);
    contourf(g2D.xs{1}, g2D.xs{2}, data2D, 0:0.5:10);
    viscircles(cov_center, R, 'Color', 'b');
    viscircles(goal_center, R, 'Color', 'b');
    viscircles(obstacle_center, R/2, 'Color', 'r');
    title(tau_until(i));
    axis equal
end

%% Compute optimal trajectory from some initial state

%set the initial state
xinit = [-0.45, 0.5, -0.25, 0.0];
xinit = [-0.1, 0.0, 0.2, -1.0];
xinit = [-0.2, 0.5, -0.25, 0.0];
xinit = [-2.5, -4.5, pi/2, 5];
% value = eval_u(g, data_until(:,:,:,:,end), xinit)
obj.x = xinit;
uMode = 'min';
TrajextraArgs.uMode = uMode; %set if control wants to min or max
TrajextraArgs.visualize = true; %show plot
TrajextraArgs.fig_num = 9; %figure number
TrajextraArgs.projDim = [1 1 0 0]; 
% TrajextraArgs.fig_filename = 'figs/earliest/';
TrajextraArgs.fig_filename = 'figs/naive/';
dataTraj = flip(data_until_obs, 5);

iter0 = 3;
[traj, traj_tau, tEarliestList, values] = ...
  computeOptTrajTestNaive(g, dataTraj, tau_until, obj, iter0, TrajextraArgs);
% [traj, traj_tau, tEarliestList, values] = ...
%   computeOptTrajTest(g, dataTraj, tau_until, obj, TrajextraArgs);

%% saving

% Grid
grid_min = [-1; -2; -1; -2; 0] ; % Lower corner of computation domain
grid_max = [1; 2; 1; 2; 1.0];    % Upper corner of computation domain

N = [21; 21; 21; 21; 21];         % Number of grid points per dimension
gt = createGrid(grid_min, grid_max, N);

%%

grid = g.vs;
data = data_until;
save('stlhj/coverage_DoubleInt_test/grid.mat', 'grid');
save('stlhj/coverage_DoubleInt_test/value.mat','data');

[derivC, derivL, derivR] = computeGradients(gt, data);
save('stlhj/coverage_DoubleInt_test/deriv_value.mat', 'derivC');