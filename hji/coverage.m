addpath(genpath('~/repos/helperOC'))
addpath(genpath('~/repos/ToolboxLS'))
% addpath(genpath('~/projects/stlcg_ctrl_syn'))

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
grid_min = [-1; -2; -1; -2]*1 ; % Lower corner of computation domain
grid_max = [1; 2; 1; 2]*1;    % Upper corner of computation domain

N = [21; 21; 21; 21];         % Number of grid points per dimension
g = createGrid(grid_min, grid_max, N);
% Use "g = createGrid(grid_min, grid_max, N);" if there are no periodic
% state space dimensions

% problem parameters
% input bounds
uMin = -3;
uMax = 3;
obj = Quad4D([0,0,0,0], uMin, uMax);

% time vector
t0 = 0;
dt = 0.05;


tau_cov1 = t0:dt:0.2;
tau_cov2 = t0:dt:0.7;
tau_goal = t0:dt:0.7;
tau_until = t0:dt:1.0;

%% environment
R = 0.15;
cov_center = [-0.25, -0.25];
goal_center = [0.05, 0.05];
obstacle_center = [-0.2,0];

[X,Y] = meshgrid(g.vs{1},g.vs{3});
Z = (X - cov_center(1)).^2 + (Y - cov_center(2)).^2 - R^2;
Z = reshape(Z, [g.N(1), 1, g.N(3), 1]);
circle_cov = repmat(Z, [1, g.N(2), 1, g.N(4)]);

Z = (X - goal_center(1)).^2 + (Y - goal_center(2)).^2 - R^2;
Z = reshape(Z, [g.N(1), 1, g.N(3), 1]);
circle_goal = repmat(Z, [1, g.N(2), 1, g.N(4)]);

Z = (X - obstacle_center(1)).^2 + (Y - obstacle_center(2)).^2 - (R/2)^2;
Z = reshape(Z, [g.N(1), 1, g.N(3), 1]);
circle_obs = repmat(Z, [1, g.N(2), 1, g.N(4)]);

%% always avoid obstacle circle
% negative inside the circle
[data_avoid_obs, tau_obs] = always(g, obj, tau_until, circle_obs);
%%
figure(1);
clf;
hold on
j = 21;
for i = 1:length(tau_obs)-1
    subplot(4, 5, i)
    contourf(X, Y, -reshape(data_avoid_obs(:,j,:,j,i), g.N(1), g.N(3)), -10:10); colorbar;
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

figure(2);
clf;
hold on
j = 21;
for i = 1:length(tau_cov1)
    subplot(2, 3, i)
    contourf(X, Y, reshape(data_always_cov(:,j,:,j,i), g.N(1), g.N(3)), -10:10); colorbar;
    viscircles([-0.25,-0.25], R, 'Color', 'b');
    title(tau_cov1(i));
    axis equal
end

%% eventually always_[0,0.2] in coverage circle
% negative values are where we want to reach (opposite of the always operator) positive ones, since there is nowhere that we want to reach during those times.
target_cov2 = -10*ones(g.N(1), g.N(2), g.N(3), g.N(4), length(tau_cov2));
target_cov2(:,:,:,:,1:length(tau_cov1)) = data_always_cov;
[data_eventually_cov, tau_cov2] = eventually(g, obj, tau_cov2, target_cov2(:,:,:,:,1), target_cov2);
%%
figure(3);
clf;
hold on
j = 11;
for i = 1:length(tau_cov2)
    subplot(4, 4, i)
    contourf(X, Y, reshape(data_eventually_cov(:,j,:,j,i), g.N(1), g.N(3)), -2:0.25:2); colorbar;
    viscircles(cov_center, R, 'Color', 'b');
    title(tau_cov2(i));
    axis equal
end


%% eventually inside goal
target_goal = circle_goal;
[data_eventually_goal, tau_cov3] = eventually(g, obj, tau_goal, target_goal);
%%

figure(4);
clf;
hold on
j = 11;
for i = 1:length(tau_cov3)
    subplot(4, 4, i)
    contourf(X, Y, reshape(data_eventually_goal(:,j,:,j,i), g.N(1), g.N(3)), -2:0.25:2); colorbar;
    viscircles(goal_center, R, 'Color', 'b');
    viscircles(cov_center, R, 'Color', 'b');
    title(tau_cov3(i));
    axis equal
end



%% coverage until goal
target = ones(N(1), N(2), N(3), N(4), length(tau_until));
target(:,:,:,:,1:15) = data_eventually_goal;
obstacle = ones(N(1), N(2), N(3), N(4), length(tau_until));
obstacle(:,:,:,:,7:end) = -data_eventually_cov;

HJIextraArgs.visualize = false;
[data_until, tau_until] = until(g, obj, tau_until, target(:,:,:,:,1), obstacle, target);

%%
figure(5);
clf;
hold on
j = 11;
for i = 1:length(tau_until)
        subplot(4, 6, i)
    contourf(X, Y, reshape(data_until(:,j,:,j,i), g.N(1), g.N(3)), -10:10); colorbar;
    viscircles(cov_center, R, 'Color', 'b');
    viscircles(goal_center, R, 'Color', 'b');
    title(tau_until(i));
    axis equal
end



%% Compute optimal trajectory from some initial state
  
%set the initial state
xinit = [-0.45, 0.5, -0.25, 0.0];
xinit = [-0.1, 0.0, 0.2, -1.0];
xinit = [-0.2, 0.5, -0.25, 0.0];

value = eval_u(g, data_until(:,:,:,:,end), xinit)
obj.x = xinit;
uMode = 'min';
TrajextraArgs.uMode = uMode; %set if control wants to min or max
TrajextraArgs.visualize = true; %show plot
TrajextraArgs.fig_num = 6; %figure number
TrajextraArgs.projDim = [1 0 1 0]; 
TrajextraArgs.fig_filename = 'figs/naive/';
dataTraj = flip(data_until, 5);

iter0 = 10
[traj, traj_tau, tEarliestList, values] = ...
  computeOptTrajTestNaive(g, dataTraj, tau_until, obj, iter0, TrajextraArgs);
%%

figure(6);
clf;
for i = 1:length(traj(1,:))
    subplot(2,7,i)
    hold on
    [gOut, dataOut] = proj(g, dataTraj(:,:,:,:,i), [0,1,0,1], traj([2,4], i));
%     contourf(X,Y,dataOut, 0:0.1:2)
    visSetIm(gOut, dataOut, 'red', 0:0.1:5)
    viscircles(cov_center, R, 'Color', 'b');
    viscircles(goal_center, R, 'Color', 'b');
    title(tau_cov3(i));
    scatter(traj(1,1:i), traj(3,1:i), 50, 'k*')
    plot(traj(1,1:i), traj(3,1:i), 'k')
    value = eval_u(g, data_until(:,:,:,:,end), traj(:,i));
    axis equal;
end


figure(7);
clf;
hold on;
viscircles(cov_center, R, 'Color', 'b');
viscircles(goal_center, R, 'Color', 'b');
plot(traj(1,:), traj(3,:), 'k')
scatter(traj(1,:), traj(3,:), 50, 'k*')


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