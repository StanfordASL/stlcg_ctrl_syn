addpath(genpath('~/proejcts/helperOC'))
addpath(genpath('~/proejcts/ToolboxLS'))
addpath(genpath('~/projects/stlcg_ctrl_syn'))

%%
% Grid
grid_min = [-5; -1; -pi/2; -1] ; % Lower corner of computation domain
grid_max = [12; 12; pi; 6];    % Upper corner of computation domain
pDim = [3];
N = [41; 41; 21; 21];         % Number of grid points per dimension
g = createGrid(grid_min, grid_max, N, pDim);
% Use "g = createGrid(grid_min, grid_max, N);" if there are no periodic
% state space dimensions

% problem parameters
uMin = [-3; -0.344];
uMax = [3; 0.344];
dMin = [0; 0];
dMax = [0; 0];
obj = KinematicBicycle([0, 0, 0, 0], uMin, uMax, dMin, dMax);

t0 = 0;
dt = 0.05;
hideDims = [0,0,1,1];


tau_cov1 = t0:dt:0.2;
tau_cov2 = t0:dt:1.2;
tau_goal = t0:dt:3.5;
tau_until = t0:dt:3.5;

%% environment
R2 = 1;
cov_center = [2, 5];
goal_center = [7,7];
obstacle_center = [4.5,6];
V_cov = 3.0;
V_goal = 1.0;
X = g.xs{1};
Y = g.xs{2};
V = g.xs{4};

circle_cov = max((X - cov_center(1)).^2 + (Y - cov_center(2)).^2 - R2^2, V - V_cov);
% circle_cov = -2 * (((V < V_cov) & (circle_cov < 0)) - 0.5) .* abs(circle_cov) .* abs(V - V_cov);% Z = reshape(Z, [g.N(1), g.N(2), 1, 1]);
% circle_cov = repmat(Z, [1, 1, g.N(3), g.N(4)]);
% circle_cov = shapeCylinder(g, [3,4], [cov_center,0,0], R);

circle_goal = max((X - goal_center(1)).^2 + (Y - goal_center(2)).^2 - R2^2, V - V_goal);
% circle_goal = -2 * (((V < V_goal) & (circle_goal < 0)) - 0.5) .* abs(circle_goal) .* abs(V - V_goal);% Z = reshape(Z, [g.N(1), g.N(2), 1, 1]);

% circle_goal = reshape(Z, [g.N(1), g.N(2), 1, 1]);
% circle_goal = repmat(Z, [1, 1, g.N(3), g.N(4)]);
% circle_goal = shapeCylinder(g, [3,4], [goal_center,0,0], R);

circle_obs = (X - obstacle_center(1)).^2 + (Y - obstacle_center(2)).^2 - R2^2;
% Z = reshape(Z, [g.N(1), g.N(2), 1, 1]);
% circle_obs = repmat(Z, [1, 1, g.N(3), g.N(4)]);
%%
figure(2)
clf;
hold on;
hideVals = [pi/2, 0.5];

[g2D, data2D] = proj(g, circle_cov, hideDims, hideVals);
contour(g2D.xs{1}, g2D.xs{2}, data2D, 0:0.01:.05, 'Color', 'b');

[g2D, data2D] = proj(g, circle_goal, hideDims, hideVals);
contour(g2D.xs{1}, g2D.xs{2}, data2D, 0:0.01:.05, 'Color', 'g');

[g2D, data2D] = proj(g, circle_obs, hideDims, hideVals);
contour(g2D.xs{1}, g2D.xs{2}, data2D, 0:0.01:.05, 'Color', 'r');

% viscircles(goal_center, R2, 'Color', 'g');
% viscircles(cov_center, R2, 'Color', 'b');
% viscircles(obstacle_center, R2, 'Color', 'r');
% axis equal
xlim([grid_min(1), grid_max(1)])
ylim([grid_min(2), grid_max(2)])


%% always avoid obstacle circle
% negative inside the circle
[data_avoid_obs, tau_obs] = always(g, obj, tau_until, circle_obs);
%%
figure(3);
clf;
hold on
hideVals = [pi/2, 2];

for i = 1:length(tau_obs)-1
    subplot(4, 5, i)
    [g2D, data2D] = proj(g, data_avoid_obs(:,:,:,:,i), hideDims, hideVals);
    contourf(g2D.xs{1}, g2D.xs{2}, data2D, -5:5);
    viscircles(obstacle_center, R2, 'Color', 'b');
    title(tau_obs(i));
    axis equal
    xlim([grid_min(1), grid_max(1)])
    ylim([grid_min(2), grid_max(2)])
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
hideVals = [pi/2, 2.98];

for i = 1:length(tau_cov1)
    subplot(2, 3, i)
    [g2D, data2D] = proj(g, data_always_cov(:,:,:,:,i), hideDims, hideVals);
    contourf(g2D.xs{1}, g2D.xs{2}, data2D, 0:10:500);
    viscircles(cov_center, R2, 'Color', 'b');
    title(tau_cov1(i));
    axis equal
end
%% eventually always_[0,0.2] in coverage circle
% negative values are where we want to reach (opposite of the always operator) positive ones, since there is nowhere that we want to reach during those times.
% target_cov2 = -10*ones(g.N(1), g.N(2), g.N(3), g.N(4), length(tau_cov2));
% target_cov2(:,:,:,:,1:length(tau_cov1)) = data_always_cov;
target_cov2 = data_always_cov(:,:,:,:,end);
compMethod = 'minVWithTarget';
[data_eventually_cov_0, tau_cov2_0] = eventually(g, obj, tau_cov2(1:end-length(tau_cov1)+1), compMethod, target_cov2);
data_eventually_cov = cat(5, data_always_cov(:,:,:,:,1:end-1), data_eventually_cov_0);

%%
figure(5);
clf;
hold on
hideVals = [pi/2, 2.5];
for i = 1:length(tau_cov2)
    subplot(4, 8, i)
    [g2D, data2D] = proj(g, data_eventually_cov(:,:,:,:,i), hideDims, hideVals);
    contourf(g2D.xs{1}, g2D.xs{2}, data2D, -10:10);
    viscircles(cov_center, R2, 'Color', 'b');
    title(tau_cov2(i));
    axis equal
end


%% eventually inside goal
target_goal = circle_goal;
compMethod = 'minVWithV0';
[data_eventually_goal, tau_cov3] = eventually(g, obj, tau_goal, compMethod, target_goal);
%%

figure(6);
clf;
hold on
hideVals = [0, 0.9];
for i = 1:3:length(tau_cov3)
    j = (i-1)/3 + 1;
    subplot(4, 5, j)
    [g2D, data2D] = proj(g, data_eventually_goal(:,:,:,:,i), hideDims, hideVals);
    contourf(g2D.xs{1}, g2D.xs{2}, data2D, -5:5);
    viscircles(goal_center, R2, 'Color', 'b');
    viscircles(cov_center, R2, 'Color', 'b');
    title(tau_cov3(i));
    axis equal
end



%% coverage until goal
target = -ones(N(1), N(2), N(3), N(4), length(tau_until));
target(:,:,:,:,1:(length(tau_until)-length(tau_cov2)+length(tau_cov1))) = data_eventually_goal(:,:,:,:,1:(length(tau_until)-length(tau_cov2)+length(tau_cov1)));
% target = data_eventually_goal;
obstacle = ones(N(1), N(2), N(3), N(4), length(tau_until));
obstacle(:,:,:,:,(length(tau_until)-length(tau_cov2)+1):end) = -data_eventually_cov;
HJIextraArgs.visualize = false;
compMethod = 'minVWithTarget';
[data_until, tau_until] = until(g, obj, tau_until, compMethod, target(:,:,:,:,1), obstacle, target);

%%
figure(7);
clf;
hold on
hideVals = [pi/6, 3];
for i = 1:3:length(tau_until)
    j = (i-1)/3 + 1;
    subplot(4, 6, j)
    [g2D, data2D] = proj(g, data_until(:,:,:,:,i), hideDims, hideVals);
    contourf(g2D.xs{1}, g2D.xs{2}, data2D, 0:10:200);
    viscircles(cov_center, R2, 'Color', 'b');
    viscircles(goal_center, R2, 'Color', 'b');
    title(tau_until(i));
    axis equal
end

%% reach avoid with obstacle
HJIextraArgs.visualize = false;
target = data_until;
obstacle = circle_obs;
compMethod = 'minVWithTarget';
[data_until_obs, tau_until_obs] = until(g, obj, tau_until, compMethod, target(:,:,:,:,1), obstacle, target);

%% reach avoid with obstacle 1
HJIextraArgs.visualize = false;
target = data_until;
obstacle = circle_obs;
compMethod = 'maxVWithTarget';
[data_until_obs1, tau_until_obs1] = until(g, obj, tau_until, compMethod, target(:,:,:,:,1), obstacle, target);
%%
HJIextraArgs.visualize = false;
target = max(-data_avoid_obs, data_until);
compMethod = 'maxVWithTarget';
[data_until_obs1, tau_until_obs1] = eventually(g, obj, tau_until, compMethod, target(:,:,:,:,1), target);

%%
figure(8);
clf;
hold on
hideDims = [0,0,1,1];
hideVals = [pi/2, 2];
for i = 1:3:length(tau_until)
    j = (i-1)/3 + 1;
    subplot(4, 6, j)
    [g2D, data2D] = proj(g, data_until_obs1(:,:,:,:,i), hideDims, hideVals);
    contourf(g2D.xs{1}, g2D.xs{2}, data2D, 0:5:200);
    viscircles(cov_center, R2, 'Color', 'b');
    viscircles(goal_center, R2, 'Color', 'b');
    viscircles(obstacle_center, R2, 'Color', 'r');
    title(tau_until(i));
    axis equal
end

%% Compute optimal trajectory from some initial state

%set the initial state
xinit = [-0.45, 0.5, -0.25, 0.0];
xinit = [-0.1, 0.0, 0.2, -1.0];
xinit = [-0.2, 0.5, -0.25, 0.0];
xinit = [0, 4, pi/4, 3];
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
TrajextraArgs.subSamples = 1;
iter0 = 1;
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
data = data_until_obs1;
save('stlhj/coverage_KinematicCar/data_until_obs.mat','data');
[derivC, derivL, derivR] = computeGradients(g, data);
save('stlhj/coverage_KinematicCar/deriv_data_until_obs.mat', 'derivC');

data = data_until;
save('stlhj/coverage_KinematicCar/data_until.mat','data');

data = data_eventually_goal;
save('stlhj/coverage_KinematicCar/data_eventually_goal.mat','data');

data = data_eventually_cov;
save('stlhj/coverage_KinematicCar/data_eventually_cov.mat','data');

data = data_always_cov;
save('stlhj/coverage_KinematicCar/data_always_cov.mat','data');

data = data_avoid_obs;
save('stlhj/coverage_KinematicCar/data_avoid_obs.mat','data');

grid = g.vs;
save('stlhj/coverage_KinematicCar/grid.mat', 'grid');
%%

grid = g.vs;
data = data_until;
save('stlhj/coverage_DoubleInt_test/grid.mat', 'grid');
save('stlhj/coverage_DoubleInt_test/value.mat','data');

[derivC, derivL, derivR] = computeGradients(gt, data);
save('stlhj/coverage_DoubleInt_test/deriv_value.mat', 'derivC');