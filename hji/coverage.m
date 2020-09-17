addpath(genpath('~/projects/helperOC'))
addpath(genpath('~/projects/ToolboxLS'))
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

%% Grid
grid_min = [-1; -1; -1; -1]*0.5 ; % Lower corner of computation domain
grid_max = [1; 1; 1; 1]*0.5;    % Upper corner of computation domain

N = [21; 21; 21; 21];         % Number of grid points per dimension
g = createGrid(grid_min, grid_max, N);
% Use "g = createGrid(grid_min, grid_max, N);" if there are no periodic
% state space dimensions

%% problem parameters

% input bounds
uMin = -3;
uMax = 3;
obj = Quad4D([0,0,0,0], uMin, uMax);


%% time vector
t0 = 0;
dt = 0.05;


tau_cov1 = t0:dt:0.2;
tau_cov2 = t0:dt:0.5;
tau_goal = t0:dt:0.2;

%% environment
R = 0.1;
[X,Y] = meshgrid(g.vs{1},g.vs{2});
Z = (X + 0.25).^2 + (Y + 0.25).^2 - R^2;
Z = reshape(Z, [g.N(1), 1, g.N(3), 1]);
circle_cov = repmat(Z, [1, g.N(2), 1, g.N(4)]);

Z = (X - 0.25).^2 + (Y - 0.25).^2 - R^2;
Z = reshape(Z, [g.N(1), 1, g.N(3), 1]);
circle_goal = repmat(Z, [1, g.N(2), 1, g.N(4)]);


%% always_[0, 0.2] in coverage circle
% positive inside the circle
target_cov1 = repmat(reshape(circle_cov, [g.N(1),g.N(2),g.N(3),g.N(4),1]), 1, 1, 1, 1, length(tau_cov1));
[data_always_cov, tau_cov1] = always(g, obj, tau_cov1, -circle_cov, -target_cov1);
% positive values are where we want to stay inside of
%%
% negative values are where you want to be.
data_always_cov = -data_always_cov;
%%
figure(2);
clf;
hold on
j = 11;
for i = 1:length(tau_cov1)
    subplot(2, 3, i)
    contourf(X, Y, reshape(data_always_cov(:,j,:,j,i), g.N(1), g.N(3)), -10:10); colorbar;
    viscircles([-0.25,-0.25], R, 'Color', 'b');
    title(tau_cov1(i));
    axis equal
end
%% eventually always_[0,0.2] in coverage circle
% negative values are where we want to reach (opposite of the always
% operator)
% positive ones, since there is nowhere that we want to reach during those
% times.
target_cov2 = -ones(g.N(1), g.N(2), g.N(3), g.N(4), length(tau_cov2));
target_cov2(:,:,:,:,1:length(tau_cov1)) = data_always_cov;
[data_eventually_cov, tau_cov2] = eventually(g, obj, tau_cov2, target_cov2(:,:,:,:,1), target_cov2);
%%
figure(3);
clf;
hold on
j = 11;
for i = 1:length(tau_cov2)
    subplot(3, 4, i)
    contourf(X, Y, reshape(data_eventually_cov(:,j,:,j,i), g.N(1), g.N(3)), -2:0.25:2); colorbar;
    viscircles([-0.25, -0.25], R, 'Color', 'b');
    title(tau_cov2(i));
    axis equal
end
