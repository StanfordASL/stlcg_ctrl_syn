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


%% Should we compute the trajectory?
compTraj = false;

%% Grid
grid_min = [-5; -2; -pi; 0]; % Lower corner of computation domain
grid_max = [15; 12; pi; 5];    % Upper corner of computation domain
N = [41;41;41;21];         % Number of grid points per dimension
pDim = 3;
g = createGrid(grid_min, grid_max, N, pDim);
% Use "g = createGrid(grid_min, grid_max, N);" if there are no periodic
% state space dimensions

%% target set
R = 0.5;
% data0 = shapeCylinder(grid,ignoreDims,center,radius)
% data0 = shapeCylinder(g, 3, [0; 0; 0], R);
demo_traj = load('/home/karen/projects/stlcg_ctrl_syn/hji/data/expert_traj.mat');
demo_traj = demo_traj.expert_traj;
[traj_length,~] = size(demo_traj);
figure(1);
clf;
i = 1;
data0 = shapeSphere(g, demo_traj(i,:), R);
psi = demo_traj(i, 3);
V = demo_traj(i, 4);
[gproj, dataproj] = proj(g, data0, [0,0,1,1], 'min');
gproj = processGrid(gproj);
scatter(demo_traj(i,1), demo_traj(i,2), '*k')
visSetIm(gproj, dataproj);
hold on
for i=2:traj_length
    
    data0 = min(data0, shapeSphere(g, demo_traj(i,:), R));
    [gproj, dataproj] = proj(g, data0, [0,0,1,1], 'min');
    gproj = processGrid(gproj);
    viscircles([demo_traj(i,1), demo_traj(i,2)], R, 'Color', 'b');
    scatter(demo_traj(i,1), demo_traj(i,2), '*k')
    visSetIm(gproj, dataproj);
    pause(0.05);
end
disp('Press any button to continue');

% waitforbuttonpress;
clf
[gproj, dataproj] = proj(g, data0, [0,0,1,1], 'min');
gproj = processGrid(gproj);
scatter(demo_traj(:,1), demo_traj(:,2), '*k')
hold on;
visSetIm(gproj, dataproj);



%% time vector
t0 = 0;
tMax = 0.5;
dt = 0.05;
tau = t0:dt:tMax;

%% problem parameters

% input bounds
uMax = 1;
dMax = 0.1;
% do dStep1 here

% control trying to min or max value function?
uMode = 'min';
dMode = 'max';
% do dStep2 here


%% Pack problem parameters

% Define dynamic system
obj = KinematicBicycle([0, 0, 0, 0], [-3, -0.344], [3, 0.344], [-0.5, -0.5], [0.5, 0.5]);

% Put grid and dynamic systems into schemeData
schemeData.grid = g;
schemeData.dynSys = obj;
schemeData.accuracy = 'high'; %set accuracy
schemeData.uMode = uMode;
schemeData.dMode = dMode;
%do dStep4 here

%% additive random noise
%do Step8 here
%HJIextraArgs.addGaussianNoiseStandardDeviation = [0; 0; 0.5];
% Try other noise coefficients, like:
%    [0.2; 0; 0]; % Noise on X state
%    [0.2,0,0;0,0.2,0;0,0,0.5]; % Independent noise on all states
%    [0.2;0.2;0.5]; % Coupled noise on all states
%    {zeros(size(g.xs{1})); zeros(size(g.xs{1})); (g.xs{1}+g.xs{2})/20}; % State-dependent noise

%% If you have obstacles, compute them here
R_obs = 2.5;
HJIextraArgs.obstacleFunction = shapeSphere(g, [7;7;0;2], R_obs);
% HJIextraArgs.obstacleFunction = shapeCylinder(g, [0;0;1;1], [7;7;0;0], R_obs);
i = 4;
j = 1;
obs = ones(size(g.xs{1}));
HJIextraArgs.obstacleFunction = obs .* (sqrt((g.xs{1}(:,:,i,j) - 7).^2 + (g.xs{2}(:,:,i,j) - 7).^2) - R_obs);

% visSetIm(g, obs);

%% Compute value function

%HJIextraArgs.visualize = true; %show plot
HJIextraArgs.visualize.valueSet = 1;
HJIextraArgs.visualize.initialValueSet = 1;
HJIextraArgs.visualize.figNum = 1; %set figure number
HJIextraArgs.visualize.deleteLastPlot = true; %delete previous plot as you update

% uncomment if you want to see a 2D slice
%HJIextraArgs.visualize.plotData.plotDims = [1 1 0]; %plot x, y
%HJIextraArgs.visualize.plotData.projpt = [0]; %project at theta = 0
%HJIextraArgs.visualize.viewAngle = [0,90]; % view 2D

%[data, tau, extraOuts] = ...
% HJIPDE_solve(data0, tau, schemeData, minWith, extraArgs)
figure(2);
[data, tau2, ~] = ...
  HJIPDE_solve(data0, tau, schemeData, 'zero', HJIextraArgs);

%% plotting
figure(3);
clf
hold on;
[gproj, dataproj] = proj(g, data, [0,0,1,1], 'min');
gOutproj = processGrid(gproj);
visSetIm(gproj, dataproj, 'r', 0);
visSetIm(gproj, dataproj, 'b', 1);
visSetIm(gproj, dataproj, 'g', 2);

[gproj, dataproj] = proj(g, data0, [0,0,1,1], 'min');
gOutproj = processGrid(gproj);
visSetIm(gproj, dataproj, 'k');
viscircles([7, 7], R_obs)
%% plotting 
figure(4);
clf;

hold on
for i = 1:traj_length
    [gproj, dataproj] = proj(g, data, [0,0,1,1], demo_traj(i,3:4));
    gOutproj = processGrid(gproj);
    visSetIm(gproj, dataproj);
end
    axis equal;
viscircles([7, 7], R_obs)

%% saving
grid = g.vs;
save('data/reach_goal/grid.mat', 'grid');
save('data/reach_goal/value.mat','data');

[derivC, derivL, derivR] = computeGradients(g, data);
save('data/reach_goal/deriv_value.mat', 'derivC');