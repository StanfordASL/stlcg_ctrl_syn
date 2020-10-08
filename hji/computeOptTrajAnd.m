function [traj, traj_tau, values] = computeOptTrajAnd(g, Vs, dVs, tau, dynSys, iter0, extraArgs)
[X,Y] = meshgrid(g.vs{1},g.vs{3});

m = length(Vs);
Vs_mat = zeros([m, size(Vs{1})]);
clns = repmat({':'}, 1, length(size(Vs{1})));

for i = 1:length(Vs)
    Vs_mat(i, clns{:}) = Vs{i};
end

union = reshape(max(Vs_mat, [], 1), size(Vs{i}));

if nargin < 5
    extraArgs = [];
end

% Default parameters
uMode = 'min';
visualize = false;
subSamples = 4;

if isfield(extraArgs, 'uMode')
    uMode = extraArgs.uMode;
end

% Visualization
if isfield(extraArgs, 'visualize') && extraArgs.visualize
    visualize = extraArgs.visualize;

    showDims = find(extraArgs.projDim);
    hideDims = ~extraArgs.projDim;

    if isfield(extraArgs,'fig_num')
        f = figure(extraArgs.fig_num);
    else
        f = figure;
    end
end

if isfield(extraArgs, 'subSamples')
    subSamples = extraArgs.subSamples;
end

clns = repmat({':'}, 1, g.dim);

if any(diff(tau)) < 0
    error('Time stamps must be in ascending order!')
end

% Time parameters
tauLength = length(tau);
dtSmall = (tau(2) - tau(1))/subSamples;
% maxIter = 1.25*tauLength;

% Initialize trajectory
traj = nan(g.dim, tauLength);
values = zeros(m+1,tauLength);



traj(:,1) = dynSys.x;
tEarliest = 1;

% Determine the earliest time that the current state is in the reachable set
% Binary search
upper = tauLength;
lower = tEarliest;

iter = min(find_earliest_BRS_ind(g, union, dynSys.x, upper, lower), iter0);
Vt = cell(m, 1);
dVt = cell(m, 1);
while iter <= tauLength

    % BRS at current time

    for i = 1:m
        Vt{i} = Vs{i}(clns{:}, iter);
        values(i, iter) = eval_u(g, Vt{i}, dynSys.x);
    end
    values(m+1,iter) = eval_u(g, union(clns{:}, iter), dynSys.x);

    R = 0.15;
    % Visualize BRS corresponding to current trajectory point
    if visualize
        plot(traj(showDims(1), 1:iter), traj(showDims(2), 1:iter), 'k.')
        hold on
        [g2D, data2D] = proj(g, union(clns{:},iter), hideDims, traj(hideDims,iter));
%         visSetIm(g2D, data2D', 'red', 0:0.5:5);
        contour(X, Y, data2D, 0:0.01:0.05, 'r', 'linewidth', 3);
        [g2D, data2D] = proj(g, reshape(Vs_mat(1,clns{:},iter), size(Vt{1})), hideDims, traj(hideDims,iter));
%         visSetIm(g2D, data2D', 'blue', 0:0.5:5);
        contour(X, Y, data2D, 0:0.01:0.05, 'b', 'linewidth', 1);
        [g2D, data2D] = proj(g, reshape(Vs_mat(2,clns{:},iter), size(Vt{2})), hideDims, traj(hideDims,iter));
%         visSetIm(g2D, data2D', 'green', 0:0.5:5);
        contour(X, Y, data2D, 0:0.01:0.05, 'g', 'linewidth', 1);
        viscircles([-0.2500   -0.2500], R, 'Color', 'b');
        viscircles([0.0500    0.0500], R, 'Color', 'b');
        viscircles([-0.2    0.], R/2, 'Color', 'r');
        tStr = sprintf('t = %.3f; value = %.3f', tau(iter), values(m+1,iter));
        title(tStr)
        drawnow

        if isfield(extraArgs, 'fig_filename')
           saveas(gcf,sprintf('%s%d', extraArgs.fig_filename, iter,'.png'))
        end

        hold off
    end

    if tEarliest == tauLength
        % Trajectory has entered the target
        break
    end

    % Update trajectory
    for j = 1:subSamples
        for i = 1:m
            dVt{i} = eval_u(g, dVs{i,iter}, dynSys.x);
        end
        u = optCtrl_and2(dynSys, 0, 0, values(1:2,iter), dVt, uMode);
        dynSys.updateState(u, dtSmall, dynSys.x);
    end

    % Record new point on nominal trajectory
    iter = iter + 1;
    traj(:,iter) = dynSys.x;
end

% Delete unused indices
traj(:,iter:end) = [];
traj_tau = tau(1:iter-1);
end