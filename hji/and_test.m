% Deriv_phi = computeGradients(g, data_until(:,:,:,:,end));
% Deriv_psi = computeGradients(g, data_eventually_cov(:,:,:,:,end));
% 
% dV = cell(1,3);
% dV{1} = eval_u(g, Deriv_phi, xinit);
% dV{2} = eval_u(g, Deriv_psi, xinit);
% dV{3} = eval_u(g, Deriv_psi, xinit);
% 
% V = [data_until(:,:,:,:,3), 2*xinit), eval_u(g, data_eventually_cov(:,:,:,:,5), 2*xinit),  eval_u(g, data_eventually_cov(:,:,:,:,5), 2*xinit)];
% % % 
% % cvx_begin
% % variable e(2,1)
% % variable u(2,1)
% % minimize max(e)
% % [deriv_phi([2;4]), deriv_psi([2;4])]' * u <= e
% % obj.uMin <= u <= obj.uMax
% % cvx_end
% 
% uOpt = optCtrl_or(obj, 0, 0, V, dV, 'min')
clns = repmat({':'}, 1, g.dim);

V = cell(2,1);
V{1} = flip(data_until, 5);
V{2} = -flip(data_avoid_obs, 5);

dV = cell(2,length(tau_until));

for t = 1:length(tau_until)
    dV{1,t} = computeGradients(g, V{1}(clns{:},t));
    dV{2,t} = computeGradients(g, V{2}(clns{:},t));
end

%%
% xinit = [-0.1, 0.0, 0.2, -1.0];
xinit = [-0.2, 0.5, -0.25, 0.0];

obj.x = xinit;
uMode = 'min';
TrajextraArgs.uMode = uMode; %set if control wants to min or max
TrajextraArgs.visualize = true; %show plot
TrajextraArgs.fig_num = 6; %figure number
TrajextraArgs.projDim = [1 0 1 0]; 
TrajextraArgs.subSamples = 4;
TrajextraArgs.fig_filename = 'figs/and/';
%%
[traj, traj_tau, values] = computeOptTrajAnd(g, V, dV, tau_until, obj, 10, TrajextraArgs);
