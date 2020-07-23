function dOpt = optDstb(obj, ~, ~, deriv, dMode)
% uOpt = optDstb(obj, t, y, deriv, uMode)
%     Dynamics of the double integrator
%     \dot{x}_1 = x_2 + d
%     \dot{x}_2 = u

%% Input processing
if nargin < 5
  dMode = 'max';
end

if ~iscell(deriv)
  deriv = num2cell(deriv);
end

dOpt = cell(obj.nd, 1);

%% Optimal control
if strcmp(dMode, 'max')
    dOpt{1} = (deriv{1}>=0)*obj.dMax(1) + (deriv{1}<0)*(obj.dMin(1));
    dOpt{2} = (deriv{2}>=0)*obj.dMax(2) + (deriv{2}<0)*(obj.dMin(2));

elseif strcmp(dMode, 'min')
    dOpt{1} = (deriv{1}<=0)*obj.dMax(1) + (deriv{1}>0)*(obj.dMin(1));
    dOpt{2} = (deriv{2}<=0)*obj.dMax(2) + (deriv{2}>0)*(obj.dMin(2));
else
  error('Unknown dMode!')
end

end