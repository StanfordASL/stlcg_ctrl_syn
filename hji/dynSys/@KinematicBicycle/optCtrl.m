function uOpt = optCtrl(obj, ~, x, deriv, uMode)
%     dims must specify dimensions of deriv

lf = 0.5;
lr = 0.7;

if nargin < 5
  uMode = 'min';
end

if ~iscell(deriv)
  deriv = num2cell(deriv);
end


convert_back = false;
if ~iscell(x)
  convert_back = true;
  x = num2cell(x);
end
psi = x{3};
uOpt = cell(obj.nu, 1); % [a, delta]
A = deriv{1}.*cos(psi) + deriv{2}.*sin(psi);
B = deriv{2}.*cos(psi) - deriv{1}.*sin(psi) + deriv{3} / lr;
alfa = atan(B ./ A);
d = atan((lr + lf)/lr * tan(alfa));
%% Optimal Control
if strcmp(uMode, 'max')
    uOpt{1} = (deriv{4}>=0) * obj.uMax(1) + (deriv{4}<0) * obj.uMin(1);
    uOpt{2} = (A >= 0) .* ((B >= 0) .* (min(d, obj.uMax(2))) + (B < 0) .* (max(d, obj.uMin(2)) )) + ...
              (A < 0) .* ((cos(atan(lr / (lr + lf) * tan(obj.uMin(2))) - alfa) >= cos(atan(lr / (lr + lf) * tan(obj.uMax(2))) - alfa)) * obj.uMin(2) + ...
                          (cos(atan(lr / (lr + lf) * tan(obj.uMin(2))) - alfa) < cos(atan(lr / (lr + lf) * tan(obj.uMax(2))) - alfa)) * obj.uMax(2));



elseif strcmp(uMode, 'min')
    uOpt{1} = (deriv{4}>=0) * obj.uMin(1) + (deriv{4}<0) * obj.uMax(1);
    uOpt{2} = (A <= 0) .* ((B <= 0) .* (min(d, obj.uMax(2))) + (B > 0) .* (max(d, obj.uMin(2)) )) + ...
              (A > 0) .* ((cos(atan(lr / (lr + lf) * tan(obj.uMin(2))) - alfa) <= cos(atan(lr / (lr + lf) * tan(obj.uMax(2))) - alfa)) * obj.uMin(2) + ...
                          (cos(atan(lr / (lr + lf) * tan(obj.uMin(2))) - alfa) > cos(atan(lr / (lr + lf) * tan(obj.uMax(2))) - alfa)) * obj.uMax(2));

else
  error('Unknown uMode!')
end
if convert_back
    uOpt = cell2mat(uOpt);
end
end