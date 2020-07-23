function dx = dynamics(obj, ~, x, u, d)
% function dx = dynamics(t, x, u)
%     Dynamics of the kinematic bicycle
%     x = (x, y, psi, V)
%     \dot{x}_1 = V cos(psi + beta)
%     \dot{x}_2 = V sin(psi + beta)
%     \dot{x}_3 = V/lr sin(beta)
%     \dot{x}_4 = a
%     beta = arctan(lr / (lf + lr) tan(delta))
lf = 0.5;
lr = 0.7;

if ~iscell(u)
  u = num2cell(u);
end

if ~iscell(d)
  d = num2cell(d);
end

a = u{1};
delta = u{2};
beta = atan(lr / (lr + lf) * tan(delta));

if iscell(x)
    dx = cell(length(obj.dims), 1);
    psi = x{3};
    V = x{4};

    dx{1} = V .* cos(psi + beta);
    dx{2} = V .* sin(psi + beta);
    dx{3} = V ./ lr .* sin(beta);
    dx{4} = a;
else
    dx = zeros(5,1);
    psi = x(3);
    V = x(4);
    dx(1) = V .* cos(psi + beta) + d{1};
    dx(2) = V .* sin(psi + beta) + d{2};
    dx(3) = V ./ lr .* sin(beta);
    dx(4) = a;
end
  
end