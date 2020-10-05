
function uOpt = optCtrl_and2(obj, ~, ~, ~, dV, uMode)
  % specific to a DynSys
  % double integrator 2D
  % compute the optimal control for phi & psi. If such a controller doesn't exist, then it will choose the optimal control that violates both formulas the least.
  % TODO: bias one of the formulas if both are infeasible.
    m = length(dV);
    n = obj.nu;
    H = zeros(m, 2);
    for h = 1:m
        H(h,:) = dV{h}([2;4]);
    end
    if strcmp(uMode, 'min')
        cvx_begin quiet
            variable e(m,1)
            variable u(n,1)
            minimize max(e)
            H * u <= e;
            obj.uMin <= u <= obj.uMax;
                                    cvx_end0cx
        elseif strcmp(uMode, 'max')
        cvx_begin quiet
            variable e(m,1)
            variable u(n,1)
            maximize min(e)
            H * u >= e;
            obj.uMin <= u <= obj.uMax;
        cvx_end
    else
        error('Unknown uMode!')
    end
    uOpt = u;
end
