function [dz, J] = softmax_diff(z, y)
    p = softmax(gather(z));             % works better on cpu
    Y = full(sparse(double(gather(y)), 1:numel(y), 1));
    dz = (p - Y) / size(p, 2);
    logp = log(p);
    cost = logp(sub2ind(size(logp), y, 1:numel(y)));
    J = -mean(cost);
end
