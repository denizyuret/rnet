function [dz, J] = softmax_diff(z, y)
    p = softmax(gather(z));             % works better on cpu
    Y = full(sparse(double(gather(y)), 1:numel(y), 1));
    dz = (p - Y) / size(p, 2);
    py = p(sub2ind(size(p), y, 1:numel(y)));
    J = -mean(log(max(py, realmin(class(py)))));
end
