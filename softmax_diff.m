function d = softmax_diff(z, y)
    Y = full(sparse(double(gather(y)), 1:numel(y), 1));
    a = softmax(z);
    d = (a - Y) / size(z, 2);
end
