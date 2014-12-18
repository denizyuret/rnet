function forwback_batch(net, x, y, batch)
    n = size(x,2);
    if nargin < 4 batch = n; end
    for i=1:batch:n
        j=min(i+batch-1, n);
        forwback(net, gpuArray(x(:,i:j)), y(:,i:j));
    end
end
