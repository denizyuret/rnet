function z = forward_batch(net, x, batch)
    n = size(x,2);
    if nargin < 3 batch = n; end
    z = [];
    for i=1:batch:n
        j=min(i+batch-1, n);
        z = [z, gather(forward(net, x(:,i:j)))];
    end
end
