function a = forward_batch(net, x, batch)
    m = size(x, 2);
    n = size(forward(net, x(:,1)), 1);
    if nargin < 3 batch = m; end
    a = zeros(n, m, 'like', x);
    % a = [];
    for i=1:batch:m
        j=min(i+batch-1, m);
        a(:,i:j) = gather(forward(net, x(:,i:j)));
        % a = [a, gather(forward(net, x(:,i:j)))];
    end
end
