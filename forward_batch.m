function a = forward_batch(net, x, batch)
    n = size(x,2);
    if nargin < 3 batch = n; end
    a{1} = zeros(size(x), 'like', x);
    for i=1:numel(net)
        a{i+1} = zeros(size(net(i).w,1), n, 'like', x);
    end
    for i=1:batch:n
        j=min(i+batch-1, n);
        ai = forward(net, x(:,i:j));
        for k=1:numel(ai)
            a{k}(:,i:j) = gather(ai{k});
        end
    end
end
