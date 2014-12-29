function y = forward(net, x, batch)
    ncol = size(x, 2);
    if nargin < 3 batch = ncol; end
    nrow = size(net{end}.w, 1);
    y = zeros(nrow, ncol, 'like', x);
    for i=1:batch:ncol
        j = min(i+batch-1, ncol);
        a = x(:,i:j);
        for l=1:numel(net)
            a = net{l}.forw(a);
        end
        y(:,i:j) = a;
    end
end
