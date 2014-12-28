function dw = forwback_batch(net, x, y, batch)
    m = size(x,2);
    if nargin < 5 batch = m; end
    dw = [];
    for i=1:batch:m
        j=min(i+batch-1, m);
        di = forwback(f, w, x(:,i:j), y(:,i:j));
        if isempty(dw)
            for l=1:numel(di)
                dw{l} = di{l};
            end
        else
            for l=1:numel(di)
                dw{l} = dw{l} + di{l};
            end
        end
    end
end
