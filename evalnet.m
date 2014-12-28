function [accuracy, cost] = evalnet(net, x, y)
    p = forward_batch(net, x, 1000);
    py = p(sub2ind(size(p), y, 1:numel(y)));
    cost = -mean(log(max(py, realmin(class(py)))));
    [~,h] = max(p);
    accuracy = mean(h==y);
end
