function [accuracy, loss] = evalnet(net, x, y)
    p = forward(net, x, 1000);
    accuracy = net{end}.accuracy(p, y);
    loss = net{end}.loss(p, y);
end
