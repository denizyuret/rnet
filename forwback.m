function [grad, a, dz] = forwback(net, x, y)
    a = forward(net, x);
    dz = softmax_diff(a{end}, y);
    grad = backward(net, a, dz);
end

