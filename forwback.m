function grad = forwback(net, x, y)
    [z,a] = forward(net, x);
    dz = softmax_diff(z, y);
    grad = backward(net, a, dz);
end
