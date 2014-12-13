function dw = forwback(w, x, y)
    [z, a] = forward(w, x);
    dz = softmax_diff(z, y);
    dw = backward(w, a, dz);
end
