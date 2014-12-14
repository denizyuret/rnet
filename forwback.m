function [dw, J] = forwback(w, x, y)
    [z, a] = forward(w, x);
    [dz, J] = softmax_diff(z, y);
    dw = backward(w, a, dz);
end
