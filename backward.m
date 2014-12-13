function dw = backward(w, a, dz)
    for i=numel(w):-1:1
        dw{i} = dz * a{i}';
        if i > 1
            dz = (a{i} > 0) .* (w{i}' * dz);
        end
    end
end
