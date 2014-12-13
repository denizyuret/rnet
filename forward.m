function [z, a] = forward(w, x)
    a{1} = x;
    for i=2:numel(w)
        a{i} = w{i-1} * a{i-1};
        a{i} = a{i} .* (a{i} > 0);
    end
    z = w{i} * a{i};
end
