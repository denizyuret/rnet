classdef soft < handle
    
    properties
        y 	% output probabilities
    end

    methods
        
        function x = forw(l, x)
            x = softmax(gather(x));
            l.y = x;
        end

        function dx = back(l, y)
            m = numel(y);  % y is a vector of correct classes
            y = full(sparse(double(gather(y)), 1:m, 1));
            dx = (l.y - y) / m;
        end

    end % methods
end % classdef
