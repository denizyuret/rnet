classdef soft < handle
    
    properties
        y 		% output probabilities
        x		% input
        w               % parameters
        dw              % gradient of parameters
    end

    methods
        
        function l = soft(w)
            l.w = w;
            l.dw = 0 * w;
        end

        function y = forw(l, x)
            y = softmax(gather(l.w * x));
            l.x = x;
            l.y = y;
        end

        function dx = back(l, dy)
            m = numel(dy);  % dy is a vector of correct classes
            dy = full(sparse(double(gather(dy)), 1:m, 1));
            dy = (l.y - dy) / m;
            l.dw = dy * l.x';
            dx = l.w' * dy;
        end

    end % methods
end % classdef
