classdef relu < handle
    
    properties
        y 		% output
        x		% input
        w               % parameters
        dw              % gradient of parameters
    end

    methods
        
        function l = relu(w)
            l.w = w;
            l.dw = 0 * w;
        end

        function y = forw(l, x)
            y = l.w * x;
            y = y .* (y > 0);
            l.y = y;
            l.x = x;
        end

        function dx = back(l, dy)
            dy = dy .* (l.y > 0);
            l.dw = dy * l.x';
            if nargout > 0
                dx = l.w' * dy;
            end
        end

    end % methods
end % classdef
