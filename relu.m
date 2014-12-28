classdef relu < handle
    
    properties
        y
    end
    methods
        
        function a = forw(l, a)
            a(:) = a .* (a > 0);
            l.y = a;
        end

        function d = back(l, d)
            d(:) = d .* (l.y > 0);
        end

    end % methods
end % classdef
