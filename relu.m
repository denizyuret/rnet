classdef relu < layer
    
    methods
        
        function l = relu(w)
            l = l@layer(w);
        end

        function x = fforw(l, x)
            x = x .* (x > 0);
        end

        function dy = fback(l, dy)
            dy = dy .* (l.y > 0);
        end

    end % methods
end % classdef
