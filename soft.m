classdef soft < layer
    
    methods
        
        function l = soft(w)
            l = l@layer(w);
        end

        function y = fforw(l, x)
            y = softmax(gather(x));
        end

        function dy = fback(l, dy)
            m = numel(dy);  % dy is a vector of correct classes
            dy = full(sparse(double(gather(dy)), 1:m, 1));
            dy = (l.y - dy) / m;
        end

    end % methods
end % classdef
