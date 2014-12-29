classdef layer < handle
    
    properties
        x,y		% input, output
        w,b             % weights, biases
        dw,db           % gradient of parameters
        dw2,db2         % sum of squared gradients for adagrad
    end
    
    methods
        
        function l = layer(w,b)
            if nargin > 0
                l.w = w;
                l.b = b;
                l.dw2 = 0 * w;
                l.db2 = 0 * b;
            end
        end

        function y = forw(l, x)
            y = l.fforw(bsxfun(@plus, l.w * x, l.b));
            l.x = x;
            l.y = y;
        end

        function dx = back(l, dy)
            dy = l.fback(dy);
            l.dw = dy * l.x';
            l.db = sum(dy, 2);
            if nargout > 0
                dx = l.w' * dy;
            end
        end

        function update(l, learningRate)
            epsilon = 1e-8;
            l.dw2 = l.dw2 + l.dw .* l.dw;
            l.db2 = l.db2 + l.db .* l.db;
            l.w = l.w - learningRate * l.dw ./ (epsilon + sqrt(l.dw2));
            l.b = l.b - 2 * learningRate * l.db ./ (epsilon + sqrt(l.db2)); % 2 is a caffe hack
        end

    end % methods
end % classdef
