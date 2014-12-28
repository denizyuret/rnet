classdef layer < handle
    
    properties
        y		% output
        x		% input
        w               % parameters
        dw              % gradient of parameters
        dw2             % sum of squared gradients for adagrad
        eps=1e-8        % numerical stability for adagrad
        bias_hack       % using first row/col of w for bias
    end
    
    methods
        
        function l = layer(w)
            if nargin > 0
                l.w = w;
                l.dw2 = 0 * w;
                l.bias_hack = (w(1,1)==1 && all(w(1,2:end)==0));
            end
        end

        function y = forw(l, x)
            y = l.fforw(l.w * x);
            l.x = x;
            l.y = y;
        end

        function dx = back(l, dy)
            dy = l.fback(dy);
            l.dw = dy * l.x';
            if nargout > 0
                dx = l.w' * dy;
            end
        end

        function x = fforw(l, x)
        % activation function: y=fforw(w*x)
        end

        function dy = fback(l, dy)
        % derivative of the activation function: dx=w'*fback(dy)
        end

        function adagrad(l, learningRate)
            l.dw2 = l.dw2 + l.dw .* l.dw;
            dw = learningRate * l.dw ./ (l.eps + sqrt(l.dw2));

            %% hackery to match caffe with bias and local lr
            if (l.bias_hack)
                dw(1,:) = 0; % use first row to create 1's for bias hack
            end 
            dw(:,1) = dw(:,1) * 2;  % caffe doubles lr for bias
            %% end of hackery

            l.w = l.w - dw;
        end

    end % methods
end % classdef
