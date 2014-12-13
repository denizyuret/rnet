function cost = softmax_cost(z, y)
    z = double(z);
    nll = bsxfun(@minus, logsumexp(z), z);
    cost = mean(nll(sub2ind(size(nll), y, 1:numel(y))));
end

% for softmax: logp = bsxfun(@minus, z, logsumexp(z));
% or just p = softmax(z)
