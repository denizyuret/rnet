function lse = logsumexp(out)
    maxout = max(out);
    lse = maxout + log(sum(exp(bsxfun(@minus, out, maxout))));
end
