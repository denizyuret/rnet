function dnet = dblnet(net)
    dnet = net;
    names = fieldnames(dnet);
    for i=1:numel(dnet)
        for n=1:numel(names)
            name = names{n};
            val = double(getfield(dnet(i), name));
            dnet(i) = setfield(dnet(i), name, val);
        end
    end
end
