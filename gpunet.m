function gnet = gpunet(net)
    gnet = net;
    names = fieldnames(gnet);
    for i=1:numel(gnet)
        for n=1:numel(names)
            name = names{n};
            val = gpuArray(getfield(gnet(i), name));
            gnet(i) = setfield(gnet(i), name, val);
        end
    end
end
