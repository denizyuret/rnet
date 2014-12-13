function dnet = dblnet(net)
    for i=1:numel(net)
        dnet{i} = double(net{i});
    end
end
