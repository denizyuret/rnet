% loads net saved from caffe in hdf5 format

function w = hdfnet(fname)
f = h5info(fname);
fprintf('Loading file %s\n', f.Name);
% this returns {bias1, weight1, bias2, weight2, ...}
r = rnet_load_group(fname, f);
% we combine biases with weights and add a row of ones to the top
% of the data matrix
for i=1:numel(r)/2
    w{i} = [r{2*i-1}, r{2*i}];
    % for the next bias to work we need to add a row of ones
    % to the next data matrix, this does that:
    if i < numel(r)/2
        w{i} = [zeros(1, size(w{i}, 2)); w{i}];
        w{i}(1,1) = 1;
    end
end
end

function r = rnet_load_group(fname, f)
fprintf('Loading group %s\n', f.Name);
r = {};
for i=1:numel(f.Datasets)
    d = f.Datasets(i);
    fprintf('Load dataset %s %s\n', f.Name, d.Name);
    r{end+1} = rnet_load_dataset(fname, f, d);
end
for i=1:numel(f.Groups)
    g=f.Groups(i);
    fprintf('Load subgroup %s\n', g.Name);
    r = [r, rnet_load_group(fname, g)];
end
end

function r = rnet_load_dataset(fname, g, d)
datasetname = [ g.Name '/' d.Name ];
fprintf('Loading dataset %s\n', datasetname);
r = h5read(fname, datasetname);
if d.Name == 'w'
    r = r';                             % weights should be (out, in)
end
end
