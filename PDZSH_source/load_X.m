function X = load_X(data_path)
% Scripts to read binary
% codes from 'https://datasets.d2.mpi-inf.mpg.de/xian/load_X.m'

fp = fopen(data_path);
nsample = fread(fp, 1, 'int');
ndim = fread(fp, 1, 'int');
X = fread(fp, nsample * ndim, 'double');
fclose(fp);

X = reshape(X, [ndim,nsample]);
X = X';

end
