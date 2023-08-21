function res=ndcg_k(n_k, I, y)
%% 
% I is the predicted oeder, an n by m matrix
% y is the ground truth relevance, an n by m matrix
% n is the size of query set
% m is the size of retrieval set
% n_k is the top number

% I = full(I);
% y = full(y);


% return the averaged ndcg for retrieving items for the users
[n,m]=size(I);

%% compute the ranks of the items for each user
ranks = zeros(size(I));

[~, ideal_I] = sort(y, 2, 'descend');
ideal_ranks = zeros(size(I));

% res = zeros(1, n_k);
res = 0;
cnt = 0;
for i=1:n
	ranks(i, I(i, :)) = 1:m;
	ideal_ranks(i, ideal_I(i, :)) = 1:m;

	nz = find(y(i, :)~=0);
    nnz = length(nz);
	pos = ranks(i, nz);
	ideal_pos = ideal_ranks(i, nz);
    
    
    nominator = y(i, nz)./log(pos + 1);
    denominator = y(i, nz) ./ log(ideal_pos + 1);
    if n_k > nnz
        % pad more 0
        nominator = padarray(nominator, [0, n_k - nnz], 0, 'post');
        denominator = padarray(denominator, [0, n_k - nnz], 0, 'post');
    elseif n_k <= nnz
        % truncate the tail, choose the top n_k items
        nominator = nominator(1:n_k);
        denominator = denominator(1:n_k);
    end
    
    if size(find(sum(denominator)==0), 2) ~= 0
        tmp = zeros(1,n_k);
    else
        tmp = sum(nominator)./ sum(denominator);
        cnt = cnt + 1;
        if tmp >=1
            disp(i);
        end
    end
    %tmp = full(tmp);
    %tmp = padarray(tmp, [0, length(k_vals) - size(tmp, 2)], 0, 'post');
	res = res + tmp;
end
res = res / cnt;
