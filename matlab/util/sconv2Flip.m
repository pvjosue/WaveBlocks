function C = sconv2Flip(A, B, flipBX, flipBY, shape)
% C = sconv2_flip(A, B, shape)
% Adapted from C = sconv2(A, B, shape) Author: Bruno Luong <brunoluong@yahoo.com>

if nargin < 3
    shape = 'full';
end

[m, n] = size(A);
[p, q] = size(B);

[i, j, a] = find(A);
[k, l, b] = find(B);

if flipBX ~= 0
    k = p - k + 1;
end
if flipBY ~= 0
    l = q - l + 1;
end

[I, K] = ndgrid(i, k);
[J, L] = ndgrid(j, l);

if gpuDeviceCount>0
% This makes things faster when running very large reconstructions
    I = gpuArray(single(I));
    K = gpuArray(single(K));
    J = gpuArray(single(J));
    L = gpuArray(single(L));
    a = gpuArray(single(a(:)));
    b = gpuArray(single(b(:).'));
    C = a*b;
else
    C = a(:)*b(:).';
end

switch lower(shape)
    case 'full'
        C = sparse(I(:)+K(:)-1,J(:)+L(:)-1, C(:), m+p-1, n+q-1);
    case 'valid'
        mnc = max([m-max(0,p-1),n-max(0,q-1)],0);
        i = I(:)+K(:)-p;
        j = J(:)+L(:)-q;
        b = i > 0 & i <= mnc(1) & ...
            j > 0 & j <= mnc(2);
        C = sparse(i(b), j(b), C(b), mnc(1), mnc(2));
    case 'same'
        i = I(:)+K(:)-ceil((p+1)/2);
        j = J(:)+L(:)-ceil((q+1)/2);
        b = i > 0 & i <= m & ...
            j > 0 & j <= n;
        i = i(b);
        j = j(b);
        C = double(C(b));
        C = sparse(i, j, C, m, n);
end

end % sconv2