

test_mat = zeros(10,10,10);
test_mat(3:7,3:7,3:7)=1;
pp = TVnorm3D_fast(test_mat);

volumeViewer(pp)


function pp = TVnorm3D_fast(x, epsilon)
% TVNORM3D_FAST 快速三维TV计算（内存优化）
%   适用于大型三维数据，使用内置diff函数提高效率

if nargin < 2
    epsilon = 1e-8;
end

[n1, n2, n3] = size(x);

% 使用内置diff函数（更快）
dh = diff(x, 1, 1);  % 沿第一维差分
dv = diff(x, 1, 2);  % 沿第二维差分
dd = diff(x, 1, 3);  % 沿第三维差分

% 为了保持尺寸一致，需要填充
dh = padarray(dh, [1, 0, 0], 'post');  % 在x方向后填充
dv = padarray(dv, [0, 1, 0], 'post');  % 在y方向后填充
dd = padarray(dd, [0, 0, 1], 'post');  % 在z方向后填充

% 计算TV范数
y = sum(sqrt(dh(:).^2 + dv(:).^2 + dd(:).^2 + epsilon));

pp = sqrt(dh.^2 + dv.^2 + dd.^2 + epsilon);

end
