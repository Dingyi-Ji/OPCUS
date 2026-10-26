function y = TVnorm3D_gpu(x, voxel_spacing)
%TVNORM3D_GPU 计算三维矩阵的各向同性 TV 范数。
%
%   y = TVnorm3D_gpu(x) 使用单位体素间距计算三维各向同性 TV：
%
%       TV(x) = sum sqrt(|D_x x|^2 + |D_y y|^2 + |D_z z|^2)
%
%   y = TVnorm3D_gpu(x, voxel_spacing) 使用指定的体素间距
%   [sx,sy,sz]。函数内部会将体素间距按最小间距归一化，因此只需要
%   保留三个方向的相对比例，避免物理单位过小导致梯度数值过大。
%
%   边界条件：
%       使用 Neumann 边界。每个方向最后一个平面上的前向差分为 0，
%       不会把三维体积的最后一层与第一层错误连接。
%
%   GPU 特性：
%       - 输入为 gpuArray 时，全部计算保留在 GPU 上；
%       - 输入为 CPU 数组时，在 CPU 上计算；
%       - 不强制转换为 double，保留输入的数据类型；
%       - 不依赖 padarray、convn 或 Image Processing Toolbox。
%
%   TwIST_GPU 中的用法：
%       phi_tv = @(x) TVnorm3D_gpu(reshape(x, grid_size));
%
%   如果需要考虑体素间距：
%       tv_spacing = [dx, dy, dz] / min([dx, dy, dz]);
%       phi_tv = @(x) TVnorm3D_gpu(reshape(x, grid_size), tv_spacing);

    if nargin < 2 || isempty(voxel_spacing)
        voxel_spacing = [1, 1, 1];
    end

    if ndims(x) ~= 3
        error('TVnorm3D_gpu:InvalidDimension', ...
            '输入 x 必须是三维矩阵。如果输入来自 TwIST，请先 reshape(x, grid_size)。');
    end

    if ~(isnumeric(x) || islogical(x))
        error('TVnorm3D_gpu:InvalidType', ...
            '输入 x 必须是数值或逻辑数组。');
    end

    if ~isfloat(x)
        x = single(x);
    end

    voxel_spacing = normalize_voxel_spacing(voxel_spacing);
    gradient_weights = 1 ./ voxel_spacing;

    [gx, gy, gz] = forward_gradient_3d(x, gradient_weights);

    gradient_magnitude = sqrt(abs(gx).^2 + abs(gy).^2 + abs(gz).^2);
    y = sum(gradient_magnitude(:));
end


function [gx, gy, gz] = forward_gradient_3d(x, gradient_weights)
%FORWARD_GRADIENT_3D 三维前向差分，Neumann 边界。
%
%   梯度数组与 x 尺寸相同。每个方向最后一个平面上的梯度为 0。

    [nx, ny, nz] = size(x);

    gx = zeros(size(x), 'like', x);
    gy = zeros(size(x), 'like', x);
    gz = zeros(size(x), 'like', x);

    if nx > 1
        gx(1:nx-1,:,:) = ...
            (x(2:nx,:,:) - x(1:nx-1,:,:)) .* gradient_weights(1);
    end

    if ny > 1
        gy(:,1:ny-1,:) = ...
            (x(:,2:ny,:) - x(:,1:ny-1,:)) .* gradient_weights(2);
    end

    if nz > 1
        gz(:,:,1:nz-1) = ...
            (x(:,:,2:nz) - x(:,:,1:nz-1)) .* gradient_weights(3);
    end
end


function voxel_spacing = normalize_voxel_spacing(voxel_spacing)
%NORMALIZE_VOXEL_SPACING 检查并归一化三维体素间距。

    if isa(voxel_spacing, 'gpuArray')
        voxel_spacing = gather(voxel_spacing);
    end

    voxel_spacing = double(voxel_spacing(:).');

    if numel(voxel_spacing) ~= 3 || ...
            any(~isfinite(voxel_spacing)) || any(voxel_spacing <= 0)
        error('TVnorm3D_gpu:InvalidVoxelSpacing', ...
            'voxel_spacing 必须是包含三个正有限数值的向量。');
    end

    % 只保留三个方向的相对比例，避免物理单位过小。
    voxel_spacing = voxel_spacing / min(voxel_spacing);
end
