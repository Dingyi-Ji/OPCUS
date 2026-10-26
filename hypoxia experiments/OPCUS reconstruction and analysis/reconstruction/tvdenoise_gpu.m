function u = tvdenoise_gpu(f, lambda, iters, voxel_spacing)
%TVDENOISE_GPU 三维各向同性 TV 邻近算子，支持 GPU。
%
%   u = tvdenoise_gpu(f, lambda, iters) 计算：
%
%       u = argmin_z  0.5*||z-f||_2^2 + lambda*TV_3D(z)
%
%   该函数可作为 TwIST_GPU 中的 Psi 邻近算子。这里 lambda 越大，
%   TV 平滑越强；lambda=0 时严格返回输入 f。
%
%   u = tvdenoise_gpu(f, lambda, iters, voxel_spacing) 可指定体素间距
%   [sx,sy,sz]。函数只使用三个方向的相对比例，并将其归一化。
%
%   数值方法：
%       使用 Chambolle 对偶投影算法。前向梯度与散度采用严格的
%       负伴随关系，并使用 Neumann 边界条件。
%
%   GPU 特性：
%       - 输入 f 为 gpuArray 时，所有中间变量都保留在 GPU 上；
%       - 输入 f 为 CPU 数组且当前存在可用 GPU 时，会自动转移到 GPU
%         计算，输出前再 gather 回 CPU；
%       - 输出形状与输入形状一致；
%       - 不强制转换为 double；
%       - 不依赖 padarray、convn 或 Image Processing Toolbox。
%
%   TwIST_GPU 中的用法：
%       tv_inner_iteration = 15;
%       psi_tv = @(x, lambda_twist) reshape( ...
%           tvdenoise_gpu(reshape(x, grid_size), ...
%           lambda_twist, tv_inner_iteration), size(x));
%
%   必须与采用相同边界和体素间距的 TVnorm3D_gpu 一起使用：
%       phi_tv = @(x) TVnorm3D_gpu(reshape(x, grid_size));

    if nargin < 4 || isempty(voxel_spacing)
        voxel_spacing = [1, 1, 1];
    end

    if ndims(f) ~= 3
        error('tvdenoise_gpu:InvalidDimension', ...
            '输入 f 必须是三维矩阵。如果输入来自 TwIST，请先 reshape(f, grid_size)。');
    end

    if ~(isnumeric(f) || islogical(f))
        error('tvdenoise_gpu:InvalidType', ...
            '输入 f 必须是数值或逻辑数组。');
    end

    lambda_value = get_numeric_scalar(lambda, 'lambda');
    if lambda_value < 0
        error('tvdenoise_gpu:InvalidLambda', ...
            'lambda 必须是非负标量。');
    end

    iters_value = get_numeric_scalar(iters, 'iters');
    if iters_value < 0 || iters_value ~= round(iters_value)
        error('tvdenoise_gpu:InvalidIterations', ...
            'iters 必须是非负整数。');
    end
    iters_value = round(iters_value);

    if lambda_value == 0 || iters_value == 0
        u = f;
        return;
    end

    if ~isfloat(f)
        f = single(f);
    end

    input_is_gpu = isa(f, 'gpuArray');
    transferred_to_gpu = false;

    if ~input_is_gpu && gpu_is_available()
        f = gpuArray(f);
        transferred_to_gpu = true;
    end

    voxel_spacing = normalize_voxel_spacing(voxel_spacing);
    gradient_weights = 1 ./ voxel_spacing;

    % 三维前向差分算子的范数不超过 sqrt(8)*max(weight)。
    % 因而稳定步长可取 1/(8*max(weight)^2)，这里留少量余量。
    dt = 0.124 / max(gradient_weights)^2;

    p1 = zeros(size(f), 'like', f);
    p2 = zeros(size(f), 'like', f);
    p3 = zeros(size(f), 'like', f);

    for iteration_index = 1:iters_value
        divp = divergence_3d(p1, p2, p3, gradient_weights);

        %  prox_{lambda*TV}(f) 的正确缩放：
        %  z = div(p) - f/lambda
        %  u = f - lambda*div(p)
        z = divp - f / lambda_value;

        [zx, zy, zz] = forward_gradient_3d(z, gradient_weights);
        gradient_magnitude = sqrt(abs(zx).^2 + abs(zy).^2 + abs(zz).^2);

        projection_denominator = 1 + dt * gradient_magnitude;
        p1 = (p1 + dt * zx) ./ projection_denominator;
        p2 = (p2 + dt * zy) ./ projection_denominator;
        p3 = (p3 + dt * zz) ./ projection_denominator;
    end

    divp_final = divergence_3d(p1, p2, p3, gradient_weights);
    u = f - lambda_value * divp_final;

    if transferred_to_gpu
        u = gather(u);
    end
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


function divp = divergence_3d(p1, p2, p3, gradient_weights)
%DIVERGENCE_3D 计算 div(p)，满足 div = -D^T。
%
%   forward_gradient_3d 中每个方向最后一个平面上的梯度为 0，因此
%   这里需要采用对应的边界形式，保证：
%
%       <D f, p> = -<f, div(p)>
%
%   该严格伴随关系是 TV 邻近算子正确收敛的必要条件。

    [nx, ny, nz] = size(p1);

    div_x = zeros(size(p1), 'like', p1);
    div_y = zeros(size(p2), 'like', p2);
    div_z = zeros(size(p3), 'like', p3);

    if nx > 1
        div_x(1,:,:) = p1(1,:,:);
        if nx > 2
            div_x(2:nx-1,:,:) = ...
                p1(2:nx-1,:,:) - p1(1:nx-2,:,:);
        end
        div_x(nx,:,:) = -p1(nx-1,:,:);
        div_x = div_x .* gradient_weights(1);
    end

    if ny > 1
        div_y(:,1,:) = p2(:,1,:);
        if ny > 2
            div_y(:,2:ny-1,:) = ...
                p2(:,2:ny-1,:) - p2(:,1:ny-2,:);
        end
        div_y(:,ny,:) = -p2(:,ny-1,:);
        div_y = div_y .* gradient_weights(2);
    end

    if nz > 1
        div_z(:,:,1) = p3(:,:,1);
        if nz > 2
            div_z(:,:,2:nz-1) = ...
                p3(:,:,2:nz-1) - p3(:,:,1:nz-2);
        end
        div_z(:,:,nz) = -p3(:,:,nz-1);
        div_z = div_z .* gradient_weights(3);
    end

    divp = div_x + div_y + div_z;
end


function voxel_spacing = normalize_voxel_spacing(voxel_spacing)
%NORMALIZE_VOXEL_SPACING 检查并归一化三维体素间距。

    if isa(voxel_spacing, 'gpuArray')
        voxel_spacing = gather(voxel_spacing);
    end

    voxel_spacing = double(voxel_spacing(:).');

    if numel(voxel_spacing) ~= 3 || ...
            any(~isfinite(voxel_spacing)) || any(voxel_spacing <= 0)
        error('tvdenoise_gpu:InvalidVoxelSpacing', ...
            'voxel_spacing 必须是包含三个正有限数值的向量。');
    end

    % 只保留三个方向的相对比例，避免物理单位过小。
    voxel_spacing = voxel_spacing / min(voxel_spacing);
end


function value = get_numeric_scalar(input_value, input_name)
%GET_NUMERIC_SCALAR 读取可能位于 GPU 上的数值标量。

    if isa(input_value, 'gpuArray')
        input_value = gather(input_value);
    end

    if ~isnumeric(input_value) || ~isscalar(input_value) || ...
            ~isfinite(input_value) || ~isreal(input_value)
        error('tvdenoise_gpu:InvalidScalar', ...
            '%s 必须是有限的实数标量。', input_name);
    end

    value = double(input_value);
end


function available = gpu_is_available()
%GPU_IS_AVAILABLE 检查当前 MATLAB 是否存在可用 GPU。

    available = false;
    try
        available = exist('gpuDeviceCount', 'file') == 2 && ...
            gpuDeviceCount > 0;
    catch
        available = false;
    end
end
