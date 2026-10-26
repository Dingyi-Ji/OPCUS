%% =========================================================
% Four-fiber ultrasound reconstruction + switchable compressed sensing (GPU continuous-frame version)

%% =========================================================

clc;
% close all;

% Modify according to the actual path of TwIST_GPU
% addpath('./TwIST_v2');

% GPU initialization
gpuDevice(1);


%% ==================== 0. User Configuration ====================

%% -------------------- Reconstruction Mode --------------------
% true  : compressed-sensing reconstruction
% false : direct DAS reconstruction
use_compressed_sensing = true;

% Two fusion modes:
% 'coherent'   : coherent summation
% 'incoherent' : incoherent summation
fusion_mode = 'incoherent';

%% -------------------- Incoherent Probabilistic-OR Fusion Settings --------------------
% Each fiber is individually normalized to [0,1] before fusion
% Using a high quantile instead of the maximum reduces the influence of isolated noise spikes on normalization
fiber_norm_lower_percentile = 0;
fiber_norm_upper_percentile = 99.999;

% Maximum number of voxels used for quantile estimation, avoiding a full sort of very large 3D volumes
fiber_norm_max_samples = 2000000;

% Global fusion weight of each fiber, in the range [0,1]
% Leave empty to set all fiber weights to 1
fiber_fusion_weights = [];

% Incoherent fusion method:
% 'probabilistic_or': 1-prod(1-w_i*I_i), preserves unique structures and suppresses over-bright intersections
% 'sum'             : keeps the original direct amplitude summation
incoherent_fusion_method = 'probabilistic_or';

%% -------------------- Continuous-Frame Reconstruction Settings --------------------
% By default, frames 1-40 are reconstructed continuously
% Reconstruct only frame 1 first to check the result
frame_indices_to_reconstruct = 1;

% You can also select arbitrary frames, e.g.:
% frame_indices_to_reconstruct = [1,5,10,20];

% Frame whose valid scan positions are used to build M
% By default, the first frame to be reconstructed is used
mask_frame_index = frame_indices_to_reconstruct(1);

%% -------------------- Inner-Product Consistency Check --------------------
run_inner_product_consistency_test = false;

% Under GPU single precision, 1e-4 is usually a reasonable relative-error threshold
inner_product_tolerance = 1e-4;

%% -------------------- Data Files of the Four Fibers --------------------
fiber_file_names = {
    '..\..\..\data\prior-guided dynamic imaging\sparse-transmit RF data\filtered_data_all0.mat';
    '..\..\..\data\prior-guided dynamic imaging\sparse-transmit RF data\filtered_data_all2.mat';
    '..\..\..\data\prior-guided dynamic imaging\sparse-transmit RF data\filtered_data_all3.mat';
    '..\..\..\data\prior-guided dynamic imaging\sparse-transmit RF data\filtered_data_all4.mat'
};

% Variable name in each MAT file
fiber_variable_names = {
    'filtered_data0';
    'filtered_data2';
    'filtered_data3';
    'filtered_data4'
};

%% -------------------- Discrete Coordinates of the Four Fibers --------------------
% Each row corresponds to one fiber:
% [fiber_x_index,fiber_y_index,fiber_z_index]
%
% The four coordinates from the original compressed-sensing code are kept for now
fiber_discrete_pos = [
    25,12,15;
    16,26,30;
    23,35,15;
    40,26,30    % Fiber 4:
];

%% -------------------- Temporal Preprocessing of Each Fiber --------------------
time_cut_start = 34;
front_zero_samples = [500, 500, 500, 500];
tail_zero_start = [1000,1000, 1000, 1000];
% front_zero_samples = [500];
% tail_zero_start = [1000];

% Threshold for valid-channel detection
% 0 means a channel is considered valid as long as its time series contains any non-zero value
signal_energy_threshold = 0;

%% -------------------- TwIST_GPU Parameters --------------------
% tau = 50 * 5e-3 * 0.05;

tau = 50 * 5e-3 * 1500;
% max_iteration = 100;
max_iteration = 10;

%% TV settings
tv_inner_iteration = 15;

% If voxel spacing is not considered for now, use [1,1,1] directly
tv_spacing = [1, 1, 1];

% If the actual voxel spacing needs to be considered, use:
% tv_spacing = [dx, dy, dz] / min([dx, dy, dz]);

% Note: psi_tv and phi_tv must be created after grid_size is defined,
% and must be passed into reconstruct_single_frame_gpu as parameters.

%% -------------------- Display and Save Settings --------------------
use_log_display = false;
% show_final_animation = true;
% The server has no graphical interface, so dynamic display is disabled
show_final_animation = false;
show_individual_fibers = false;

% Whether to keep all frames in memory
store_all_frames = true;

% Whether to save a MAT file for each frame
save_each_frame_result = false;

% Whether to save the overall result containing all frames
% save_overall_result = false;
% overall_output_file = 'four_fiber_continuous_reconstruction.mat';
% The output file path is the same directory as the input data
[data_dir, ~, ~] = fileparts(fiber_file_names{1});

% Save the overall result
save_overall_result = false;
overall_output_file = fullfile(data_dir, 'reconstruction_all_frames_0812M1_5min_firstframe_tau1500.mat');



%% ==================== 1. Imaging Parameters ====================

% Scan parameters kept consistent with the previous version of the code
sensor_row = 101;
sensor_column = 101;
delay_time = 1;

% Speed of sound, in m/s
c = 1540;

% Sampling interval, in s
dt = 1 / 50e6;

% Spatial resolution, in m
dx = 0.05e-3;
dy = dx;
dz = dx;

% 3D reconstruction grid
grid_size = [250, 250, 200];

% After grid_size is defined, create the TV proximity operator and the TV-norm function handle
psi_tv = @(x, lambda_twist) reshape( ...
    tvdenoise_gpu( ...
        reshape(x, grid_size), ...
        lambda_twist, ...
        tv_inner_iteration, ...
        tv_spacing), ...
    size(x));

phi_tv = @(x) TVnorm3D_gpu( ...
    reshape(x, grid_size), ...
    tv_spacing);

% Discrete spacing
discrete_interval = [11*5/3/2/2, 20*5/3/2/2];

% Spacing of the transmit array in the x and y directions
array_pitch_x = 14.7e-3 / 1080 * discrete_interval(1);
array_pitch_y = 14.7e-3 / 1920 * discrete_interval(2);

source_dx = array_pitch_x;
source_dy = array_pitch_y;

% Transmit-array offset
array_offset_x = ...
    (grid_size(1)*dx - sensor_row*array_pitch_x) / 2;

array_offset_y = ...
    (grid_size(2)*dy - sensor_column*array_pitch_y) / 2;

total_points = sensor_row * sensor_column;
n_fiber = size(fiber_discrete_pos, 1);


%% ==================== 2. Parameter Validation ====================

if numel(fiber_file_names) ~= n_fiber
    error('The number of fiber_file_names must equal the number of fiber coordinates.');
end

if numel(fiber_variable_names) ~= n_fiber
    error('The number of fiber_variable_names must equal the number of fiber coordinates.');
end

if numel(front_zero_samples) ~= n_fiber
    error('The length of front_zero_samples must equal the number of fibers.');
end

if numel(tail_zero_start) ~= n_fiber
    error('The length of tail_zero_start must equal the number of fibers.');
end

if ~any(strcmpi(fusion_mode, {'coherent', 'incoherent'}))
    error('fusion_mode can only be ''coherent'' or ''incoherent''.');
end

if any(~isfinite(fiber_discrete_pos(:)))
    error('fiber_discrete_pos contains invalid coordinates.');
end

if ~any(strcmpi(incoherent_fusion_method, ...
        {'probabilistic_or', 'sum'}))
    error(['incoherent_fusion_method can only be ', ...
        '''probabilistic_or'' or ''sum''.']);
end

if isempty(fiber_fusion_weights)
    fiber_fusion_weights = ones(1, n_fiber);
end

fiber_fusion_weights = double(fiber_fusion_weights(:).');

if numel(fiber_fusion_weights) ~= n_fiber
    error('The length of fiber_fusion_weights must equal the number of fibers.');
end

if any(~isfinite(fiber_fusion_weights)) || ...
        any(fiber_fusion_weights < 0) || ...
        any(fiber_fusion_weights > 1)
    error('All weights in fiber_fusion_weights must lie within [0,1].');
end

if fiber_norm_lower_percentile < 0 || ...
        fiber_norm_lower_percentile > 100 || ...
        fiber_norm_upper_percentile < 0 || ...
        fiber_norm_upper_percentile > 100 || ...
        fiber_norm_lower_percentile >= fiber_norm_upper_percentile
    error('The fiber normalization quantiles must satisfy 0 <= lower < upper <= 100.');
end

if ~isfinite(fiber_norm_max_samples) || fiber_norm_max_samples < 1000
    error('fiber_norm_max_samples must be a finite value no less than 1000.');
end

fusion_settings = struct();
fusion_settings.method = lower(incoherent_fusion_method);
fusion_settings.lower_percentile = fiber_norm_lower_percentile;
fusion_settings.upper_percentile = fiber_norm_upper_percentile;
fusion_settings.max_samples = round(fiber_norm_max_samples);
fusion_settings.weights = single(fiber_fusion_weights);


%% ==================== 3. Load the Four-Fiber Data ====================

fiber_data_cell = cell(n_fiber, 1);
frame_count_all = zeros(n_fiber, 1);

for fiber_idx = 1:n_fiber

    current_file = fiber_file_names{fiber_idx};
    current_variable = fiber_variable_names{fiber_idx};

    if exist(current_file, 'file') ~= 2
        error('Fiber data file not found: %s', current_file);
    end

    loaded_data = load(current_file, current_variable);

    if ~isfield(loaded_data, current_variable)
        error('File %s does not contain variable %s.', current_file, current_variable);
    end

    current_data = loaded_data.(current_variable);

    if ndims(current_data) == 3
        % [row,column,time], treated as a single frame
        current_row = size(current_data, 1);
        current_column = size(current_data, 2);
        current_frame_count = 1;

    elseif ndims(current_data) == 4
        % [frame,row,column,time]
        current_frame_count = size(current_data, 1);
        current_row = size(current_data, 2);
        current_column = size(current_data, 3);

    else
        error(['%s: %s must be a 3D or 4D array; ', ...
               'supported layouts are [row,column,time] or [frame,row,column,time].'], ...
               current_file, current_variable);
    end

    if current_row ~= sensor_row || current_column ~= sensor_column
        error(['Fiber %d data dimension error. ', ...
               'Expected [row,column]=[%d,%d], got [%d,%d].'], ...
               fiber_idx, sensor_row, sensor_column, ...
               current_row, current_column);
    end

    fiber_data_cell{fiber_idx} = current_data;
    frame_count_all(fiber_idx) = current_frame_count;
end

available_frame_num = min(frame_count_all);

if any(frame_indices_to_reconstruct < 1) || ...
        any(frame_indices_to_reconstruct > available_frame_num)
    error(['frame_indices_to_reconstruct exceeds the data range. ', ...
           'The number of frames commonly available across the four fibers is %d.'], available_frame_num);
end

if mask_frame_index < 1 || mask_frame_index > available_frame_num
    error('mask_frame_index exceeds the data range.');
end

n_reconstruction_frames = numel(frame_indices_to_reconstruct);

fprintf('Four-fiber data loading completed.\n');
fprintf('Maximum number of frames available for joint reconstruction: %d\n', available_frame_num);
fprintf('Number of frames to reconstruct this time: %d\n', n_reconstruction_frames);
fprintf('Reconstruction frame indices: ');
disp(frame_indices_to_reconstruct);


%% ==================== 4. Build the Valid-Channel Matrix M Using the Reference Frame ====================

fprintf('Building the sampling matrix M using frame %d.\n', mask_frame_index);

[d_real_reference, d_imag_reference, ...
 sensor_data_reference, time_length] = ...
    preprocess_single_frame_gpu( ...
    fiber_data_cell, ...
    mask_frame_index, ...
    sensor_row, ...
    sensor_column, ...
    time_cut_start, ...
    front_zero_samples, ...
    tail_zero_start, ...
    delay_time);

valid_mask_all = false(sensor_row, sensor_column, n_fiber);

for fiber_idx = 1:n_fiber

    current_sensor_data = sensor_data_reference{fiber_idx};

    channel_energy = gather(squeeze( ...
        sum(abs(current_sensor_data).^2, 3)));

    valid_mask_all(:, :, fiber_idx) = ...
        channel_energy > signal_energy_threshold;

    fprintf('Fiber %d valid channels: %d/%d (%.2f%%).\n', ...
        fiber_idx, ...
        nnz(valid_mask_all(:, :, fiber_idx)), ...
        total_points, ...
        100*nnz(valid_mask_all(:, :, fiber_idx))/total_points);
end

% Use the union of the valid scan positions of the four fibers
% The valid positions are already the result of random sampling, so they are not shuffled again
sample_mask = any(valid_mask_all, 3);

% If you really want to use only the valid positions of the first fiber, change it to:
% sample_mask = valid_mask_all(:, :, 1);

% Extract the valid positions directly in the natural linear-index order of the scan matrix
sampled_indices = find(sample_mask(:));

if isempty(sampled_indices)
    error('No valid scan positions were detected; unable to build the sampling matrix M.');
end

m = numel(sampled_indices);

% Build the selection matrix M
% Each row of M contains only a single 1, so the 2-norm of every row already equals 1
M = zeros(m, total_points, 'single');

M(sub2ind( ...
    [m, total_points], ...
    (1:m).', ...
    sampled_indices(:))) = 1;

M_gpu = gpuArray(M);

fprintf('Sampling matrix M built: %d x %d.\n', m, total_points);
fprintf('Total number of valid scan positions: %d/%d (%.2f%%).\n', ...
    m, total_points, 100*m/total_points);

%% Visualize the valid scan positions and the sampling matrix M
figure('Color', 'w', ...
    'Name', 'Four-fiber valid scan positions and sampling matrix M', ...
    'Position', [80, 80, 1400, 800]);

for fiber_idx = 1:n_fiber
    subplot(2, 3, fiber_idx);
    imagesc(valid_mask_all(:, :, fiber_idx));
    axis image;
    colormap(gca, 'gray');
    colorbar;
    title(sprintf('Fiber %d valid positions: %d', ...
        fiber_idx, nnz(valid_mask_all(:, :, fiber_idx))));
    xlabel('Column');
    ylabel('Row');
end

subplot(2, 3, 5);
imagesc(sample_mask);
axis image;
colormap(gca, 'gray');
colorbar;
title(sprintf('Union of four-fiber valid positions: %d', m));
xlabel('Column');
ylabel('Row');

subplot(2, 3, 6);
imagesc(M);
axis image;
colormap(gca, 'gray');
colorbar;
title(sprintf('Sampling matrix M: %d x %d', m, total_points));
xlabel('Linear index of the original scan positions');
ylabel('Valid sampling row index');

sgtitle('Sampling matrix M generated directly from the valid scan positions');
drawnow;


%% ==================== 5. Build the GPU Geometry Model ====================

%% 5.1 3D voxel coordinates
line_x = gpuArray(single( ...
    linspace(dx, grid_size(1)*dx, grid_size(1))));

line_y = gpuArray(single( ...
    linspace(dy, grid_size(2)*dy, grid_size(2))));

line_z = gpuArray(single( ...
    linspace(dz, grid_size(3)*dz, grid_size(3))));

% Use ndgrid so that the coordinate matrices have size [nx,ny,nz]
[x_m, y_m, z_m] = ndgrid(line_x, line_y, line_z);

%% 5.2 Physical coordinates of the four fibers
sensor_pos_all = zeros(n_fiber, 3);

for fiber_idx = 1:n_fiber

    fiber_x_index = fiber_discrete_pos(fiber_idx, 1);
    fiber_y_index = fiber_discrete_pos(fiber_idx, 2);
    fiber_z_index = fiber_discrete_pos(fiber_idx, 3);

    sensor_pos_all(fiber_idx, :) = [ ...
        fiber_x_index*source_dx + array_offset_x, ...
        fiber_y_index*source_dy + array_offset_y, ...
        fiber_z_index*c*dt];
end

sensor_pos_all = gpuArray(single(sensor_pos_all));

fprintf('Physical coordinates of the four fibers, in m:\n');
disp(gather(sensor_pos_all));

%% 5.3 Physical coordinates of the transmit elements
source_pos_all = zeros(total_points, 3);
source_index_all = zeros(total_points, 2);

for source_idx = 1:total_points

    [i, j] = ind2sub([sensor_row, sensor_column], source_idx);

    source_index_all(source_idx, :) = [i, j];

    source_pos_all(source_idx, :) = [ ...
        i*source_dx + array_offset_x, ...
        j*source_dy + array_offset_y, ...
        dz];
end

source_pos_all = gpuArray(single(source_pos_all));

%% 5.4 Precompute the distances r2 from all voxels to the four fibers
numel_volume = prod(grid_size);

x_flat = x_m(:);
y_flat = y_m(:);
z_flat = z_m(:);

r2_all = gpuArray.zeros( ...
    numel_volume, n_fiber, 'single');

for fiber_idx = 1:n_fiber

    sensor_pos = sensor_pos_all(fiber_idx, :);

    r2_all(:, fiber_idx) = sqrt( ...
        (x_flat - sensor_pos(1)).^2 + ...
        (y_flat - sensor_pos(2)).^2 + ...
        (z_flat - sensor_pos(3)).^2);
end

%% 5.5 Pack all geometric parameters into a structure
geometry = struct();

geometry.c = c;
geometry.dt = dt;
geometry.dx = dx;
geometry.dy = dy;
geometry.dz = dz;
geometry.grid_size = grid_size;
geometry.sensor_row = sensor_row;
geometry.sensor_column = sensor_column;
geometry.total_points = total_points;
geometry.numel_volume = numel_volume;
geometry.n_fiber = n_fiber;

geometry.source_dx = source_dx;
geometry.source_dy = source_dy;
geometry.array_offset_x = array_offset_x;
geometry.array_offset_y = array_offset_y;

geometry.source_pos_all = source_pos_all;
geometry.source_index_all = source_index_all;
geometry.sensor_pos_all = sensor_pos_all;

geometry.x_flat = x_flat;
geometry.y_flat = y_flat;
geometry.z_flat = z_flat;
geometry.r2_all = r2_all;

% The forward and adjoint operators only need to traverse the valid scan positions kept by M
geometry.active_source_indices = sampled_indices(:);


%% ==================== 6. Inner-Product Consistency Check of the Forward and Adjoint Operators ====================

if run_inner_product_consistency_test

    fprintf('\n========== Starting the inner-product consistency check of the forward and adjoint operators ==========\n');

    inner_product_results = ...
        run_inner_product_consistency_test_gpu( ...
        M_gpu, geometry, time_length, inner_product_tolerance);

    fprintf('========== Inner-product consistency check completed ==========\n\n');
end


%% ==================== 7. Continuous-Frame Reconstruction ====================

if store_all_frames

    reconstruction_all_frames = zeros( ...
        [n_reconstruction_frames, grid_size], 'single');

    estimated_memory_gb = ...
        numel(reconstruction_all_frames) * 4 / 1024^3;

    fprintf('Storing all frames is expected to require %.2f GB of memory.\n', ...
        estimated_memory_gb);
else
    reconstruction_all_frames = [];
end

fiber_amplitudes_all = cell(n_reconstruction_frames, 1);

for local_frame_idx = 1:n_reconstruction_frames

    current_frame_index = ...
        frame_indices_to_reconstruct(local_frame_idx);

    fprintf('\n========================================\n');
    fprintf('Starting reconstruction of frame %d, sequence %d/%d.\n', ...
        current_frame_index, local_frame_idx, n_reconstruction_frames);
    fprintf('========================================\n');

    %% Preprocessing of the current frame
    if current_frame_index == mask_frame_index

        % Directly reuse the preprocessing results obtained when M was built
        d_real_current = d_real_reference;
        d_imag_current = d_imag_reference;
        current_time_length = time_length;

    else

        [d_real_current, d_imag_current, ...
         ~, current_time_length] = ...
            preprocess_single_frame_gpu( ...
            fiber_data_cell, ...
            current_frame_index, ...
            sensor_row, ...
            sensor_column, ...
            time_cut_start, ...
            front_zero_samples, ...
            tail_zero_start, ...
            delay_time);
    end

    if current_time_length ~= time_length
        error('The time length of frame %d differs from the reference frame.', current_frame_index);
    end

    %% Reconstruction of the current frame
    [pa_img_current_gpu, fiber_amplitudes_current] = ...
        reconstruct_single_frame_gpu( ...
        d_real_current, ...
        d_imag_current, ...
        M_gpu, ...
        geometry, ...
        time_length, ...
        use_compressed_sensing, ...
        fusion_mode, ...
        tau, ...
        max_iteration, ...
        psi_tv, ...
        phi_tv, ...
        fusion_settings);

    pa_img_current = gather(pa_img_current_gpu);

    %% Store in memory
    if store_all_frames
        reconstruction_all_frames( ...
            local_frame_idx, :, :, :) = pa_img_current;
    end

    fiber_amplitudes_all{local_frame_idx} = ...
        fiber_amplitudes_current;

    %% Optional: save frame by frame
    if save_each_frame_result

        current_output_file = sprintf( ...
            'four_fiber_reconstruction_frame_%04d.mat', ...
            current_frame_index);

        pa_img_1 = pa_img_current;
        fiber_amplitudes = fiber_amplitudes_current;

        save(current_output_file, ...
            'pa_img_1', ...
            'fiber_amplitudes', ...
            'M', ...
            'sample_mask', ...
            'valid_mask_all', ...
            'sampled_indices', ...
            'fiber_discrete_pos', ...
            'fusion_mode', ...
            'fusion_settings', ...
            'use_compressed_sensing', ...
            '-v7.3');

        fprintf('The result of frame %d has been saved to: %s\n', ...
            current_frame_index, current_output_file);
    end

    fprintf('Reconstruction of frame %d completed.\n', current_frame_index);

    clear d_real_current d_imag_current
    clear pa_img_current_gpu pa_img_current
end

% Release the reference-frame data
clear d_real_reference d_imag_reference sensor_data_reference


%% ==================== 8. Save the Overall Result ====================

first_frame_mip_png_file = ...
    fullfile(data_dir, 'first_frame_three_direction_mip_0812M1_5min_tau1500.png');

if save_overall_result

    if ~store_all_frames
        warning('store_all_frames is false, so the overall result containing all frames cannot be saved.');
    else
        fprintf('Saving the overall continuous-frame reconstruction result...\n');

        save(overall_output_file, ...
            'reconstruction_all_frames', ...
            '-v7.3');

        fprintf('The overall continuous-frame reconstruction result has been saved to: %s\n', overall_output_file);
    end
end

%% ==================== 8.1 Save the Three-Direction MIP PNG of the First Frame ====================

if ~store_all_frames || isempty(reconstruction_all_frames)
    error('store_all_frames is false, so the three-direction MIP images of the first frame cannot be generated.');
end

% The dimensions of reconstruction_all_frames are [frame,x,y,z]
first_frame_volume = squeeze( ...
    reconstruction_all_frames(1, :, :, :));

% Keep consistent with the three-direction maximum-projection approach used in the current code
yz_mip = squeeze(max(first_frame_volume, [], 1));
xz_mip = squeeze(max(first_frame_volume, [], 2));
xy_mip = squeeze(max(first_frame_volume, [], 3));

% Use a unified display range for the first frame
valid_values = first_frame_volume(isfinite(first_frame_volume));

if isempty(valid_values)
    mip_clim = [0, 1];
else
    mip_clim = [min(valid_values(:)), max(valid_values(:))];

    if mip_clim(1) == mip_clim(2)
        padding = max(abs(mip_clim(1))*1e-6, 1e-6);
        mip_clim = mip_clim + [-padding, padding];
    end
end

mip_figure = figure( ...
    'Visible', 'on', ...
    'Color', 'w', ...
    'Position', [100, 100, 1500, 480]);

subplot(1, 3, 1);
imagesc(yz_mip);
axis image;
colormap gray;
colorbar;
caxis(mip_clim);
title('YZ MIP');
xlabel('Y index');
ylabel('Z index');

subplot(1, 3, 2);
imagesc(xz_mip);
axis image;
colormap gray;
colorbar;
caxis(mip_clim);
title('XZ MIP');
xlabel('X index');
ylabel('Z index');

subplot(1, 3, 3);
imagesc(xy_mip);
axis image;
colormap gray;
colorbar;
caxis(mip_clim);
title('XY MIP');
xlabel('X index');
ylabel('Y index');

sgtitle('First Frame 3D Reconstruction: Three-Direction MIP');

exportgraphics( ...
    mip_figure, ...
    first_frame_mip_png_file, ...
    'Resolution', 300);

% close(mip_figure);

fprintf('The three-direction maximum-intensity projections of the first frame have been saved to: %s\n', ...
    first_frame_mip_png_file);


%% ==================== 9. Display of the Continuous-Frame Results ====================

if store_all_frames && show_final_animation

    figure('Color', 'w', ...
        'Name', 'Four-fiber continuous-frame reconstruction results', ...
        'Position', [80, 80, 1500, 500]);

    for local_frame_idx = 1:n_reconstruction_frames

        current_frame_index = ...
            frame_indices_to_reconstruct(local_frame_idx);

        current_volume = squeeze( ...
            reconstruction_all_frames(local_frame_idx, :, :, :));

        if use_log_display
            current_volume = log(current_volume + 1e-6);
        end

        current_volume = log(current_volume+1e-2);

        subplot(1, 3, 1);
        imagesc(squeeze(max(current_volume, [], 1)));
        axis image;
        colormap gray;
        colorbar;
        title('YZ maximum projection');

        subplot(1, 3, 2);
        imagesc(squeeze(max(current_volume, [], 2)));
        axis image;
        colormap gray;
        colorbar;
        title('XZ maximum projection');

        subplot(1, 3, 3);
        imagesc(squeeze(max(current_volume, [], 3)));
        axis image;
        colormap gray;
        colorbar;
        title('XY maximum projection');

        sgtitle(sprintf( ...
            'Four-fiber %s reconstruction, frame %d, sequence %d/%d', ...
            fusion_mode, current_frame_index, ...
            local_frame_idx, n_reconstruction_frames));

        drawnow;
        pause(0.3);
    end
end


%% =========================================================
% Local function definitions
%% =========================================================

function frame_data = get_single_frame_data(all_data, frame_idx)
% Extract one frame from 3D or 4D data
%
% Supported inputs:
%   [row,column,time]
%   [frame,row,column,time]

    if ndims(all_data) == 3

        if frame_idx ~= 1
            error('The current data has only one frame, but frame %d was requested.', frame_idx);
        end

        frame_data = all_data;

    elseif ndims(all_data) == 4

        if frame_idx > size(all_data, 1)
            error('frame_idx exceeds the number of frames in the data.');
        end

        frame_data = squeeze(all_data(frame_idx, :, :, :));

    else
        error('The input data must be a 3D or 4D array.');
    end
end


function [d_real_cell, d_imag_cell, sensor_data_cell, time_length] = ...
    preprocess_single_frame_gpu( ...
    fiber_data_cell, ...
    frame_idx, ...
    sensor_row, ...
    sensor_column, ...
    time_cut_start, ...
    front_zero_samples, ...
    tail_zero_start, ...
    delay_time)
% Perform temporal preprocessing and GPU Hilbert transform on the four-fiber data of the current frame

    n_fiber = numel(fiber_data_cell);
    total_points = sensor_row * sensor_column;

    d_real_cell = cell(n_fiber, 1);
    d_imag_cell = cell(n_fiber, 1);
    sensor_data_cell = cell(n_fiber, 1);

    time_length = [];

    for fiber_idx = 1:n_fiber

        current_data = get_single_frame_data( ...
            fiber_data_cell{fiber_idx}, frame_idx);

        if size(current_data, 1) ~= sensor_row || ...
                size(current_data, 2) ~= sensor_column
            error('The [row,column] dimensions of the current frame data are inconsistent with the settings.');
        end

        %% Temporal truncation
        if size(current_data, 3) < time_cut_start
            error('The data length of fiber %d in the current frame is insufficient.', fiber_idx);
        end

        current_data = current_data(:, :, time_cut_start:end);

        %% Zero the front part
        current_front_zero = front_zero_samples(fiber_idx);

        if current_front_zero > 0
            current_front_zero = min( ...
                current_front_zero, size(current_data, 3));

            current_data(:, :, 1:current_front_zero) = 0;
        end

        %% Zero the tail part
        current_tail_zero_start = tail_zero_start(fiber_idx);

        if isfinite(current_tail_zero_start) && ...
                current_tail_zero_start <= size(current_data, 3)
            current_data(:, :, current_tail_zero_start:end) = 0;
        end

        %% Transfer to the GPU
        if size(current_data, 3) < delay_time
            error('The data length of fiber %d in the current frame is shorter than delay_time.', fiber_idx);
        end

        sensor_data = gpuArray(single( ...
            current_data(:, :, delay_time:end)));

        current_time_length = size(sensor_data, 3);

        if isempty(time_length)
            time_length = current_time_length;
        elseif current_time_length ~= time_length
            error('The time lengths of the four fibers in the current frame are inconsistent.');
        end

        sensor_data_cell{fiber_idx} = sensor_data;

        %% GPU Hilbert transform
        tmp_data = reshape(sensor_data, [], time_length);
        tmp_data = tmp_data.';

        hilbert_data = hilbert(tmp_data);
        hilbert_data = hilbert_data.';

        sensor_data_real = real( ...
            reshape(hilbert_data, size(sensor_data)));

        sensor_data_imag = imag( ...
            reshape(hilbert_data, size(sensor_data)));

        % Convert to [total_points,time_length] on the GPU
        d_real_cell{fiber_idx} = reshape( ...
            sensor_data_real, total_points, time_length);

        d_imag_cell{fiber_idx} = reshape( ...
            sensor_data_imag, total_points, time_length);
    end
end


function [pa_img_1_gpu, fiber_amplitudes] = ...
    reconstruct_single_frame_gpu( ...
    d_real_cell, ...
    d_imag_cell, ...
    M_gpu, ...
    geometry, ...
    time_length, ...
    use_compressed_sensing, ...
    fusion_mode, ...
    tau, ...
    max_iteration, ...
    psi_tv, ...
    phi_tv, ...
    fusion_settings)
% Perform four-fiber GPU reconstruction of the current frame

    n_fiber = geometry.n_fiber;
    m = size(M_gpu, 1);
    sensor_row = geometry.sensor_row;
    sensor_column = geometry.sensor_column;
    grid_size = geometry.grid_size;

    if use_compressed_sensing

        fprintf('The current frame uses compressed-sensing reconstruction; fusion mode: %s\n', fusion_mode);

        if strcmpi(fusion_mode, 'coherent')

            %% Compressed-sensing coherent summation
            d_real_joint = gpuArray.zeros( ...
                n_fiber*m, time_length, 'single');

            d_imag_joint = gpuArray.zeros( ...
                n_fiber*m, time_length, 'single');

            for fiber_idx = 1:n_fiber

                row_start = (fiber_idx-1)*m + 1;
                row_end = fiber_idx*m;

                d_real_joint(row_start:row_end, :) = ...
                    M_gpu * d_real_cell{fiber_idx};

                d_imag_joint(row_start:row_end, :) = ...
                    M_gpu * d_imag_cell{fiber_idx};
            end

            forward_joint = @(x) quad_fiber_forward_gpu( ...
                x, M_gpu, m, geometry, time_length);

            adjoint_joint = @(y) quad_fiber_adjoint_gpu( ...
                y, M_gpu, m, geometry, time_length);

            disp('Current frame: joint compressed-sensing reconstruction of the coherent real part...');
            tic;
            x_real = TwIST_GPU( ...
                d_real_joint, ...
                forward_joint, ...
                tau, ...
                'AT', adjoint_joint, ...
                'MaxiterA', max_iteration, ...
                'Verbose', 0,...
                'PSI', psi_tv, ...
                'PHI', phi_tv, ...
                'Sparse', 0, ...
                'StopCriterion', 1, ...
                'Debias', 0);
            toc;

            disp('Current frame: joint compressed-sensing reconstruction of the coherent imaginary part...');
            tic;
            x_imag = TwIST_GPU( ...
                d_imag_joint, ...
                forward_joint, ...
                tau, ...
                'AT', adjoint_joint, ...
                'MaxiterA', max_iteration, ...
                'Verbose', 0,...
                'PSI', psi_tv, ...
                'PHI', phi_tv, ...
                'Sparse', 0, ...
                'StopCriterion', 1, ...
                'Debias', 0);
            toc;

            x_real = reshape(x_real, grid_size);
            x_imag = reshape(x_imag, grid_size);

            pa_img_1_gpu = sqrt(x_real.^2 + x_imag.^2);

            fiber_amplitudes = [];

        else

            %% Compressed-sensing incoherent summation
            % fiber_amplitudes always stores the original amplitude of each fiber;
            % pa_img_1_gpu stores the final fused result
            fiber_amplitudes = zeros( ...
                [grid_size, n_fiber], 'single');

            use_probabilistic_or = strcmpi( ...
                fusion_settings.method, 'probabilistic_or');

            if use_probabilistic_or
                % Compute 1-prod(1-I_f) using the complementary-product form
                fusion_complement_gpu = gpuArray.ones( ...
                    grid_size, 'single');
            else
                % 'sum' mode keeps the original direct amplitude summation
                pa_img_1_gpu = gpuArray.zeros( ...
                    grid_size, 'single');
            end

            for fiber_idx = 1:n_fiber

                fprintf('Current frame: compressed-sensing reconstruction of the real part of fiber %d...\n', fiber_idx);

                forward_single = @(x) M_gpu * reshape( ...
                    forward_das_single_fiber_gpu( ...
                    x, geometry, fiber_idx, time_length), ...
                    [], time_length);

                adjoint_single = @(y) adjoint_das_single_fiber_gpu( ...
                    reshape(M_gpu' * y, ...
                    sensor_row, sensor_column, time_length), ...
                    geometry, fiber_idx, time_length);

                tic;
                x_real = TwIST_GPU( ...
                    M_gpu * d_real_cell{fiber_idx}, ...
                    forward_single, ...
                    tau, ...
                    'AT', adjoint_single, ...
                    'MaxiterA', max_iteration, ...
                    'Verbose', 0,...
                    'PSI', psi_tv, ...
                    'PHI', phi_tv, ...
                    'Sparse', 0, ...
                    'StopCriterion', 1, ...
                    'Debias', 0);
                toc;

                fprintf('Current frame: compressed-sensing reconstruction of the imaginary part of fiber %d...\n', fiber_idx);

                tic;
                x_imag = TwIST_GPU( ...
                    M_gpu * d_imag_cell{fiber_idx}, ...
                    forward_single, ...
                    tau, ...
                    'AT', adjoint_single, ...
                    'MaxiterA', max_iteration, ...
                    'Verbose', 0,...
                    'PSI', psi_tv, ...
                    'PHI', phi_tv, ...
                    'Sparse', 0, ...
                    'StopCriterion', 1, ...
                    'Debias', 0);
                toc;

                x_real = reshape(x_real, grid_size);
                x_imag = reshape(x_imag, grid_size);

                current_fiber_amplitude = sqrt( ...
                    x_real.^2 + x_imag.^2);

                % Save the unnormalized original amplitude of each fiber
                fiber_amplitudes(:, :, :, fiber_idx) = ...
                    gather(single(current_fiber_amplitude));

                if use_probabilistic_or

                    % Normalize each fiber independently to [0,1] first
                    [current_normalized_amplitude, ...
                     norm_lower_value, ...
                     norm_upper_value] = ...
                        normalize_fiber_amplitude_gpu( ...
                        current_fiber_amplitude, ...
                        fusion_settings.lower_percentile, ...
                        fusion_settings.upper_percentile, ...
                        fusion_settings.max_samples);

                    current_weight = single( ...
                        fusion_settings.weights(fiber_idx));

                    weighted_normalized_amplitude = min(max( ...
                        current_weight * current_normalized_amplitude, ...
                        0), 1);

                    % Probabilistic OR: F=1-prod(1-w_f*I_f)
                    fusion_complement_gpu = ...
                        fusion_complement_gpu .* ...
                        (1 - weighted_normalized_amplitude);

                    fprintf(['Fiber %d normalization quantile range: ', ...
                             '[%.6g, %.6g], fusion weight: %.3g.\n'], ...
                        fiber_idx, norm_lower_value, ...
                        norm_upper_value, current_weight);
                else
                    pa_img_1_gpu = ...
                        pa_img_1_gpu + current_fiber_amplitude;
                end
            end

            if use_probabilistic_or
                pa_img_1_gpu = 1 - fusion_complement_gpu;
            end
        end

    else

        fprintf('The current frame uses direct DAS reconstruction; fusion mode: %s\n', fusion_mode);

        fiber_amplitudes = zeros( ...
            [grid_size, n_fiber], 'single');

        if strcmpi(fusion_mode, 'coherent')

            %% Direct DAS coherent summation
            pa_real_sum = gpuArray.zeros(grid_size, 'single');
            pa_imag_sum = gpuArray.zeros(grid_size, 'single');

            for fiber_idx = 1:n_fiber

                fprintf('Current frame: coherent DAS reconstruction of fiber %d in progress...\n', fiber_idx);

                downsampled_real = M_gpu' * ...
                    (M_gpu * d_real_cell{fiber_idx});

                downsampled_imag = M_gpu' * ...
                    (M_gpu * d_imag_cell{fiber_idx});

                data_real = reshape( ...
                    downsampled_real, ...
                    sensor_row, sensor_column, time_length);

                data_imag = reshape( ...
                    downsampled_imag, ...
                    sensor_row, sensor_column, time_length);

                pa_real = adjoint_das_single_fiber_gpu( ...
                    data_real, geometry, fiber_idx, time_length);

                pa_imag = adjoint_das_single_fiber_gpu( ...
                    data_imag, geometry, fiber_idx, time_length);

                current_fiber_amplitude = sqrt( ...
                    pa_real.^2 + pa_imag.^2);

                fiber_amplitudes(:, :, :, fiber_idx) = ...
                    gather(single(current_fiber_amplitude));

                pa_real_sum = pa_real_sum + pa_real;
                pa_imag_sum = pa_imag_sum + pa_imag;
            end

            pa_img_1_gpu = sqrt( ...
                pa_real_sum.^2 + pa_imag_sum.^2);

        else

            %% Direct DAS incoherent summation
            use_probabilistic_or = strcmpi( ...
                fusion_settings.method, 'probabilistic_or');

            if use_probabilistic_or
                % Compute 1-prod(1-I_f) using the complementary-product form
                fusion_complement_gpu = gpuArray.ones( ...
                    grid_size, 'single');
            else
                % 'sum' mode keeps the original direct amplitude summation
                pa_img_1_gpu = gpuArray.zeros( ...
                    grid_size, 'single');
            end

            for fiber_idx = 1:n_fiber

                fprintf('Current frame: incoherent DAS reconstruction of fiber %d in progress...\n', fiber_idx);

                downsampled_real = M_gpu' * ...
                    (M_gpu * d_real_cell{fiber_idx});

                downsampled_imag = M_gpu' * ...
                    (M_gpu * d_imag_cell{fiber_idx});

                data_real = reshape( ...
                    downsampled_real, ...
                    sensor_row, sensor_column, time_length);

                data_imag = reshape( ...
                    downsampled_imag, ...
                    sensor_row, sensor_column, time_length);

                pa_real = adjoint_das_single_fiber_gpu( ...
                    data_real, geometry, fiber_idx, time_length);

                pa_imag = adjoint_das_single_fiber_gpu( ...
                    data_imag, geometry, fiber_idx, time_length);

                current_fiber_amplitude = sqrt( ...
                    pa_real.^2 + pa_imag.^2);

                % Save the unnormalized original amplitude of each fiber
                fiber_amplitudes(:, :, :, fiber_idx) = ...
                    gather(single(current_fiber_amplitude));

                if use_probabilistic_or

                    % Normalize each fiber independently to [0,1] first
                    [current_normalized_amplitude, ...
                     norm_lower_value, ...
                     norm_upper_value] = ...
                        normalize_fiber_amplitude_gpu( ...
                        current_fiber_amplitude, ...
                        fusion_settings.lower_percentile, ...
                        fusion_settings.upper_percentile, ...
                        fusion_settings.max_samples);

                    current_weight = single( ...
                        fusion_settings.weights(fiber_idx));

                    weighted_normalized_amplitude = min(max( ...
                        current_weight * current_normalized_amplitude, ...
                        0), 1);

                    % Probabilistic OR: F=1-prod(1-w_f*I_f)
                    fusion_complement_gpu = ...
                        fusion_complement_gpu .* ...
                        (1 - weighted_normalized_amplitude);
                    % fusion_complement_gpu = ...
                    % fusion_complement_gpu+weighted_normalized_amplitude;

                    fprintf(['Fiber %d normalization quantile range: ', ...
                             '[%.6g, %.6g], fusion weight: %.3g.\n'], ...
                        fiber_idx, norm_lower_value, ...
                        norm_upper_value, current_weight);
                else
                    pa_img_1_gpu = ...
                        pa_img_1_gpu + current_fiber_amplitude;
                end
            end

            if use_probabilistic_or
                pa_img_1_gpu = 1 - fusion_complement_gpu;
                % pa_img_1_gpu = fusion_complement_gpu;
            end
        end
    end
end


function results = run_inner_product_consistency_test_gpu( ...
    M_gpu, geometry, time_length, tolerance)
% Verify the inner-product consistency of the four-fiber joint forward operator and the joint adjoint operator
%
% Joint coherent forward operator:
%   A_c(x) = [M*A_1(x);
%             M*A_2(x);
%             M*A_3(x);
%             M*A_4(x)]
%
% It should satisfy:
%   <A_c(x),y> = <x,A_c^T(y)>
%
% Also verify:
%   A_c^T(y) = sum_f A_f^T(M^T*y_f)

    n_fiber = geometry.n_fiber;
    m = size(M_gpu, 1);

    fprintf('Generating a random test volume and random joint measurements...\n');

    % Random test image
    x_test = gpuArray(single(randn(geometry.grid_size)));

    % Random joint measurements, with dimension [n_fiber*m,time_length]
    y_test_joint = gpuArray(single( ...
        randn(n_fiber*m, time_length)));

    %% 1. Joint coherent forward operator
    forward_joint_result = quad_fiber_forward_gpu( ...
        x_test, M_gpu, m, geometry, time_length);

    %% 2. Test the forward and adjoint operators of the four fibers separately
    fiber_lhs = zeros(n_fiber, 1);
    fiber_rhs = zeros(n_fiber, 1);
    fiber_relative_error = zeros(n_fiber, 1);

    adjoint_sum_expected = gpuArray.zeros( ...
        geometry.grid_size, 'single');

    for fiber_idx = 1:n_fiber

        row_start = (fiber_idx-1)*m + 1;
        row_end = fiber_idx*m;

        y_test_fiber = y_test_joint(row_start:row_end, :);
        forward_fiber_result = ...
            forward_joint_result(row_start:row_end, :);

        adjoint_fiber_result = adjoint_das_single_fiber_gpu( ...
            reshape(M_gpu' * y_test_fiber, ...
            geometry.sensor_row, ...
            geometry.sensor_column, ...
            time_length), ...
            geometry, fiber_idx, time_length);

        adjoint_sum_expected = ...
            adjoint_sum_expected + adjoint_fiber_result;

        % <A_f*x, y_f>
        fiber_lhs(fiber_idx) = gpu_inner_product( ...
            forward_fiber_result, y_test_fiber);

        % <x, A_f^T*y_f>
        fiber_rhs(fiber_idx) = gpu_inner_product( ...
            x_test, adjoint_fiber_result);

        denominator = max([ ...
            abs(fiber_lhs(fiber_idx)), ...
            abs(fiber_rhs(fiber_idx)), ...
            eps('double')]);

        fiber_relative_error(fiber_idx) = ...
            abs(fiber_lhs(fiber_idx) - fiber_rhs(fiber_idx)) / ...
            denominator;

        fprintf(['Fiber %d: <Ax,y>=%.10e, <x,A''y>=%.10e, ', ...
                 'relative error=%.3e\n'], ...
            fiber_idx, ...
            fiber_lhs(fiber_idx), ...
            fiber_rhs(fiber_idx), ...
            fiber_relative_error(fiber_idx));
    end

    %% 3. Test the joint coherent adjoint operator
    adjoint_joint_result = quad_fiber_adjoint_gpu( ...
        y_test_joint, M_gpu, m, geometry, time_length);

    % <A_c*x, y>
    joint_lhs = gpu_inner_product( ...
        forward_joint_result, y_test_joint);

    % <x, A_c^T*y>
    joint_rhs = gpu_inner_product( ...
        x_test, adjoint_joint_result);

    joint_denominator = max([ ...
        abs(joint_lhs), ...
        abs(joint_rhs), ...
        eps('double')]);

    joint_relative_error = ...
        abs(joint_lhs - joint_rhs) / joint_denominator;

    %% 4. Compare the joint coherent result with the sum of the four independent results
    fiber_lhs_sum = sum(fiber_lhs);
    fiber_rhs_sum = sum(fiber_rhs);

    lhs_sum_difference = abs(joint_lhs - fiber_lhs_sum);
    rhs_sum_difference = abs(joint_rhs - fiber_rhs_sum);

    lhs_sum_relative_error = lhs_sum_difference / ...
        max([abs(joint_lhs), abs(fiber_lhs_sum), eps('double')]);

    rhs_sum_relative_error = rhs_sum_difference / ...
        max([abs(joint_rhs), abs(fiber_rhs_sum), eps('double')]);

    %% 5. Directly compare the joint adjoint image with the sum of the four adjoint images
    adjoint_difference_norm = gather(norm( ...
        adjoint_joint_result(:) - adjoint_sum_expected(:)));

    adjoint_expected_norm = gather(norm( ...
        adjoint_sum_expected(:)));

    adjoint_relative_difference = adjoint_difference_norm / ...
        max(adjoint_expected_norm, eps('single'));

    %% 6. Output the result table
    operator_name = arrayfun(@(fiber_idx) ...
        sprintf('Fiber %d', fiber_idx), ...
        1:n_fiber, 'UniformOutput', false).';

    operator_name{end+1, 1} = 'Coherent joint';

    lhs_all = [fiber_lhs(:); joint_lhs];
    rhs_all = [fiber_rhs(:); joint_rhs];
    relative_error_all = [fiber_relative_error(:); joint_relative_error];

    result_table = table( ...
        operator_name, ...
        lhs_all, ...
        rhs_all, ...
        relative_error_all, ...
        'VariableNames', { ...
        'Operator', ...
        'ForwardInnerProduct', ...
        'AdjointInnerProduct', ...
        'RelativeError'});

    disp(result_table);

    fprintf('Joint forward inner product: <A_cx,y> = %.10e\n', joint_lhs);
    fprintf('Sum of the four independent forward inner products = %.10e\n', fiber_lhs_sum);
    fprintf('Relative error between the two = %.3e\n', lhs_sum_relative_error);

    fprintf('Joint adjoint inner product: <x,A_c^Ty> = %.10e\n', joint_rhs);
    fprintf('Sum of the four independent adjoint inner products = %.10e\n', fiber_rhs_sum);
    fprintf('Relative error between the two = %.3e\n', rhs_sum_relative_error);

    fprintf(['Relative difference between the joint adjoint image and ', ...
             'the sum of the four independent adjoint images = %.3e\n'], ...
        adjoint_relative_difference);

    %% 7. Determine pass or fail
    all_fiber_pass = all(fiber_relative_error < tolerance);
    joint_pass = joint_relative_error < tolerance;

    coherent_sum_pass = ...
        lhs_sum_relative_error < tolerance && ...
        rhs_sum_relative_error < tolerance && ...
        adjoint_relative_difference < tolerance;

    if all_fiber_pass && joint_pass && coherent_sum_pass
        fprintf('Inner-product consistency check passed.\n');
    else
        warning(['The inner-product consistency check has errors exceeding the threshold; ', ...
                 'please check the forward operator, adjoint operator, sampling matrix M, and GPU precision.']);
    end

    %% 8. Save the results
    results = struct();

    results.fiber_lhs = fiber_lhs;
    results.fiber_rhs = fiber_rhs;
    results.fiber_relative_error = fiber_relative_error;

    results.joint_lhs = joint_lhs;
    results.joint_rhs = joint_rhs;
    results.joint_relative_error = joint_relative_error;

    results.fiber_lhs_sum = fiber_lhs_sum;
    results.fiber_rhs_sum = fiber_rhs_sum;
    results.lhs_sum_relative_error = lhs_sum_relative_error;
    results.rhs_sum_relative_error = rhs_sum_relative_error;
    results.adjoint_relative_difference = adjoint_relative_difference;

    results.result_table = result_table;
    results.all_fiber_pass = all_fiber_pass;
    results.joint_pass = joint_pass;
    results.coherent_sum_pass = coherent_sum_pass;
end


function value = gpu_inner_product(a, b)
% Inner product of GPU arrays
%
% For real operators, this is equivalent to sum(a(:).*b(:))
% conj is used so that it also works for complex operators

    value = double(gather(real( ...
        sum(conj(a(:)) .* b(:)))));
end


function [normalized_image, lower_value, upper_value] = ...
    normalize_fiber_amplitude_gpu( ...
    amplitude, ...
    lower_percentile, ...
    upper_percentile, ...
    max_samples)
% Robustly normalize the amplitude image of a single fiber to [0,1]
%
% Use low and high quantiles instead of a simple maximum:
%   I_norm=(I-P_lower)/(P_upper-P_lower)
%
% This prevents individual noise spikes from over-compressing the dynamic range of the whole fiber image.
% Both input and output stay on the GPU.

    if isa(amplitude, 'gpuArray')
        amplitude_gpu = single(amplitude);
    else
        amplitude_gpu = gpuArray(single(amplitude));
    end

    % The amplitude is theoretically non-negative; NaN and Inf are also removed to prevent quantile estimation from failing
    amplitude_gpu = max(amplitude_gpu, 0);
    amplitude_gpu(~isfinite(amplitude_gpu)) = 0;

    [lower_value, upper_value] = estimate_percentile_range_gpu( ...
        amplitude_gpu, ...
        lower_percentile, ...
        upper_percentile, ...
        max_samples);

    scale_value = upper_value - lower_value;
    scale_tolerance = 100 * eps(max( ...
        1, max(abs([lower_value, upper_value]))));

    normalized_image = zeros( ...
        size(amplitude_gpu), 'like', amplitude_gpu);

    if isfinite(scale_value) && scale_value > scale_tolerance

        normalized_image = ...
            (amplitude_gpu - lower_value) / scale_value;

        normalized_image = min(max(normalized_image, 0), 1);

    else
        % If the low and high quantiles are almost identical, the image is usually all-zero or nearly constant.
        % For a non-zero constant image, fall back to maximum normalization to avoid incorrectly zeroing valid signals.
        max_value = double(gather(max(amplitude_gpu(:))));

        if isfinite(max_value) && max_value > 0
            normalized_image = min(max( ...
                amplitude_gpu / max_value, 0), 1);

            lower_value = 0;
            upper_value = max_value;
        end
    end
end


function [lower_value, upper_value] = estimate_percentile_range_gpu( ...
    data, lower_percentile, upper_percentile, max_samples)
% Estimate low and high quantiles on the GPU, without relying on the Statistics and Machine Learning Toolbox
%
% For very large 3D volumes, sample at equal intervals before sorting, avoiding a full sort of all voxels every time.

    if isa(data, 'gpuArray')
        data_gpu = single(data);
    else
        data_gpu = gpuArray(single(data));
    end

    data_flat = data_gpu(:);
    data_flat = data_flat(isfinite(data_flat));
    data_flat = max(data_flat, 0);

    if isempty(data_flat)
        lower_value = 0;
        upper_value = 0;
        return;
    end

    n_total = numel(data_flat);

    if n_total > max_samples
        sample_stride = ceil(n_total / max_samples);
        data_flat = data_flat(1:sample_stride:n_total);
    end

    sorted_values = sort(data_flat);

    lower_value = percentile_from_sorted_values( ...
        sorted_values, lower_percentile);

    upper_value = percentile_from_sorted_values( ...
        sorted_values, upper_percentile);
end


function value = percentile_from_sorted_values(sorted_values, percentile)
% Extract quantiles from sorted data using linear interpolation

    n_values = numel(sorted_values);

    if n_values == 0
        value = 0;
        return;
    end

    rank_position = 1 + (n_values - 1) * percentile / 100;

    lower_index = max(1, floor(rank_position));
    upper_index = min(n_values, ceil(rank_position));
    interpolation_weight = rank_position - lower_index;

    value = double(gather( ...
        (1 - interpolation_weight) * sorted_values(lower_index) + ...
        interpolation_weight * sorted_values(upper_index)));
end


function y = quad_fiber_forward_gpu(  ...
    x, M_gpu, m, geometry, time_length)
% Four-fiber joint forward operator (GPU)
%
% For the same 3D image x, compute the measurement signals corresponding to the four fibers separately;
% then use the same sampling matrix M to keep the valid scan positions;
% finally stack them vertically into joint measurements of size [n_fiber*m,time_length].

    n_fiber = geometry.n_fiber;
    total_points = geometry.total_points;

    y = gpuArray.zeros( ...
        n_fiber*m, time_length, 'single');

    for fiber_idx = 1:n_fiber

        y_full = forward_das_single_fiber_gpu( ...
            x, geometry, fiber_idx, time_length);

        y_full = reshape( ...
            y_full, total_points, time_length);

        y_sampled = M_gpu * y_full;

        row_start = (fiber_idx-1)*m + 1;
        row_end = fiber_idx*m;

        y(row_start:row_end, :) = y_sampled;
    end
end


function x = quad_fiber_adjoint_gpu( ...
    y, M_gpu, m, geometry, time_length)
% Four-fiber joint adjoint operator (GPU)
%
% Split the joint measurements into the data of the four fibers;
% restore each one to the full scan dimension via M';
% perform single-fiber adjoint reconstruction separately;
% finally add the four adjoint results together to achieve coherent summation.

    n_fiber = geometry.n_fiber;
    sensor_row = geometry.sensor_row;
    sensor_column = geometry.sensor_column;

    x = gpuArray.zeros(geometry.grid_size, 'single');

    for fiber_idx = 1:n_fiber

        row_start = (fiber_idx-1)*m + 1;
        row_end = fiber_idx*m;

        y_compressed = y(row_start:row_end, :);

        y_full = reshape( ...
            M_gpu' * y_compressed, ...
            sensor_row, sensor_column, time_length);

        x_current = adjoint_das_single_fiber_gpu( ...
            y_full, geometry, fiber_idx, time_length);

        x = x + x_current;
    end
end


function y = forward_das_single_fiber_gpu( ...
    x, geometry, fiber_idx, time_length)
% Single-fiber DAS forward operator (GPU)
%
% Physical model:
%   transmit element -> voxel -> receive point of the current fiber
%
% Consistent with the forward model of the original compressed-sensing code;
% no r1*r2 compensation is included.

    persistent accumarray_gpu_supported

    if isempty(accumarray_gpu_supported)
        accumarray_gpu_supported = true;
    end

    if isa(x, 'gpuArray')
        x_gpu = single(x);
    else
        x_gpu = gpuArray(single(x));
    end

    x_flat = x_gpu(:);

    sensor_row = geometry.sensor_row;
    sensor_column = geometry.sensor_column;
    c = geometry.c;
    dt = geometry.dt;

    x_flat_c = geometry.x_flat;
    y_flat_c = geometry.y_flat;
    z_flat_c = geometry.z_flat;

    % Distances from the current fiber to all voxels
    r2_all = geometry.r2_all(:, fiber_idx);

    y_gpu = gpuArray.zeros( ...
        sensor_row, sensor_column, time_length, 'single');

    active_source_indices = ...
        geometry.active_source_indices(:).';

    for source_idx = active_source_indices

        i = geometry.source_index_all(source_idx, 1);
        j = geometry.source_index_all(source_idx, 2);

        source_pos = geometry.source_pos_all(source_idx, :);

        % Distances r1 from the current transmit element to all voxels
        r1_all = sqrt( ...
            (x_flat_c - source_pos(1)).^2 + ...
            (y_flat_c - source_pos(2)).^2 + ...
            (z_flat_c - source_pos(3)).^2);

        % Transmit element -> voxel -> fiber receive point
        R_all = r1_all + r2_all;

        % Sample index corresponding to the propagation time
        time_idx = int32(round( ...
            R_all / (c*dt))) + int32(1);

        valid_idx = ...
            (time_idx >= 1) & (time_idx <= time_length);

        if gather(any(valid_idx(:)))

            current_time_idx = double(time_idx(valid_idx));
            current_values = x_flat(valid_idx);

            if accumarray_gpu_supported

                try
                    % GPU-side accumarray
                    % fill_value must have the same data type as current_values,
                    % otherwise some GPU versions report "the fifth input and output of sum do not have the same class"
                    fill_value = zeros(1, 1, 'like', current_values);

                    current_signal = accumarray( ...
                        current_time_idx(:), ...
                        current_values(:), ...
                        [time_length, 1], ...
                        @sum, ...
                        fill_value);

                catch
                    % If the current MATLAB version does not support GPU accumarray, fall back automatically
                    accumarray_gpu_supported = false;

                    current_time_idx_cpu = gather(current_time_idx(:));
                    current_values_cpu = gather(current_values(:));
                    fill_value_cpu = zeros( ...
                        1, 1, 'like', current_values_cpu);

                    current_signal_cpu = accumarray( ...
                        current_time_idx_cpu, ...
                        current_values_cpu, ...
                        [time_length, 1], ...
                        @sum, ...
                        fill_value_cpu);

                    current_signal = gpuArray(current_signal_cpu);
                end

            else
                % CPU fallback version
                current_time_idx_cpu = gather(current_time_idx(:));
                current_values_cpu = gather(current_values(:));
                fill_value_cpu = zeros( ...
                    1, 1, 'like', current_values_cpu);

                current_signal_cpu = accumarray( ...
                    current_time_idx_cpu, ...
                    current_values_cpu, ...
                    [time_length, 1], ...
                    @sum, ...
                    fill_value_cpu);

                current_signal = gpuArray(current_signal_cpu);
            end

            y_gpu(i, j, :) = reshape( ...
                current_signal, 1, 1, time_length);
        end
    end

    % Stay on the GPU; do not gather
    y = y_gpu;
end


function x_all = adjoint_das_single_fiber_gpu( ...
    sensor_data, geometry, fiber_idx, time_length)
% Single-fiber DAS adjoint operator (GPU)
%
% Physical model:
%   detected signal -> 3D image back-projection
%
% The following from the original code has been removed:
%   complement = sqrt(r1.*r2)
% Therefore, no distance compensation is performed here.

    if isa(sensor_data, 'gpuArray')
        s_data = single(sensor_data);
    else
        s_data = gpuArray(single(sensor_data));
    end

    c = geometry.c;
    dt = geometry.dt;

    x_flat_c = geometry.x_flat;
    y_flat_c = geometry.y_flat;
    z_flat_c = geometry.z_flat;

    % Distances from the current fiber to all voxels
    r2_all = geometry.r2_all(:, fiber_idx);

    % Use flat vectors for GPU-side accumulation
    pa_flat = gpuArray.zeros( ...
        geometry.numel_volume, 1, 'single');

    active_source_indices = ...
        geometry.active_source_indices(:).';

    for source_idx = active_source_indices

        i = geometry.source_index_all(source_idx, 1);
        j = geometry.source_index_all(source_idx, 2);

        % If the current scan position has no signal, skip it directly
        current_channel = s_data(i, j, :);

        if gather(sum(abs(current_channel(:)))) == 0
            continue;
        end

        source_pos = geometry.source_pos_all(source_idx, :);

        % Distances r1 from the transmit elements to all voxels
        r1_all = sqrt( ...
            (x_flat_c - source_pos(1)).^2 + ...
            (y_flat_c - source_pos(2)).^2 + ...
            (z_flat_c - source_pos(3)).^2);

        % Total propagation distance
        R_all = r1_all + r2_all;

        % Time sample index
        time_idx = int32(round( ...
            R_all / (c*dt))) + int32(1);

        valid_idx = ...
            (time_idx >= 1) & (time_idx <= time_length);

        if ~gather(any(valid_idx(:)))
            continue;
        end

        current_time_idx = time_idx(valid_idx);

        current_signal = squeeze( ...
            s_data(i, j, current_time_idx));

        current_signal = current_signal(:);

        % No r1*r2 compensation; direct back-projection accumulation
        pa_flat(valid_idx) = ...
            pa_flat(valid_idx) + current_signal;
    end

    % Stay on the GPU; do not gather
    x_all = reshape( ...
        pa_flat, geometry.grid_size);
end

