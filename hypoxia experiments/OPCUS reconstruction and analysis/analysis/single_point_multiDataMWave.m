
function [mmode_set, info] = ...
    single_point_multiDataMWave(varargin)
%COMPARE_SINGLE_POINT_MMODE_4D_SMOOTHANDINTERP
% Select a point on the XY projection and compare the cylindrical M-waves of 2-6 4D dynamic datasets.
%
% Input data dimensions:
%   Each dataset is [x,y,z,t]; all datasets must have identical sizes.
%
% Workflow:
%   1. Display the YZ, XZ, and XY maximum-intensity projections of the first frame of the first dataset;
%   2. Select a point on the XY projection, or specify [x,y] directly via SelectedPoint;
%   3. Build an XY ROI with the given radius, centered at that point;
%   4. Average the ROI intensity at each z slice to obtain the [z,time] cylindrical M-wave;
%   5. Compute the M-wave at exactly the same position for 2-6 datasets;
%   6. Display the M-waves of all datasets in stacked rows with a common display range and synchronized zooming.
%
% Basic usage:
%   data_sets = {data1,data2,data3,data4,data5,data6};
%
%   [mmode_set, info] = ...
%       compare_single_point_mmode_4d_smoothAndInterp( ...
%       data_sets, 1, ...
%       'FrameRate', 27, ...
%       'TimeScale', 20, ...
%       'SmoothSigma', [0.4,1.8], ...
%       'Labels', {'0min','10min','20min','30min','40min','50min'});
%
% You can also pass 2-6 datasets directly:
%
%   [mmode_set, info] = ...
%       compare_single_point_mmode_4d_smoothAndInterp( ...
%       data1, data2, data3, 1, ...
%       'FrameRate', 27, ...
%       'TimeScale', 20);
%
% If the point position is already known, mouse selection can be skipped:
%
%   [mmode_set, info] = ...
%       compare_single_point_mmode_4d_smoothAndInterp( ...
%       data_sets, 1, ...
%       'SelectedPoint', [64,52]);
%
% Outputs:
%   mmode_set
%       1 x n_datasets cell array.
%       mmode_set{data_index} is the M-wave of the data_index-th dataset,
%       with size [z_output,time_output].
%
%   info
%       Information including the selected point, ROI, raw M-waves, intermediate results, display ranges, and figure handles.
%
% Main parameters:
%   radius
%       XY neighborhood radius around the sampling point.
%       When Neighborhood='square':
%           radius=0: only the sampling point itself is used;
%           radius=1: the sampling point plus its 8-neighborhood, i.e., a 3x3 region.
%
%   'SelectedPoint'
%       1x2 vector [x,y], used to skip interactive point selection.
%
%   'Labels'
%       Dataset labels, e.g., {'0min','10min',...,'50min'}.
%
%   'FrameRate'
%       Frame rate of the volume data; default 27 Hz.
%
%   'TimeScale'
%       Interpolation factor along time; default 20; no interpolation when set to 1.
%
%   'DepthScale'
%       Interpolation factor along z; default 1.
%
%   'SmoothSigma'
%       2D Gaussian smoothing standard deviation of the M-wave, [z direction,time direction]; default [0.4,1.8].
%
%   'ShowPointFigures'
%       Whether to generate the final M-wave comparison figure; default true.
%
%   'ShowIntermediate'
%       Whether to additionally show the raw M-waves and the pre-smoothing M-waves; default false.

%% ==================== 0. Parse Dataset Inputs ====================

[data_sets, radius, remaining_args] = parse_dataset_inputs(varargin);
n_datasets = numel(data_sets);

%% ==================== 1. Parameter Parsing ====================

parser = inputParser;
parser.FunctionName = mfilename;

addParameter(parser, 'SelectedPoint', [], ...
    @(x) isempty(x) || ...
    (isnumeric(x) && numel(x) == 2 && all(isfinite(x(:)))));

addParameter(parser, 'Neighborhood', 'square', @isTextScalar);

addParameter(parser, 'Labels', {}, ...
    @(x) isempty(x) || isstring(x) || ...
    (iscell(x) && all(cellfun(@isTextScalar, x))));

addParameter(parser, 'Label1', [], @isTextScalarOrEmpty);
addParameter(parser, 'Label2', [], @isTextScalarOrEmpty);

addParameter(parser, 'FrameRate', 27, ...
    @(x) isnumeric(x) && isscalar(x) && isfinite(x) && x > 0);

addParameter(parser, 'TimeScale', 20, ...
    @(x) isnumeric(x) && isscalar(x) && isfinite(x) && x >= 1);

addParameter(parser, 'DepthScale', 1, ...
    @(x) isnumeric(x) && isscalar(x) && isfinite(x) && x >= 1);

addParameter(parser, 'InterpMethod', 'pchip', @isTextScalar);

addParameter(parser, 'SmoothSigma', [0.4, 1.8], ...
    @(x) isnumeric(x) && isvector(x) && ...
    (numel(x) == 1 || numel(x) == 2) && ...
    all(isfinite(x)) && all(x >= 0));

addParameter(parser, 'NormalizeEachFrame', false, ...
    @(x) islogical(x) && isscalar(x));

addParameter(parser, 'ShowSelection', true, ...
    @(x) islogical(x) && isscalar(x));

addParameter(parser, 'ShowPointFigures', true, ...
    @(x) islogical(x) && isscalar(x));

addParameter(parser, 'ShowIntermediate', false, ...
    @(x) islogical(x) && isscalar(x));

addParameter(parser, 'ProjectionDisplayPercentile', [0, 100], ...
    @(x) isnumeric(x) && numel(x) == 2 && ...
    all(isfinite(x)) && all(x >= 0) && all(x <= 100) && x(2) > x(1));

addParameter(parser, 'ProjectionCLim', [], ...
    @(x) isempty(x) || ...
    (isnumeric(x) && numel(x) == 2 && ...
    all(isfinite(x)) && x(2) > x(1)));

addParameter(parser, 'MaxSamplesForCLim', 1000000, ...
    @(x) isnumeric(x) && isscalar(x) && ...
    isfinite(x) && x >= 1000);

addParameter(parser, 'MModeDisplayPercentile', [0, 100], ...
    @(x) isnumeric(x) && numel(x) == 2 && ...
    all(isfinite(x)) && all(x >= 0) && all(x <= 100) && x(2) > x(1));

addParameter(parser, 'MModeCLim', [], ...
    @(x) isempty(x) || ...
    (isnumeric(x) && numel(x) == 2 && ...
    all(isfinite(x)) && x(2) > x(1)));

addParameter(parser, 'Colormap', [], ...
    @(x) isempty(x) || ...
    (isnumeric(x) && size(x, 2) == 3));

addParameter(parser, 'ZSpacing', 1, ...
    @(x) isnumeric(x) && isscalar(x) && isfinite(x) && x > 0);

addParameter(parser, 'LinkAxes', true, ...
    @(x) islogical(x) && isscalar(x));

addParameter(parser, 'MarkerSize', 8, ...
    @(x) isnumeric(x) && isscalar(x) && ...
    isfinite(x) && x > 0);

parse(parser, remaining_args{:});
opt = parser.Results;

if isscalar(opt.SmoothSigma)
    opt.SmoothSigma = [opt.SmoothSigma, opt.SmoothSigma];
else
    opt.SmoothSigma = double(opt.SmoothSigma(:).');
end

opt.Neighborhood = lower(strtrim(char(opt.Neighborhood)));
opt.InterpMethod = lower(strtrim(char(opt.InterpMethod)));

if ~ismember(opt.Neighborhood, {'square','disk'})
    error('Neighborhood must be ''square'' or ''disk''.');
end

valid_interp_methods = {'nearest','linear','spline','pchip','makima'};

if ~ismember(opt.InterpMethod, valid_interp_methods)
    error('Unsupported InterpMethod: %s.', opt.InterpMethod);
end

if isempty(opt.Colormap)
    current_colormap = gray(256);
else
    current_colormap = opt.Colormap;
end

%% ==================== 2. Set Dataset Labels ====================

data_labels = arrayfun( ...
    @(index) sprintf('Dataset %d', index), ...
    1:n_datasets, ...
    'UniformOutput', false);

if ~isempty(opt.Labels)

    if isstring(opt.Labels)
        custom_labels = cellstr(opt.Labels);
    else
        custom_labels = opt.Labels;
    end

    if numel(custom_labels) ~= n_datasets
        error('The number of Labels must equal the number of datasets (%d).', n_datasets);
    end

    data_labels = cellfun(@char, custom_labels, ...
        'UniformOutput', false);
end

if ~isempty(opt.Label1)
    data_labels{1} = char(opt.Label1);
end

if n_datasets >= 2 && ~isempty(opt.Label2)
    data_labels{2} = char(opt.Label2);
end

%% ==================== 3. Validate Input Data ====================

if ~(isnumeric(radius) && isscalar(radius) && ...
        isfinite(radius) && radius >= 0)
    error('radius must be a non-negative finite scalar.');
end

reference_size = size(data_sets{1});

for data_index = 1:n_datasets

    current_data = data_sets{data_index};

    if ~(isnumeric(current_data) || islogical(current_data))
        error('Input data #%d must be a numeric or logical array.', data_index);
    end

    if ndims(current_data) ~= 4
        error('Input data #%d must be a 4D array [x,y,z,t].', data_index);
    end

    if ~isreal(current_data)
        error('Input data #%d must be real-valued.', data_index);
    end

    if ~isequal(size(current_data), reference_size)
        error('All 4D datasets must have identical sizes.');
    end
end

nx = reference_size(1);
ny = reference_size(2);
nz = reference_size(3);
n_time = reference_size(4);

if nx < 1 || ny < 1 || nz < 1 || n_time < 1
    error('None of the four dimensions of the input data may be empty.');
end

%% ==================== 4. Display the Three-Direction Projections of the First Frame of the First Dataset ====================

first_frame_1 = data_sets{1}(:, :, :, 1);
first_frame_1 = gather_if_needed(first_frame_1);
first_frame_1 = double(first_frame_1);

yz_mip = squeeze(max(first_frame_1, [], 1));
xz_mip = squeeze(max(first_frame_1, [], 2));
xy_mip = squeeze(max(first_frame_1, [], 3));

if isempty(opt.ProjectionCLim)
    projection_clim = local_percentile_range( ...
        first_frame_1, ...
        opt.ProjectionDisplayPercentile, ...
        opt.MaxSamplesForCLim);
else
    projection_clim = double(opt.ProjectionCLim(:).');
end

if opt.ShowSelection

    selection_figure = figure( ...
        'Color', 'w', ...
        'Name', 'Select the position of the single-point cylindrical M-wave', ...
        'Position', [50, 100, 1550, 500]);

    colormap(selection_figure, current_colormap);

    ax_yz = subplot(1, 3, 1);
    imagesc(ax_yz, 1:ny, 1:nz, yz_mip.');
    axis(ax_yz, 'image');
    set(ax_yz, 'YDir', 'normal');
    caxis(ax_yz, projection_clim);
    colorbar(ax_yz);
    xlabel(ax_yz, 'Y index');
    ylabel(ax_yz, 'Z index');
    title(ax_yz, [data_labels{1}, ': YZ maximum-intensity projection']);

    ax_xz = subplot(1, 3, 2);
    imagesc(ax_xz, 1:nx, 1:nz, xz_mip.');
    axis(ax_xz, 'image');
    set(ax_xz, 'YDir', 'normal');
    caxis(ax_xz, projection_clim);
    colorbar(ax_xz);
    xlabel(ax_xz, 'X index');
    ylabel(ax_xz, 'Z index');
    title(ax_xz, [data_labels{1}, ': XZ maximum-intensity projection']);

    ax_xy = subplot(1, 3, 3);
    imagesc(ax_xy, 1:nx, 1:ny, xy_mip.');
    axis(ax_xy, 'image');
    set(ax_xy, 'YDir', 'normal');
    caxis(ax_xy, projection_clim);
    colorbar(ax_xy);
    xlabel(ax_xy, 'X index');
    ylabel(ax_xy, 'Y index');
    title(ax_xy, 'XY projection: click to select a cylinder center point');

else

    selection_figure = [];
    ax_yz = [];
    ax_xz = [];
    ax_xy = [];

    if isempty(opt.SelectedPoint)
        error(['When ShowSelection is false, the sampling point [x,y] ', ...
            'must be provided via SelectedPoint.']);
    end
end

%% ==================== 5. Acquire a Single Sampling Point ====================

if isempty(opt.SelectedPoint)

    fprintf('\n========================================\n');
    fprintf('Please click a point on the XY projection as the cylinder center.\n');
    fprintf('M-waves will be computed at exactly the same XY position for all datasets.\n');
    fprintf('========================================\n');

    axes(ax_xy); %#ok<LAXES>
    [clicked_x, clicked_y] = ginput(1);

    if numel(clicked_x) ~= 1 || numel(clicked_y) ~= 1
        error('No sampling point was acquired.');
    end

    selected_point_raw = [clicked_x, clicked_y];

else
    selected_point_raw = double(opt.SelectedPoint(:).');
end

selected_point_indices = round(selected_point_raw);
selected_point_indices(1) = min(max(selected_point_indices(1), 1), nx);
selected_point_indices(2) = min(max(selected_point_indices(2), 1), ny);

center_x = selected_point_indices(1);
center_y = selected_point_indices(2);

%% ==================== 6. Mark the Sampling Point in the Three-Direction Projections ====================

if opt.ShowSelection && ~isempty(selection_figure) && ishandle(selection_figure)

    hold(ax_xy, 'on');
    plot(ax_xy, center_x, center_y, 'ro', ...
        'MarkerSize', opt.MarkerSize + 2, ...
        'LineWidth', 2);
    text(ax_xy, center_x, center_y, '1', ...
        'Color', 'r', ...
        'FontSize', 12, ...
        'FontWeight', 'bold', ...
        'HorizontalAlignment', 'center', ...
        'VerticalAlignment', 'bottom');
    title(ax_xy, sprintf( ...
        'XY projection: cylinder center selected at [x=%d, y=%d]', ...
        center_x, center_y));

    % The cylinder spans all slices along z, so its XY position is shown as vertical lines in the XZ/YZ projections.
    hold(ax_xz, 'on');
    plot(ax_xz, [center_x, center_x], [1, nz], ...
        'r--', 'LineWidth', 1.2);

    hold(ax_yz, 'on');
    plot(ax_yz, [center_y, center_y], [1, nz], ...
        'r--', 'LineWidth', 1.2);

    drawnow;
end

%% ==================== 7. Compute the Multi-Dataset Cylindrical M-Waves at the Single Point ====================

time_raw_s = (0:n_time-1).' / opt.FrameRate;
time_raw_s = time_raw_s(:).';

z_raw = (1:nz) * opt.ZSpacing;

[x_range, y_range, roi_mask] = build_xy_roi( ...
    nx, ny, center_x, center_y, radius, opt.Neighborhood);

roi_info = struct( ...
    'center_xy', [center_x, center_y], ...
    'center_xy_raw', selected_point_raw, ...
    'x_range', x_range, ...
    'y_range', y_range, ...
    'mask', roi_mask, ...
    'pixel_count', nnz(roi_mask));

mmode_raw_set = cell(1, n_datasets);
mmode_final_set = cell(1, n_datasets);
mmode_before_smoothing_set = cell(1, n_datasets);
processing_info_set = cell(1, n_datasets);

fprintf('Computing cylindrical M-waves at single-point position [%d,%d] for %d datasets...\n', ...
    center_x, center_y, n_datasets);

for data_index = 1:n_datasets

    mmode_raw_set{data_index} = compute_cylinder_mmode( ...
        data_sets{data_index}, x_range, y_range, roi_mask);

    [current_mmode_final, current_processing] = ...
        postprocess_mmode( ...
        mmode_raw_set{data_index}, ...
        time_raw_s, z_raw, opt);

    mmode_final_set{data_index} = current_mmode_final;
    mmode_before_smoothing_set{data_index} = ...
        current_processing.before_smoothing;
    processing_info_set{data_index} = current_processing;
end

time_output_s = processing_info_set{1}.time_output_s;
z_output = processing_info_set{1}.z_output;

%% ==================== 8. Generate the Single-Point Multi-Dataset M-Wave Comparison Figure ====================

if isempty(opt.MModeCLim)
    final_clim = local_percentile_range( ...
        combine_mmode_values(mmode_final_set), ...
        opt.MModeDisplayPercentile, ...
        opt.MaxSamplesForCLim);
else
    final_clim = double(opt.MModeCLim(:).');
end

intermediate_raw_figure = [];
intermediate_presmooth_figure = [];
point_figure = [];

if opt.ShowIntermediate

    raw_clim = local_percentile_range( ...
        combine_mmode_values(mmode_raw_set), ...
        opt.MModeDisplayPercentile, ...
        opt.MaxSamplesForCLim);

    intermediate_raw_figure = show_mmode_stack( ...
        mmode_raw_set, ...
        time_raw_s, z_raw, ...
        'Single point: raw cylindrical M-waves', ...
        raw_clim, current_colormap, ...
        data_labels, opt.LinkAxes, ...
        [center_x, center_y]);

    presmooth_clim = local_percentile_range( ...
        combine_mmode_values(mmode_before_smoothing_set), ...
        opt.MModeDisplayPercentile, ...
        opt.MaxSamplesForCLim);

    intermediate_presmooth_figure = show_mmode_stack( ...
        mmode_before_smoothing_set, ...
        time_output_s, z_output, ...
        'Single point: M-waves after interpolation, before smoothing', ...
        presmooth_clim, current_colormap, ...
        data_labels, opt.LinkAxes, ...
        [center_x, center_y]);
end

if opt.ShowPointFigures

    point_figure = show_mmode_stack( ...
        mmode_final_set, ...
        time_output_s, z_output, ...
        'Single point: final cylindrical M-waves (interpolation + Gaussian smoothing)', ...
        final_clim, current_colormap, ...
        data_labels, opt.LinkAxes, ...
        [center_x, center_y]);
end

fprintf('Single-point multi-dataset cylindrical M-wave computation completed.\n');

%% ==================== 9. Output Information ====================

info = struct();

info.n_datasets = n_datasets;
info.n_points = 1;
info.data_labels = data_labels;
info.data_size = reference_size;
info.radius = radius;
info.neighborhood = opt.Neighborhood;

info.selected_point_raw = selected_point_raw;
info.selected_point_indices = selected_point_indices;
info.roi_info = roi_info;

info.time_raw_s = time_raw_s;
info.time_output_s = time_output_s;
info.z_raw = z_raw;
info.z_output = z_output;

info.mmode_raw_set = mmode_raw_set;
info.mmode_before_smoothing_set = mmode_before_smoothing_set;
info.mmode_final_set = mmode_final_set;
info.processing_info_set = processing_info_set;

info.projection_clim = projection_clim;
info.final_clim = final_clim;

if opt.ShowSelection && ~isempty(selection_figure) && ...
        ishandle(selection_figure)
    info.selection_figure = selection_figure;
    info.selection_axes_yz = ax_yz;
    info.selection_axes_xz = ax_xz;
    info.selection_axes_xy = ax_xy;
else
    info.selection_figure = [];
    info.selection_axes_yz = [];
    info.selection_axes_xz = [];
    info.selection_axes_xy = [];
end

info.point_figure = point_figure;
info.intermediate_raw_figure = intermediate_raw_figure;
info.intermediate_presmooth_figure = intermediate_presmooth_figure;

info.options = opt;

mmode_set = mmode_final_set;

end

%% =========================================================
% Local functions
%% =========================================================

function [data_sets, radius, remaining_args] = ...
    parse_dataset_inputs(input_args)
% Supports cell input or 2-6 4D datasets passed directly in sequence.

if isempty(input_args)
    error('Please provide at least two 4D datasets and the radius.');
end

if iscell(input_args{1})

    data_sets = input_args{1};

    if numel(input_args) < 2
        error('When datasets are passed as a cell array, the second argument must be the radius.');
    end

    radius = input_args{2};
    remaining_args = input_args(3:end);

else

    dataset_count = 0;
    data_sets = {};

    while dataset_count < numel(input_args)

        current_input = input_args{dataset_count + 1};

        if ~(isnumeric(current_input) || islogical(current_input)) || ...
                ndims(current_input) ~= 4
            break;
        end

        dataset_count = dataset_count + 1;
        data_sets{dataset_count} = current_input; %#ok<AGROW>
    end

    if dataset_count == 0 || dataset_count >= numel(input_args)
        error(['Unable to identify the radius argument. Use data1,...,dataN,radius, ', ...
            'or {data1,...,dataN},radius.']);
    end

    radius = input_args{dataset_count + 1};
    remaining_args = input_args(dataset_count + 2:end);
end

if numel(data_sets) < 2
    error('At least two 4D datasets are required.');
end

if numel(data_sets) > 6
    error('The current function supports comparing at most 6 datasets simultaneously.');
end

end

function [x_range, y_range, roi_mask] = build_xy_roi( ...
    nx, ny, center_x, center_y, radius, neighborhood)
% Build the XY ROI around a single sampling point.

switch neighborhood

    case 'square'
        integer_radius = round(radius);

        x_min = max(1, center_x - integer_radius);
        x_max = min(nx, center_x + integer_radius);
        y_min = max(1, center_y - integer_radius);
        y_max = min(ny, center_y + integer_radius);

        x_range = x_min:x_max;
        y_range = y_min:y_max;
        roi_mask = true(numel(x_range), numel(y_range));

    case 'disk'
        x_min = max(1, floor(center_x - radius));
        x_max = min(nx, ceil(center_x + radius));
        y_min = max(1, floor(center_y - radius));
        y_max = min(ny, ceil(center_y + radius));

        x_range = x_min:x_max;
        y_range = y_min:y_max;

        [x_grid, y_grid] = ndgrid(x_range, y_range);

        roi_mask = ...
            (x_grid - center_x).^2 + ...
            (y_grid - center_y).^2 <= radius^2 + eps;

    otherwise
        error('Unknown Neighborhood type.');
end

if ~any(roi_mask(:))
    error('The ROI of sampling point [%d,%d] contains no valid pixels.', ...
        center_x, center_y);
end

end

function mmode = compute_cylinder_mmode( ...
    data4d, x_range, y_range, roi_mask)
% For each z slice, average within the XY ROI to obtain the [nz,time] M-wave.

nz = size(data4d, 3);
n_time = size(data4d, 4);

mmode = zeros(nz, n_time, 'double');

for z_index = 1:nz

    current_slab = data4d(x_range, y_range, z_index, :);
    current_slab = reshape( ...
        current_slab, numel(roi_mask), n_time);

    current_slab = current_slab(roi_mask(:), :);
    current_slab = gather_if_needed(current_slab);

    mmode(z_index, :) = mean( ...
        double(current_slab), 1, 'omitnan');
end

end

function [m_processed, processing] = postprocess_mmode( ...
    m_raw, time_raw_s, z_raw, opt)
% Process the M-waves in the order "z interpolation -> optional per-frame normalization -> time interpolation -> Gaussian smoothing".

m_stage = double(m_raw);

processing = struct();
processing.raw = m_raw;
processing.time_raw_s = time_raw_s;
processing.z_raw = z_raw;

if opt.DepthScale > 1 && numel(z_raw) >= 2

    n_depth_output = max( ...
        numel(z_raw), round(numel(z_raw) * opt.DepthScale));

    z_stage = linspace( ...
        z_raw(1), z_raw(end), n_depth_output);

    m_stage = fill_nan_for_processing(m_stage);

    m_stage = interp1( ...
        z_raw, m_stage, z_stage, ...
        opt.InterpMethod, 'extrap');

else
    z_stage = z_raw;
end

processing.z_output = z_stage;
processing.after_depth_interpolation = m_stage;

if opt.NormalizeEachFrame
    m_stage = normalize_each_time_frame(m_stage);
end

processing.after_normalization = m_stage;

if opt.TimeScale > 1 && numel(time_raw_s) >= 2

    n_time_output = max( ...
        numel(time_raw_s), ...
        round(numel(time_raw_s) * opt.TimeScale));

    time_output_s = linspace( ...
        time_raw_s(1), time_raw_s(end), n_time_output);

    m_stage = fill_nan_for_processing(m_stage);

    m_stage = interp1( ...
        time_raw_s, m_stage.', time_output_s, ...
        opt.InterpMethod, 'extrap').';

else
    time_output_s = time_raw_s;
end

processing.time_output_s = time_output_s;
processing.after_time_interpolation = m_stage;

if any(opt.SmoothSigma > 0)
    m_stage = fill_nan_for_processing(m_stage);
    m_processed = gaussian_smooth_2d(m_stage, opt.SmoothSigma);
else
    m_processed = m_stage;
end

processing.before_smoothing = m_stage;
processing.final = m_processed;

end

function m_normalized = normalize_each_time_frame(m_input)
% For each time frame, perform min-max normalization along z separately.

m_normalized = zeros(size(m_input), 'double');

for time_index = 1:size(m_input, 2)

    current_frame = double(m_input(:, time_index));
    finite_values = current_frame(isfinite(current_frame));

    if isempty(finite_values)
        m_normalized(:, time_index) = current_frame;
        continue;
    end

    min_value = min(finite_values);
    max_value = max(finite_values);
    value_range = max_value - min_value;

    if value_range <= eps(max(abs([min_value,max_value,1])))
        m_normalized(:, time_index) = 0;
    else
        m_normalized(:, time_index) = ...
            (current_frame - min_value) / value_range;
    end
end

end

function m_filled = fill_nan_for_processing(m_input)
% Fill NaN values before interpolation and smoothing to prevent NaN propagation.

m_filled = double(m_input);

if all(isfinite(m_filled(:)))
    return;
end

try
    m_filled = fillmissing(m_filled, 'nearest', 1);
    m_filled = fillmissing(m_filled, 'nearest', 2);
catch
end

if any(~isfinite(m_filled(:)))
    finite_values = m_filled(isfinite(m_filled));

    if isempty(finite_values)
        fill_value = 0;
    else
        fill_value = median(finite_values);
    end

    m_filled(~isfinite(m_filled)) = fill_value;
end

end

function output = gaussian_smooth_2d(input, sigma)
% Prefer imgaussfilt; fall back to local convolution when the Image Processing Toolbox is unavailable.

sigma = double(sigma(:).');

if isscalar(sigma)
    sigma = [sigma, sigma];
end

if all(sigma <= 0)
    output = double(input);
    return;
end

input = double(input);

if exist('imgaussfilt', 'file') == 2 || ...
        exist('imgaussfilt', 'builtin') == 5

    try
        output = imgaussfilt( ...
            input, sigma, ...
            'Padding', 'replicate', ...
            'FilterDomain', 'spatial');
        return;
    catch
    end
end

output = input;

if sigma(1) > 0
    kernel_depth = gaussian_kernel_1d(sigma(1));
    output = convolve_replicate(output, kernel_depth(:));
end

if sigma(2) > 0
    kernel_time = gaussian_kernel_1d(sigma(2));
    output = convolve_replicate(output, kernel_time(:).');
end

end

function kernel = gaussian_kernel_1d(sigma)
% Generate a normalized 1D Gaussian kernel.

kernel_radius = max(1, ceil(3 * sigma));
kernel_coordinate = -kernel_radius:kernel_radius;

kernel = exp(-kernel_coordinate.^2 / (2 * sigma^2));
kernel = kernel / sum(kernel);

end

function output = convolve_replicate(input, kernel)
% Perform 1D convolution along rows or columns, using replicate padding at the boundaries.

[n_rows, n_columns] = size(input);
pad_width = (numel(kernel) - 1) / 2;

if size(kernel, 1) > 1

    padded = [
        repmat(input(1,:), pad_width, 1);
        input;
        repmat(input(end,:), pad_width, 1)
    ];

    filtered = conv2(padded, kernel, 'same');
    output = filtered( ...
        pad_width + 1:pad_width + n_rows, :);

else

    padded = [
        repmat(input(:,1), 1, pad_width), ...
        input, ...
        repmat(input(:,end), 1, pad_width)
    ];

    filtered = conv2(padded, kernel, 'same');
    output = filtered( ...
        :, pad_width + 1:pad_width + n_columns);
end

end

function current_figure = show_mmode_stack( ...
    mmode_cell, time_s, z_coordinates, ...
    figure_title, color_limits, ...
    current_colormap, data_labels, link_axes_enabled, ...
    center_xy)
% Display the M-waves of all datasets at the same point position in stacked rows.

n_datasets = numel(mmode_cell);
figure_height = min(1000, 180 + 170 * n_datasets);

current_figure = figure( ...
    'Color', 'w', ...
    'Name', figure_title, ...
    'Position', [100, 60, 1150, figure_height]);

colormap(current_figure, current_colormap);

sgtitle(current_figure, figure_title, ...
    'FontSize', 14, ...
    'FontWeight', 'bold');

axes_list = gobjects(1, n_datasets);

for data_index = 1:n_datasets

    current_axes = subplot(n_datasets, 1, data_index);
    axes_list(data_index) = current_axes;

    imagesc( ...
        current_axes, time_s, z_coordinates, ...
        mmode_cell{data_index});

    set(current_axes, 'YDir', 'normal');
    axis(current_axes, 'tight');
    caxis(current_axes, color_limits);
    colorbar(current_axes);

    ylabel(current_axes, 'Z index');

    title(current_axes, sprintf( ...
        '%s: %s, center [%d,%d]', ...
        data_labels{data_index}, ...
        figure_title, ...
        center_xy(1), center_xy(2)));

    if data_index == n_datasets
        xlabel(current_axes, 'Time (s)');
    end
end

if link_axes_enabled && n_datasets > 1
    linkaxes(axes_list, 'xy');
end

drawnow;

end

function combined_values = combine_mmode_values(mmode_cell)
% Merge multiple M-waves to compute the common display range.

combined_values = cellfun( ...
    @(x) x(:), mmode_cell, ...
    'UniformOutput', false);

combined_values = vertcat(combined_values{:});

end

function limits = local_percentile_range( ...
    values, percentiles, max_samples)
% Compute the display range without relying on any toolbox.

values = values(:);

if isempty(values)
    limits = [0, 1];
    return;
end

n_total = numel(values);

if n_total > max_samples
    sample_stride = ceil(n_total / max_samples);
    values = values(1:sample_stride:n_total);
end

values = values(isfinite(values));

if isempty(values)
    limits = [0, 1];
    return;
end

if percentiles(1) == 0 && percentiles(2) == 100

    limits = [ ...
        double(min(values(:))), ...
        double(max(values(:)))];

else

    sorted_values = sort(double(values(:)));

    lower_value = local_percentile( ...
        sorted_values, percentiles(1));

    upper_value = local_percentile( ...
        sorted_values, percentiles(2));

    limits = [lower_value, upper_value];
end

if limits(1) == limits(2)
    padding = max(abs(limits(1)) * 1e-6, 1e-6);
    limits = limits + [-padding, padding];
end

end

function value = local_percentile(sorted_values, percentile)
% Compute quantiles from sorted data.

n_values = numel(sorted_values);

if n_values == 0
    value = 0;
    return;
end

rank_position = ...
    1 + (n_values - 1) * percentile / 100;

lower_index = max(1, floor(rank_position));
upper_index = min(n_values, ceil(rank_position));

interpolation_weight = rank_position - lower_index;

value = ...
    (1 - interpolation_weight) * ...
    sorted_values(lower_index) + ...
    interpolation_weight * ...
    sorted_values(upper_index);

end

function output = gather_if_needed(input)
% If the input is a GPU array, gather it back to the CPU.

if isa(input, 'gpuArray')
    output = gather(input);
else
    output = input;
end

end

function tf = isTextScalar(x)
% Check whether the input is a single text scalar.

tf = ischar(x) || (isstring(x) && isscalar(x));

end

function tf = isTextScalarOrEmpty(x)
% Check whether the input is empty or a single text scalar.

tf = isempty(x) || isTextScalar(x);

end
