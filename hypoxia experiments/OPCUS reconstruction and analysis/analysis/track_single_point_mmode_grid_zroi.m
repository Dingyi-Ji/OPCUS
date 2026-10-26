
function [z_trajectory_set, peak_to_valley_distance, tracking_info] = ...
    track_single_point_mmode_grid_zroi(mmode_set, varargin)
%TRACK_SINGLE_POINT_MMODE_GRID_ZROI
% Extract the z-direction bright-line trajectory from single-point, multi-dataset cylindrical M-waves, and plot
% the peak-to-valley z-distance of each dataset.
%
% Inputs:
%   mmode_set
%       1 x n_datasets or n_datasets x 1 cell array, usually produced
%       directly by compare_single_point_mmode_4d_smoothAndInterp.
%       Each cell is a 2D M-wave matrix [z,time].
%
% Outputs:
%   z_trajectory_set
%       1 x n_datasets cell array; each cell stores a 1 x time z-trajectory.
%
%   peak_to_valley_distance
%       1 x n_datasets vector; each element is the peak-to-valley z-distance of the corresponding dataset.
%       By default the raw definition is used: max(z trajectory)-min(z trajectory).
%
%   tracking_info
%       Structure containing the ROI, original trajectories, smoothed trajectories, peak/valley positions, peak/valley times,
%       figure handles, and all parameters.
%
% Typical usage:
%   [mmode_set, mmode_info] = ...
%       compare_single_point_mmode_4d_smoothAndInterp( ...
%       data_sets, 1, 'Labels', {'0min','10min','20min'});
%
%   [z_tracks, dz_distance, track_info] = ...
%       track_single_point_mmode_grid_zroi( ...
%       mmode_set, ...
%       'Labels', mmode_info.data_labels, ...
%       'Time', mmode_info.time_output_s, ...
%       'ZCoordinates', mmode_info.z_output, ...
%       'ZRange', [60,150], ...
%       'ZRangeUnits', 'index', ...
%       'TopPercent', 5, ...
%       'WeightPower', 1);
%
% If ZRange is not specified and SelectZROI=true, the program displays the M-waves of all datasets;
% click twice on any one of the subplots to select the z-direction ROI shared by all datasets.
%
% Bright-line tracking method:
%   1. At each time point, only pixels within the z ROI are used;
%   2. The background at that time point is estimated with BackgroundPercentile;
%   3. Among the background-subtracted positive values, the top TopPercent brightest pixels are selected;
%   4. The brightness-weighted average of the z-coordinates of the selected pixels gives the z position at the current time point.
%
% This version does not include the 9-point peak-to-peak matrix, the spatial 3x3 matrix, or the deformation-rate matrix.

%% ==================== 1. Input Validation ====================

if nargin < 1
    error(['Please provide a single-point multi-dataset M-wave cell array. Example: ', ...
        'track_single_point_mmode_grid_zroi(mmode_set).']);
end

if ~iscell(mmode_set) || isempty(mmode_set)
    error('mmode_set must be a non-empty cell array.');
end

if ~(isvector(mmode_set) && numel(mmode_set) >= 1 && ...
        numel(mmode_set) <= 6)
    error('mmode_set must be a vector cell array containing 1-6 M-waves.');
end

mmode_set = mmode_set(:).';
n_datasets = numel(mmode_set);
reference_size = [];

for data_index = 1:n_datasets

    current_mmode = mmode_set{data_index};

    if isa(current_mmode, 'gpuArray')
        current_mmode = gather(current_mmode);
    end

    if ~isnumeric(current_mmode) || ndims(current_mmode) ~= 2 || ...
            isempty(current_mmode)
        error('mmode_set{%d} must be a non-empty 2D numeric matrix [z,time].', ...
            data_index);
    end

    if ~isreal(current_mmode)
        error('mmode_set{%d} must be a real-valued matrix.', data_index);
    end

    current_mmode = double(current_mmode);
    current_mmode(~isfinite(current_mmode)) = NaN;
    mmode_set{data_index} = current_mmode;

    if isempty(reference_size)
        reference_size = size(current_mmode);
    elseif ~isequal(size(current_mmode), reference_size)
        error(['All M-wave matrices must have identical sizes. ', ...
            'The size of mmode_set{%d} differs from the first M-wave.'], data_index);
    end
end

nz = reference_size(1);
n_time = reference_size(2);

%% ==================== 2. Optional Parameters ====================

parser = inputParser;
parser.FunctionName = mfilename;

addParameter(parser, 'Labels', {}, ...
    @(x) isempty(x) || isstring(x) || ...
    (iscell(x) && all(cellfun(@isTextScalar, x))));

addParameter(parser, 'ZRange', [], ...
    @(x) isempty(x) || ...
    (isnumeric(x) && numel(x) == 2 && all(isfinite(x))));

addParameter(parser, 'ZRangeUnits', 'index', @isTextScalar);

addParameter(parser, 'ZCoordinates', [], ...
    @(x) isempty(x) || ...
    (isnumeric(x) && isvector(x) && numel(x) == nz && ...
    all(isfinite(x))));

addParameter(parser, 'Time', [], ...
    @(x) isempty(x) || ...
    (isnumeric(x) && isvector(x) && numel(x) == n_time && ...
    all(isfinite(x))));

addParameter(parser, 'FrameRate', [], ...
    @(x) isempty(x) || ...
    (isnumeric(x) && isscalar(x) && isfinite(x) && x > 0));

addParameter(parser, 'TimeAxisLabel', '', @isTextScalar);

addParameter(parser, 'SelectZROI', true, ...
    @(x) islogical(x) && isscalar(x));

addParameter(parser, 'SelectionDataIndex', 1, ...
    @(x) isnumeric(x) && isscalar(x) && isfinite(x) && ...
    x >= 1 && x <= n_datasets && x == round(x));

addParameter(parser, 'TopPercent', 5, ...
    @(x) isnumeric(x) && isscalar(x) && isfinite(x) && ...
    x > 0 && x <= 100);

addParameter(parser, 'BackgroundPercentile', 20, ...
    @(x) isnumeric(x) && isscalar(x) && isfinite(x) && ...
    x >= 0 && x <= 100);

addParameter(parser, 'WeightPower', 1, ...
    @(x) isnumeric(x) && isscalar(x) && isfinite(x) && x >= 0);

addParameter(parser, 'MinPositivePixels', 1, ...
    @(x) isnumeric(x) && isscalar(x) && isfinite(x) && ...
    x >= 1 && x == round(x));

addParameter(parser, 'MaxStep', inf, ...
    @(x) isnumeric(x) && isscalar(x) && ...
    (isinf(x) || (isfinite(x) && x > 0)));

addParameter(parser, 'SmoothWindow', 1, ...
    @(x) isnumeric(x) && isscalar(x) && isfinite(x) && ...
    x >= 1 && x == round(x));

addParameter(parser, 'SmoothMethod', 'median', @isTextScalar);

addParameter(parser, 'PeakToValleyMetric', 'raw', @isTextScalar);

addParameter(parser, 'MModeDisplayPercentile', [0,100], ...
    @(x) isnumeric(x) && numel(x) == 2 && all(isfinite(x)) && ...
    all(x >= 0) && all(x <= 100) && x(1) < x(2));

addParameter(parser, 'CLim', [], ...
    @(x) isempty(x) || ...
    (isnumeric(x) && numel(x) == 2 && all(isfinite(x)) && ...
    x(1) < x(2)));

addParameter(parser, 'MaxSamplesForCLim', 1000000, ...
    @(x) isnumeric(x) && isscalar(x) && isfinite(x) && x >= 1000);

addParameter(parser, 'Colormap', [], ...
    @(x) isempty(x) || ...
    (isnumeric(x) && size(x,2) == 3 && all(isfinite(x(:)))));

addParameter(parser, 'ShowRawTrajectory', false, ...
    @(x) islogical(x) && isscalar(x));

addParameter(parser, 'ShowTrackingFigure', true, ...
    @(x) islogical(x) && isscalar(x));

addParameter(parser, 'ShowSummaryPlot', true, ...
    @(x) islogical(x) && isscalar(x));

addParameter(parser, 'ShowValues', true, ...
    @(x) islogical(x) && isscalar(x));

addParameter(parser, 'LinkAxes', true, ...
    @(x) islogical(x) && isscalar(x));

addParameter(parser, 'Verbose', true, ...
    @(x) islogical(x) && isscalar(x));

parse(parser, varargin{:});
opt = parser.Results;

opt.ZRangeUnits = lower(strtrim(char(opt.ZRangeUnits)));
opt.SmoothMethod = lower(strtrim(char(opt.SmoothMethod)));
opt.PeakToValleyMetric = lower(strtrim(char(opt.PeakToValleyMetric)));

if ~ismember(opt.ZRangeUnits, {'index','coordinate'})
    error('ZRangeUnits must be ''index'' or ''coordinate''.');
end

if ~ismember(opt.SmoothMethod, {'median','mean'})
    error('SmoothMethod must be ''median'' or ''mean''.');
end

if ~ismember(opt.PeakToValleyMetric, {'raw','robust'})
    error('PeakToValleyMetric must be ''raw'' or ''robust''.');
end

if isempty(opt.Labels)
    data_labels = arrayfun(@(index) sprintf('Dataset %d', index), ...
        1:n_datasets, 'UniformOutput', false);
else
    if isstring(opt.Labels)
        data_labels = cellstr(opt.Labels);
    else
        data_labels = opt.Labels;
    end

    if numel(data_labels) ~= n_datasets
        error('The number of Labels must equal the number of datasets (%d).', n_datasets);
    end

    data_labels = cellfun(@char, data_labels, ...
        'UniformOutput', false);
end

%% ==================== 3. Z Coordinates and Time Coordinates ====================

if isempty(opt.ZCoordinates)
    z_coordinates = (1:nz).';
    z_axis_label = 'z index';
    z_unit_text = 'pixels';
else
    z_coordinates = double(opt.ZCoordinates(:));

    if any(diff(z_coordinates) <= 0)
        error('ZCoordinates must be strictly increasing.');
    end

    z_axis_label = 'z coordinate';
    z_unit_text = 'coordinate units';
end

if ~isempty(opt.Time)
    time_values = double(opt.Time(:).');

    if numel(time_values) > 1 && any(diff(time_values) <= 0)
        error('Time must be strictly increasing.');
    end

    if strlength(string(opt.TimeAxisLabel)) > 0
        time_axis_label = char(opt.TimeAxisLabel);
    else
        time_axis_label = 'Time';
    end

elseif ~isempty(opt.FrameRate)
    time_values = (0:n_time-1) / double(opt.FrameRate);

    if strlength(string(opt.TimeAxisLabel)) > 0
        time_axis_label = char(opt.TimeAxisLabel);
    else
        time_axis_label = 'Time (s)';
    end

else
    time_values = 1:n_time;

    if strlength(string(opt.TimeAxisLabel)) > 0
        time_axis_label = char(opt.TimeAxisLabel);
    else
        time_axis_label = 'Frame';
    end
end

%% ==================== 4. Select or Read the Common z-Direction ROI ====================

selection_figure = [];

if ~isempty(opt.ZRange)

    z_range_input = sort(double(opt.ZRange(:).'));

    switch opt.ZRangeUnits

        case 'index'
            z_index_min = max(1, ceil(z_range_input(1)));
            z_index_max = min(nz, floor(z_range_input(2)));

            if z_index_min > z_index_max
                error('The ZRange index range exceeds the z dimension of the M-wave, or the range is empty.');
            end

            roi_mask = false(nz,1);
            roi_mask(z_index_min:z_index_max) = true;

        case 'coordinate'
            roi_mask = z_coordinates >= z_range_input(1) & ...
                z_coordinates <= z_range_input(2);

            if ~any(roi_mask)
                error('The ZRange coordinate range contains no z sampling points.');
            end
    end

elseif opt.SelectZROI

    fprintf(['Please click twice on an M-wave panel to select the ', ...
        'lower and upper bounds of the z-direction ROI shared by all datasets.\n']);

    selection_figure = figure( ...
        'Color', 'w', ...
        'Name', 'Select the z-direction ROI of the single-point M-waves', ...
        'Position', [70,70,1200, min(900,180+150*n_datasets)]);

    selection_axes = gobjects(1,n_datasets);
    selection_values = combine_sampled_values( ...
        mmode_set, opt.MaxSamplesForCLim);
    selection_limits = compute_display_limits( ...
        selection_values, opt.MModeDisplayPercentile, opt.CLim);

    for data_index = 1:n_datasets

        current_axes = subplot(n_datasets,1,data_index);
        selection_axes(data_index) = current_axes;

        display_mmode_image( ...
            current_axes, mmode_set{data_index}, ...
            time_values, z_coordinates, selection_limits, ...
            opt.Colormap, time_axis_label, z_axis_label);

        title(current_axes, data_labels{data_index}, ...
            'Interpreter', 'none');
    end

    if n_datasets > 1
        linkaxes(selection_axes, 'xy');
    end

    sgtitle(selection_figure, ...
        'Click twice to select the common z-direction ROI (shared by all datasets)', ...
        'Interpreter', 'none');
    drawnow;

    try
        [~, y_clicked] = ginput(2);
    catch selection_error
        error('z-direction ROI selection was cancelled: %s', selection_error.message);
    end

    if numel(y_clicked) < 2
        error('Two clicks are required to define the z-direction ROI.');
    end

    selected_z_range = sort(y_clicked(1:2));
    roi_mask = z_coordinates >= selected_z_range(1) & ...
        z_coordinates <= selected_z_range(2);

    if ~any(roi_mask)
        error('The selected z-direction ROI contains no z sampling points.');
    end

    if isgraphics(selection_figure)
        close(selection_figure);
    end

else
    roi_mask = true(nz,1);
end

roi_indices = find(roi_mask);
z_index_range = [min(roi_indices), max(roi_indices)];
z_coordinate_range = [ ...
    min(z_coordinates(roi_mask)), ...
    max(z_coordinates(roi_mask))];

%% ==================== 5. Extract the z-Trajectory and Peak-to-Valley Distance of Each Dataset ====================

z_trajectory_set = cell(1,n_datasets);
track_info_set = cell(1,n_datasets);

raw_peak_to_valley = nan(1,n_datasets);
robust_peak_to_valley = nan(1,n_datasets);
valid_time_count = zeros(1,n_datasets);
peak_z = nan(1,n_datasets);
valley_z = nan(1,n_datasets);
peak_time = nan(1,n_datasets);
valley_time = nan(1,n_datasets);

for data_index = 1:n_datasets

    [current_trajectory, current_info] = track_one_mmode( ...
        mmode_set{data_index}, z_coordinates, roi_mask, ...
        opt, time_values);

    z_trajectory_set{data_index} = current_trajectory;
    track_info_set{data_index} = current_info;

    raw_peak_to_valley(data_index) = ...
        current_info.peak_to_valley_raw;
    robust_peak_to_valley(data_index) = ...
        current_info.peak_to_valley_robust;
    valid_time_count(data_index) = current_info.valid_time_count;
    peak_z(data_index) = current_info.peak_z;
    valley_z(data_index) = current_info.valley_z;
    peak_time(data_index) = current_info.peak_time;
    valley_time(data_index) = current_info.valley_time;
end

switch opt.PeakToValleyMetric

    case 'raw'
        peak_to_valley_distance = raw_peak_to_valley;
        metric_label = 'raw peak-to-valley distance';
        display_peak_z = peak_z;
        display_valley_z = valley_z;

    case 'robust'
        peak_to_valley_distance = robust_peak_to_valley;
        metric_label = 'P95-P5 robust peak-to-valley distance';
        display_peak_z = arrayfun(@(info) info.robust_p95_z, ...
            [track_info_set{:}]);
        display_valley_z = arrayfun(@(info) info.robust_p5_z, ...
            [track_info_set{:}]);
end

%% ==================== 6. Plot the Trajectories and ROI ====================

tracking_figure = [];

if opt.ShowTrackingFigure

    tracking_figure = plot_tracking_stack( ...
        mmode_set, z_trajectory_set, track_info_set, ...
        data_labels, time_values, z_coordinates, ...
        z_coordinate_range, display_peak_z, display_valley_z, ...
        peak_to_valley_distance, metric_label, z_unit_text, ...
        opt, time_axis_label, z_axis_label);
end

%% ==================== 7. Plot the Peak-to-Valley z-Distance of Each Dataset ====================

summary_figure = [];

if opt.ShowSummaryPlot

    summary_figure = plot_peak_to_valley_summary( ...
        peak_to_valley_distance, data_labels, ...
        metric_label, z_unit_text, opt);
end

%% ==================== 8. Output Information ====================

tracking_info = struct();
tracking_info.n_datasets = n_datasets;
tracking_info.data_labels = data_labels;
tracking_info.z_coordinates = z_coordinates;
tracking_info.time_values = time_values;
tracking_info.roi_mask = roi_mask;
tracking_info.roi_indices = roi_indices;
tracking_info.z_index_range = z_index_range;
tracking_info.z_coordinate_range = z_coordinate_range;
tracking_info.track_info_set = track_info_set;
tracking_info.raw_peak_to_valley = raw_peak_to_valley;
tracking_info.robust_peak_to_valley = robust_peak_to_valley;
tracking_info.peak_to_valley_metric = opt.PeakToValleyMetric;
tracking_info.metric_label = metric_label;
tracking_info.valid_time_count = valid_time_count;
tracking_info.peak_z = peak_z;
tracking_info.valley_z = valley_z;
tracking_info.peak_time = peak_time;
tracking_info.valley_time = valley_time;
tracking_info.selection_figure = selection_figure;
tracking_info.tracking_figure = tracking_figure;
tracking_info.summary_figure = summary_figure;
tracking_info.options = opt;

if opt.Verbose
    fprintf('Single-point multi-dataset M-wave z-trajectory extraction completed.\n');
    fprintf('Common z-direction ROI index range: [%d, %d]\n', ...
        z_index_range(1), z_index_range(2));
    fprintf('Common z-direction ROI coordinate range: [%g, %g]\n', ...
        z_coordinate_range(1), z_coordinate_range(2));
    fprintf('Peak-to-valley distance metric: %s\n', metric_label);

    for data_index = 1:n_datasets
        fprintf('%s: peak-to-valley z-distance = %.6g %s\n', ...
            data_labels{data_index}, ...
            peak_to_valley_distance(data_index), z_unit_text);
    end
end

end

%% =========================================================
% Local function: single M-wave trajectory extraction
%% =========================================================

function [z_trajectory, single_info] = track_one_mmode( ...
    mmode, z_coordinates, roi_mask, opt, time_values)

n_time = size(mmode,2);

[z_raw, per_time] = compute_brightline_trajectory( ...
    mmode, z_coordinates, roi_mask, opt);

z_continuity = z_raw;
continuity_corrected = false(1,n_time);

if isfinite(opt.MaxStep)

    last_valid_index = find(isfinite(z_continuity),1,'first');

    if ~isempty(last_valid_index)

        for time_index = last_valid_index+1:n_time

            if ~isfinite(z_raw(time_index))
                continue;
            end

            previous_z = z_continuity(last_valid_index);

            if abs(z_raw(time_index)-previous_z) > opt.MaxStep

                local_mask = roi_mask & ...
                    abs(z_coordinates-previous_z) <= opt.MaxStep;

                [z_candidate, ~] = compute_brightline_trajectory( ...
                    mmode(:,time_index), ...
                    z_coordinates, local_mask, opt);

                continuity_corrected(time_index) = true;

                if isfinite(z_candidate)
                    z_continuity(time_index) = z_candidate;
                else
                    z_continuity(time_index) = NaN;
                end
            end

            if isfinite(z_continuity(time_index))
                last_valid_index = time_index;
            end
        end
    end
end

z_trajectory = z_continuity;
smooth_window = min(round(opt.SmoothWindow), n_time);

if smooth_window > 1

    if mod(smooth_window,2) == 0
        smooth_window = smooth_window + 1;
    end

    smooth_window = min(smooth_window,n_time);

    switch opt.SmoothMethod

        case 'median'
            z_trajectory = movmedian( ...
                z_continuity, smooth_window, 'omitnan');

        case 'mean'
            z_trajectory = movmean( ...
                z_continuity, smooth_window, 'omitnan');
    end
end

finite_trajectory = z_trajectory(isfinite(z_trajectory));

if isempty(finite_trajectory)
    peak_to_valley_raw_value = NaN;
    peak_to_valley_robust_value = NaN;
    peak_z_value = NaN;
    valley_z_value = NaN;
    robust_p95_value = NaN;
    robust_p5_value = NaN;
    peak_time_value = NaN;
    valley_time_value = NaN;
else
    peak_z_value = max(finite_trajectory);
    valley_z_value = min(finite_trajectory);
    peak_to_valley_raw_value = peak_z_value - valley_z_value;

    robust_p95_value = local_percentile(finite_trajectory,95);
    robust_p5_value = local_percentile(finite_trajectory,5);
    peak_to_valley_robust_value = robust_p95_value - robust_p5_value;

    [~, peak_index] = max(z_trajectory, [], 'omitnan');
    [~, valley_index] = min(z_trajectory, [], 'omitnan');

    peak_time_value = time_values(peak_index);
    valley_time_value = time_values(valley_index);
end

single_info = struct();
single_info.z_raw = z_raw;
single_info.z_continuity = z_continuity;
single_info.z_trajectory = z_trajectory;
single_info.continuity_corrected = continuity_corrected;
single_info.selected_count = per_time.selected_count;
single_info.background_value = per_time.background_value;
single_info.threshold_value = per_time.threshold_value;
single_info.weight_sum = per_time.weight_sum;
single_info.selected_z_min = per_time.selected_z_min;
single_info.selected_z_max = per_time.selected_z_max;
single_info.peak_to_valley_raw = peak_to_valley_raw_value;
single_info.peak_to_valley_robust = peak_to_valley_robust_value;
single_info.peak_z = peak_z_value;
single_info.valley_z = valley_z_value;
single_info.robust_p95_z = robust_p95_value;
single_info.robust_p5_z = robust_p5_value;
single_info.peak_time = peak_time_value;
single_info.valley_time = valley_time_value;
single_info.valid_time_count = nnz(isfinite(z_trajectory));
single_info.smooth_window = smooth_window;

end

function [z_estimated, per_time] = compute_brightline_trajectory( ...
    mmode, z_coordinates, roi_mask, opt)

n_time = size(mmode,2);
z_estimated = nan(1,n_time);

per_time = struct();
per_time.selected_count = zeros(1,n_time);
per_time.background_value = nan(1,n_time);
per_time.threshold_value = nan(1,n_time);
per_time.weight_sum = nan(1,n_time);
per_time.selected_z_min = nan(1,n_time);
per_time.selected_z_max = nan(1,n_time);

for time_index = 1:n_time

    [current_z, current_stats] = estimate_single_time_position( ...
        mmode(:,time_index), z_coordinates, roi_mask, opt);

    z_estimated(time_index) = current_z;
    per_time.selected_count(time_index) = current_stats.selected_count;
    per_time.background_value(time_index) = current_stats.background_value;
    per_time.threshold_value(time_index) = current_stats.threshold_value;
    per_time.weight_sum(time_index) = current_stats.weight_sum;
    per_time.selected_z_min(time_index) = current_stats.selected_z_min;
    per_time.selected_z_max(time_index) = current_stats.selected_z_max;
end

end

function [z_estimated, stats] = estimate_single_time_position( ...
    column_values, z_coordinates, roi_mask, opt)

stats = struct();
stats.selected_count = 0;
stats.background_value = NaN;
stats.threshold_value = NaN;
stats.weight_sum = NaN;
stats.selected_z_min = NaN;
stats.selected_z_max = NaN;

z_estimated = NaN;
valid_mask = roi_mask & isfinite(column_values);

if nnz(valid_mask) < opt.MinPositivePixels
    return;
end

valid_values = double(column_values(valid_mask));
valid_z = z_coordinates(valid_mask);

background_value = local_percentile( ...
    valid_values, opt.BackgroundPercentile);

residual_values = valid_values - background_value;
residual_values(~isfinite(residual_values)) = 0;
residual_values = max(residual_values,0);

if nnz(residual_values > 0) < opt.MinPositivePixels
    stats.background_value = background_value;
    return;
end

n_selected = max(1, ...
    ceil(numel(residual_values) * opt.TopPercent / 100));
n_selected = min(n_selected, numel(residual_values));

[~, sort_order] = sort(residual_values,'descend');
positive_order = sort_order(residual_values(sort_order) > 0);

if isempty(positive_order)
    stats.background_value = background_value;
    return;
end

selected_local_indices = positive_order( ...
    1:min(n_selected,numel(positive_order)));

if numel(selected_local_indices) < opt.MinPositivePixels
    stats.background_value = background_value;
    return;
end

selected_z = valid_z(selected_local_indices);
selected_residual = residual_values(selected_local_indices);

if opt.WeightPower == 0
    selected_weights = ones(size(selected_residual));
else
    normalized_residual = selected_residual / max(selected_residual);
    selected_weights = normalized_residual .^ opt.WeightPower;
end

weight_sum = sum(selected_weights);

if ~isfinite(weight_sum) || weight_sum <= 0
    stats.background_value = background_value;
    return;
end

z_estimated = sum(selected_z .* selected_weights) / weight_sum;

stats.selected_count = numel(selected_local_indices);
stats.background_value = background_value;
stats.threshold_value = min(selected_residual);
stats.weight_sum = weight_sum;
stats.selected_z_min = min(selected_z);
stats.selected_z_max = max(selected_z);

end

%% =========================================================
% Local function: plotting
%% =========================================================

function tracking_figure = plot_tracking_stack( ...
    mmode_set, z_trajectory_set, track_info_set, ...
    data_labels, time_values, z_coordinates, ...
    z_coordinate_range, display_peak_z, display_valley_z, ...
    peak_to_valley_distance, metric_label, z_unit_text, ...
    opt, time_axis_label, z_axis_label)

n_datasets = numel(mmode_set);
figure_height = min(1000, 180 + 180*n_datasets);

tracking_figure = figure( ...
    'Color','w', ...
    'Name','Single-point multi-dataset M-wave trajectories', ...
    'Position',[80,60,1200,figure_height]);

all_values = combine_sampled_values( ...
    mmode_set, opt.MaxSamplesForCLim);
display_limits = compute_display_limits( ...
    all_values, opt.MModeDisplayPercentile, opt.CLim);

axes_list = gobjects(1,n_datasets);

for data_index = 1:n_datasets

    current_axes = subplot(n_datasets,1,data_index);
    axes_list(data_index) = current_axes;

    display_mmode_image( ...
        current_axes, mmode_set{data_index}, ...
        time_values, z_coordinates, display_limits, ...
        opt.Colormap, time_axis_label, z_axis_label);

    hold(current_axes,'on');

    plot(current_axes, time_values([1,end]), ...
        z_coordinate_range([1,1]), ...
        'c--','LineWidth',1.2);
    plot(current_axes, time_values([1,end]), ...
        z_coordinate_range([2,2]), ...
        'c--','LineWidth',1.2);

    if opt.ShowRawTrajectory
        plot(current_axes, time_values, ...
            track_info_set{data_index}.z_raw, ...
            '-','Color',[0.65,0.65,0.65], ...
            'LineWidth',0.8);
    end

    plot(current_axes, time_values, ...
        z_trajectory_set{data_index}, ...
        'r-','LineWidth',1.6);

    if isfinite(display_peak_z(data_index))
        plot(current_axes, time_values([1,end]), ...
            display_peak_z(data_index)*[1,1], ...
            ':','Color',[0.95,0.65,0.10], ...
            'LineWidth',1.4);
    end

    if isfinite(display_valley_z(data_index))
        plot(current_axes, time_values([1,end]), ...
            display_valley_z(data_index)*[1,1], ...
            ':','Color',[0.10,0.65,0.30], ...
            'LineWidth',1.4);
    end

    if strcmp(opt.PeakToValleyMetric,'raw')
        if isfinite(track_info_set{data_index}.peak_time) && ...
                isfinite(display_peak_z(data_index))
            plot(current_axes, ...
                track_info_set{data_index}.peak_time, ...
                display_peak_z(data_index), ...
                'o','MarkerSize',6, ...
                'MarkerFaceColor',[0.95,0.65,0.10], ...
                'MarkerEdgeColor','k');
        end

        if isfinite(track_info_set{data_index}.valley_time) && ...
                isfinite(display_valley_z(data_index))
            plot(current_axes, ...
                track_info_set{data_index}.valley_time, ...
                display_valley_z(data_index), ...
                's','MarkerSize',6, ...
                'MarkerFaceColor',[0.10,0.65,0.30], ...
                'MarkerEdgeColor','k');
        end
    end

    if isfinite(peak_to_valley_distance(data_index))
        distance_text = sprintf('Δz = %.4g %s', ...
            peak_to_valley_distance(data_index), z_unit_text);
    else
        distance_text = 'Δz = NaN';
    end

    title(current_axes, sprintf('%s: %s, %s', ...
        data_labels{data_index}, metric_label, distance_text), ...
        'Interpreter','none');

    if data_index == 1

        if opt.ShowRawTrajectory && ...
                strcmp(opt.PeakToValleyMetric,'raw')
            legend(current_axes, ...
                {'ROI lower bound','ROI upper bound','Original trajectory','Final trajectory', ...
                'Peak','Valley'}, ...
                'Location','best');
        elseif opt.ShowRawTrajectory
            legend(current_axes, ...
                {'ROI lower bound','ROI upper bound','Original trajectory','Final trajectory', ...
                'P95','P5'}, ...
                'Location','best');
        elseif strcmp(opt.PeakToValleyMetric,'raw')
            legend(current_axes, ...
                {'ROI lower bound','ROI upper bound','Final trajectory','Peak','Valley'}, ...
                'Location','best');
        else
            legend(current_axes, ...
                {'ROI lower bound','ROI upper bound','Final trajectory','P95','P5'}, ...
                'Location','best');
        end
    end
end

if opt.LinkAxes && n_datasets > 1
    linkaxes(axes_list,'xy');
end

sgtitle(tracking_figure, ...
    sprintf('Single-point M-wave bright-line trajectories and %s', metric_label), ...
    'Interpreter','none');
drawnow;

end

function summary_figure = plot_peak_to_valley_summary( ...
    peak_to_valley_distance, data_labels, metric_label, ...
    z_unit_text, opt)

n_datasets = numel(peak_to_valley_distance);

summary_figure = figure( ...
    'Color','w', ...
    'Name','Single-point peak-to-valley z-distance of each dataset', ...
    'Position',[150,150,850,500]);

current_axes = axes('Parent',summary_figure);
hold(current_axes,'on');

plot(current_axes, 1:n_datasets, peak_to_valley_distance, ...
    '-o', ...
    'LineWidth',1.8, ...
    'MarkerSize',8, ...
    'MarkerFaceColor',[0.20,0.45,0.85], ...
    'Color',[0.20,0.45,0.85]);

finite_values = peak_to_valley_distance( ...
    isfinite(peak_to_valley_distance));

if ~isempty(finite_values)
    y_min = min(finite_values);
    y_max = max(finite_values);

    if y_min == y_max
        y_padding = max(abs(y_min)*0.05,1);
    else
        y_padding = (y_max-y_min)*0.12;
    end

    ylim(current_axes,[y_min-y_padding,y_max+y_padding]);
end

if opt.ShowValues

    for data_index = 1:n_datasets

        if ~isfinite(peak_to_valley_distance(data_index))
            continue;
        end

        text(current_axes, data_index, ...
            peak_to_valley_distance(data_index), ...
            sprintf('  %.4g', peak_to_valley_distance(data_index)), ...
            'HorizontalAlignment','left', ...
            'VerticalAlignment','middle', ...
            'FontWeight','bold');
    end
end

xlim(current_axes,[0.5,n_datasets+0.5]);
xticks(current_axes,1:n_datasets);
xticklabels(current_axes,data_labels);
xtickangle(current_axes,0);
xlabel(current_axes,'Dataset');
ylabel(current_axes,sprintf('Peak-to-valley z-distance (%s)',z_unit_text));
title(current_axes,sprintf('%s of the single-point M-waves',metric_label), ...
    'Interpreter','none');
grid(current_axes,'on');
box(current_axes,'on');
drawnow;

end

function display_mmode_image( ...
    current_axes, mmode, time_values, z_coordinates, ...
    display_limits, current_colormap, ...
    time_axis_label, z_axis_label)

imagesc(current_axes,time_values,z_coordinates,mmode);
set(current_axes,'YDir','normal');
axis(current_axes,'tight');
caxis(current_axes,display_limits);

if isempty(current_colormap)
    colormap(current_axes,gray(256));
else
    colormap(current_axes,current_colormap);
end

colorbar(current_axes);
xlabel(current_axes,time_axis_label);
ylabel(current_axes,z_axis_label);

end

%% =========================================================
% Local function: display range and general utilities
%% =========================================================

function display_limits = compute_display_limits( ...
    values, display_percentile, fixed_limits)

if ~isempty(fixed_limits)
    display_limits = double(fixed_limits(:).');
    return;
end

values = double(values(:));
values = values(isfinite(values));

if isempty(values)
    display_limits = [0,1];
    return;
end

lower_limit = local_percentile(values,display_percentile(1));
upper_limit = local_percentile(values,display_percentile(2));

if ~isfinite(lower_limit) || ~isfinite(upper_limit)
    display_limits = [0,1];
elseif lower_limit == upper_limit
    delta = max(abs(lower_limit))*0.01;

    if delta == 0
        delta = 1;
    end

    display_limits = [lower_limit-delta,upper_limit+delta];
else
    display_limits = [lower_limit,upper_limit];
end

end

function sampled_values = combine_sampled_values( ...
    mmode_cell, max_samples)

total_values = sum(cellfun(@numel,mmode_cell));
sample_step = max(1,ceil(total_values/max_samples));
sampled_values = [];

for cell_index = 1:numel(mmode_cell)

    current_values = double(mmode_cell{cell_index}(:));
    sampled_values = [sampled_values; ...
        current_values(1:sample_step:end)]; %#ok<AGROW>
end

sampled_values = sampled_values(isfinite(sampled_values));

end

function percentile_value = local_percentile(values,percentile)

values = double(values(:));
values = values(isfinite(values));

if isempty(values)
    percentile_value = NaN;
    return;
end

percentile = min(max(double(percentile),0),100);

if numel(values) == 1
    percentile_value = values(1);
    return;
end

values = sort(values);
position = 1 + (numel(values)-1)*percentile/100;

lower_index = floor(position);
upper_index = ceil(position);
fraction = position - lower_index;

percentile_value = ...
    values(lower_index)*(1-fraction) + ...
    values(upper_index)*fraction;

end

function tf = isTextScalar(x)

tf = ischar(x) || (isstring(x) && isscalar(x));

end
