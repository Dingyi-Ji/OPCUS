function [peak_amplitudes, heart_rate_bpm, peak_frequency_hz, summary_info, ...
    spatial_peak_amplitudes, spatial_peak_to_peak_amplitudes] = ...
    compare_heart_rate_peak_amplitudes( ...
    z_trajectory_set, time_values, hr_range_hz, varargin)
%COMPARE_HEART_RATE_PEAK_AMPLITUDES
% Estimate the heart rate from each of multiple M-wave z-trajectories,
% and plot the spectral amplitude at the selected heart-rate frequency
% across datasets as a line chart.
%
% Inputs:
%   z_trajectory_set
%       1 x n_datasets cell array; each cell stores one z-trajectory.
%
%   time_values
%       Time coordinates corresponding to the z-trajectories, in seconds.
%
%   hr_range_hz
%       Heart-rate search range, in Hz, e.g., [4,12].
%
% Optional parameters:
%   'Labels'
%       Name of each dataset.
%
%   'ShowIndividualPlots'
%       Whether to show the trajectory and spectrum plot for each
%       dataset individually; default false.
%
%   'ShowHarmonic'
%       Whether to mark the second harmonic in the individual spectrum
%       plots; default false.
%
%   'ShowSummaryPlot'
%       Whether to show the summary line chart of peak amplitudes;
%       default true.
%
%   'SummaryValue'
%       Value used in the summary line chart:
%       'spectral_amplitude': raw FFT amplitude;
%       'spatial_peak': spatial-domain peak amplitude;
%       'spatial_peak_to_peak': spatial-domain peak-to-peak vibration
%       amplitude (default).
%
%   'EstimateOptions'
%       Additional parameter cells passed to
%       estimate_heart_rate_from_z_track.
%
% Outputs:
%   peak_amplitudes
%       1 x n_datasets vector; amplitude at the selected heart-rate
%       frequency for each dataset.
%
%   heart_rate_bpm
%       Estimated heart rate of each dataset, in BPM.
%
%   peak_frequency_hz
%       Selected dominant heart-rate frequency of each dataset, in Hz.
%
%   summary_info
%       Structure containing the power spectra, amplitude spectra,
%       spatial amplitude spectra, heart rates, figure handles, etc.
%
%   spatial_peak_amplitudes
%       Spatial-domain peak amplitude of each dataset, in the same
%       units as the z-trajectories.
%
%   spatial_peak_to_peak_amplitudes
%       Spatial-domain peak-to-peak vibration amplitude of each
%       dataset, in the same units as the z-trajectories.
%
% Typical usage:
%   [peak_amplitudes, hr_bpm, peak_frequency, summary_info, ...
%    spatial_peak_amplitudes, spatial_p2p_amplitudes] = ...
%       compare_heart_rate_peak_amplitudes( ...
%       z_trajectory_set, ...
%       tracking_info.time_values, ...
%       [4,12], ...
%       'Labels', tracking_info.data_labels);

%% ==================== 1. Input and Parameter Validation ====================

if ~iscell(z_trajectory_set) || isempty(z_trajectory_set) || ...
        ~isvector(z_trajectory_set)
    error('z_trajectory_set must be a non-empty vector cell array.');
end

z_trajectory_set = z_trajectory_set(:).';
n_datasets = numel(z_trajectory_set);

parser = inputParser;
parser.FunctionName = mfilename;

addParameter(parser, 'Labels', {}, ...
    @(x) isempty(x) || isstring(x) || ...
    (iscell(x) && all(cellfun(@isTextScalar, x))));

addParameter(parser, 'ShowIndividualPlots', false, ...
    @(x) islogical(x) && isscalar(x));

addParameter(parser, 'ShowHarmonic', false, ...
    @(x) islogical(x) && isscalar(x));

addParameter(parser, 'ShowSummaryPlot', true, ...
    @(x) islogical(x) && isscalar(x));

addParameter(parser, 'SummaryValue', 'spatial_peak_to_peak', ...
    @isTextScalar);

addParameter(parser, 'EstimateOptions', {}, ...
    @(x) iscell(x) && mod(numel(x),2) == 0);

addParameter(parser, 'ShowValues', true, ...
    @(x) islogical(x) && isscalar(x));

addParameter(parser, 'Verbose', true, ...
    @(x) islogical(x) && isscalar(x));

parse(parser, varargin{:});
opt = parser.Results;

opt.SummaryValue = lower(strtrim(char(opt.SummaryValue)));

if ~ismember(opt.SummaryValue, ...
        {'spectral_amplitude','spatial_peak','spatial_peak_to_peak'})
    error(['SummaryValue must be ''spectral_amplitude'', ', ...
        '''spatial_peak'', or ''spatial_peak_to_peak''.']);
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

%% ==================== 2. Estimate Heart Rate and Peak Amplitude for Each Dataset ====================

peak_amplitudes = nan(1,n_datasets);
peak_powers = nan(1,n_datasets);
spatial_peak_amplitudes = nan(1,n_datasets);
spatial_peak_to_peak_amplitudes = nan(1,n_datasets);
heart_rate_bpm = nan(1,n_datasets);
peak_frequency_hz = nan(1,n_datasets);

frequency_set = cell(1,n_datasets);
power_spectrum_set = cell(1,n_datasets);
amplitude_spectrum_set = cell(1,n_datasets);
spatial_amplitude_spectrum_set = cell(1,n_datasets);
hr_info_set = cell(1,n_datasets);

for data_index = 1:n_datasets

    [heart_rate_bpm(data_index), ...
     peak_frequency_hz(data_index), ...
     frequency_set{data_index}, ...
     power_spectrum_set{data_index}, ...
     hr_info_set{data_index}, ...
     peak_amplitudes(data_index), ...
     amplitude_spectrum_set{data_index}, ...
     spatial_peak_amplitudes(data_index), ...
     spatial_peak_to_peak_amplitudes(data_index), ...
     spatial_amplitude_spectrum_set{data_index}] = ...
        estimate_heart_rate_from_z_track( ...
        time_values, ...
        z_trajectory_set{data_index}, ...
        hr_range_hz, ...
        'ShowPlot', opt.ShowIndividualPlots, ...
        'PlotTitle', data_labels{data_index}, ...
        'ShowHarmonic', opt.ShowHarmonic, ...
        'Verbose', opt.Verbose, ...
        opt.EstimateOptions{:});

    peak_powers(data_index) = hr_info_set{data_index}.peak_power;
end

switch opt.SummaryValue

    case 'spectral_amplitude'
        summary_values = peak_amplitudes;
        summary_ylabel = 'Raw FFT amplitude at the selected heart-rate frequency';
        summary_title = 'FFT amplitude at the selected heart-rate frequency across datasets';

    case 'spatial_peak'
        summary_values = spatial_peak_amplitudes;
        summary_ylabel = 'Spatial-domain peak vibration amplitude';
        summary_title = 'Spatial-domain peak vibration amplitude at the heart rate across datasets';

    case 'spatial_peak_to_peak'
        summary_values = spatial_peak_to_peak_amplitudes;
        summary_ylabel = 'Spatial-domain peak-to-peak vibration amplitude';
        summary_title = 'Spatial-domain peak-to-peak vibration amplitude at the heart rate across datasets';
end

%% ==================== 3. Plot the Peak-Amplitude Line Chart Across Datasets ====================

summary_figure = [];

if opt.ShowSummaryPlot

    summary_figure = figure( ...
        'Color', 'w', ...
        'Name', 'Heart-rate frequency amplitude comparison across datasets', ...
        'Position', [150,130,880,520]);

    current_axes = axes('Parent', summary_figure);
    hold(current_axes, 'on');

    plot(current_axes, 1:n_datasets, summary_values, ...
        '-o', ...
        'Color', [0.15,0.40,0.85], ...
        'LineWidth', 2.0, ...
        'MarkerSize', 8, ...
        'MarkerFaceColor', [0.15,0.40,0.85], ...
        'MarkerEdgeColor', 'k');

    finite_values = summary_values(isfinite(summary_values));

    if ~isempty(finite_values)
        y_min = min(finite_values);
        y_max = max(finite_values);

        if y_min == y_max
            y_padding = max(abs(y_min)*0.05, 1e-12);
        else
            y_padding = (y_max-y_min)*0.15;
        end

        ylim(current_axes, [y_min-y_padding, y_max+y_padding]);
    end

    if opt.ShowValues

        for data_index = 1:n_datasets

            if ~isfinite(summary_values(data_index))
                continue;
            end

            text(current_axes, data_index, ...
                summary_values(data_index), ...
                sprintf('  %.4g', summary_values(data_index)), ...
                'HorizontalAlignment', 'left', ...
                'VerticalAlignment', 'middle', ...
                'FontWeight', 'bold');
        end
    end

    xlim(current_axes, [0.5,n_datasets+0.5]);
    xticks(current_axes, 1:n_datasets);
    xticklabels(current_axes, data_labels);
    xlabel(current_axes, 'Dataset');
    ylabel(current_axes, summary_ylabel);
    title(current_axes, summary_title, ...
        'Interpreter', 'none');
    grid(current_axes, 'on');
    box(current_axes, 'on');
    drawnow;
end

%% ==================== 4. Output Summary Information ====================

summary_info = struct();
summary_info.n_datasets = n_datasets;
summary_info.data_labels = data_labels;
summary_info.hr_range_hz = hr_range_hz;
summary_info.peak_powers = peak_powers;
summary_info.spatial_peak_amplitudes = spatial_peak_amplitudes;
summary_info.spatial_peak_to_peak_amplitudes = ...
    spatial_peak_to_peak_amplitudes;
summary_info.summary_values = summary_values;
summary_info.summary_value_type = opt.SummaryValue;
summary_info.frequency_set = frequency_set;
summary_info.power_spectrum_set = power_spectrum_set;
summary_info.amplitude_spectrum_set = amplitude_spectrum_set;
summary_info.spatial_amplitude_spectrum_set = ...
    spatial_amplitude_spectrum_set;
summary_info.hr_info_set = hr_info_set;
summary_info.summary_figure = summary_figure;
summary_info.options = opt;

if opt.Verbose
    fprintf('Heart-rate frequency amplitude of each dataset:\n');

    for data_index = 1:n_datasets
        fprintf(['%s: HR %.2f BPM, dominant frequency %.4f Hz, ', ...
            'spatial peak amplitude %.6g, peak-to-peak amplitude %.6g\n'], ...
            data_labels{data_index}, ...
            heart_rate_bpm(data_index), ...
            peak_frequency_hz(data_index), ...
            spatial_peak_amplitudes(data_index), ...
            spatial_peak_to_peak_amplitudes(data_index));
    end
end

end

function tf = isTextScalar(x)

tf = ischar(x) || (isstring(x) && isscalar(x));

end