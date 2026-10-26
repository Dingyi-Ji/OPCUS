function [heart_rate_bpm, peak_frequency_hz, frequency_hz, spectrum, hr_info, ...
    peak_amplitude, amplitude_spectrum, spatial_peak_amplitude, ...
    spatial_peak_to_peak_amplitude, spatial_amplitude_spectrum] = ...
    estimate_heart_rate_from_z_track(time_values, z_trajectory, hr_range_hz, varargin)
%ESTIMATE_HEART_RATE_FROM_Z_TRACK
% Estimate the heart rate from a single-point M-wave z-trajectory, and
% visualize the spectrum and the selected heart-rate peak.
%
% Inputs:
%   time_values
%       Time coordinates corresponding to the M-wave trajectory, in seconds.
%
%   z_trajectory
%       Z-direction trajectory extracted by track_single_point_mmode_grid_zroi.
%
%   hr_range_hz
%       Expected heart-rate frequency range, in Hz.
%       For example, [4,12] corresponds to 240-720 BPM.
%
% Optional parameters:
%   'ShowPlot'
%       Whether to show the trajectory and spectrum plots; default true.
%
%   'PlotTitle'
%       Title of the current spectrum plot; default ''.
%
%   'FrequencyDisplayMax'
%       Maximum frequency shown in the spectrum plot; by default it is
%       automatically set to 1.5x the upper heart-rate limit, without
%       exceeding the Nyquist frequency.
%
%   'ShowHarmonic'
%       Whether to mark the second-harmonic position; default false.
%
% Outputs:
%   heart_rate_bpm
%       Estimated heart rate, in BPM.
%
%   peak_frequency_hz
%       Dominant frequency corresponding to the heart rate, in Hz.
%
%   frequency_hz
%       Frequency axis.
%
%   spectrum
%       Single-sided power spectrum, equal to abs(FFT).^2.
%
%   hr_info
%       Structure containing the uniform time axis, the processed
%       trajectory, the sampling frequency, figure handles, etc.
%
%   peak_amplitude
%       Raw FFT amplitude at the selected heart-rate frequency, equal to
%       sqrt(power at the selected frequency).
%
%   amplitude_spectrum
%       Raw single-sided FFT amplitude spectrum, equal to sqrt(spectrum).
%
%   spatial_peak_amplitude
%       Peak amplitude A converted back to the spatial domain, in the
%       same units as the z-trajectory.
%       For non-DC frequencies, it is approximated as
%       2*abs(FFT)/sum(Hann window).
%
%   spatial_peak_to_peak_amplitude
%       Spatial-domain peak-to-peak vibration range, equal to
%       2*spatial_peak_amplitude.
%
%   spatial_amplitude_spectrum
%       Single-sided amplitude spectrum converted to the spatial domain,
%       in the same units as the z-trajectory.
%
% Typical usage:
%   [hr_bpm, f_peak, f, spectrum, hr_info, peak_amplitude, ...
%    amplitude_spectrum, spatial_peak_amplitude, ...
%    spatial_peak_to_peak_amplitude] = ...
%       estimate_heart_rate_from_z_track( ...
%       tracking_info.time_values, ...
%       z_trajectory_set{1}, ...
%       [4,12], ...
%       'PlotTitle', tracking_info.data_labels{1});

%% ==================== 1. Parameter Parsing ====================

parser = inputParser;
parser.FunctionName = mfilename;

addParameter(parser, 'ShowPlot', true, ...
    @(x) islogical(x) && isscalar(x));

addParameter(parser, 'PlotTitle', '', ...
    @(x) ischar(x) || (isstring(x) && isscalar(x)));

addParameter(parser, 'FrequencyDisplayMax', [], ...
    @(x) isempty(x) || ...
    (isnumeric(x) && isscalar(x) && isfinite(x) && x > 0));

addParameter(parser, 'ShowHarmonic', false, ...
    @(x) islogical(x) && isscalar(x));

addParameter(parser, 'Verbose', true, ...
    @(x) islogical(x) && isscalar(x));

parse(parser, varargin{:});
opt = parser.Results;

if isempty(hr_range_hz) || ~isnumeric(hr_range_hz) || ...
        numel(hr_range_hz) ~= 2 || any(~isfinite(hr_range_hz)) || ...
        hr_range_hz(1) <= 0 || hr_range_hz(2) <= hr_range_hz(1)
    error('hr_range_hz must be an increasing positive range of the form [f_min,f_max].');
end

hr_range_hz = sort(double(hr_range_hz(:).'));

%% ==================== 2. Prepare the Trajectory and Time Axis ====================

time_values = double(time_values(:).');
z_trajectory = double(z_trajectory(:).');

if numel(time_values) ~= numel(z_trajectory)
    error('time_values and z_trajectory must have the same length.');
end

valid_mask = isfinite(time_values) & isfinite(z_trajectory);
time_values = time_values(valid_mask);
z_trajectory = z_trajectory(valid_mask);

if numel(time_values) < 10
    error('Too few valid trajectory points to estimate the heart rate.');
end

[time_values, unique_order] = unique(time_values, 'sorted');
z_trajectory = z_trajectory(unique_order);

if numel(time_values) < 10
    error('Too few valid trajectory points remain after removing duplicate time points.');
end

time_differences = diff(time_values);

if any(time_differences <= 0)
    error('time_values must be strictly increasing.');
end

% Interpolate the trajectory onto a strictly uniform time axis so that
% the FFT frequencies are correct.
dt = median(time_differences);
time_uniform = time_values(1):dt:time_values(end);

z_uniform = interp1( ...
    time_values, z_trajectory, time_uniform, 'pchip');

finite_uniform = isfinite(z_uniform);

if nnz(finite_uniform) < 10
    error('Too few valid trajectory points after uniform-time interpolation.');
end

if any(~finite_uniform)
    z_uniform(~finite_uniform) = interp1( ...
        time_uniform(finite_uniform), ...
        z_uniform(finite_uniform), ...
        time_uniform(~finite_uniform), ...
        'nearest', 'extrap');
end

sampling_frequency = 1 / dt;
nyquist_frequency = sampling_frequency / 2;

if hr_range_hz(2) > nyquist_frequency

    warning(['The heart-rate search upper limit %.3g Hz exceeds the ', ...
        'Nyquist frequency %.3g Hz; the search limit has been truncated.'], ...
        hr_range_hz(2), nyquist_frequency);

    hr_range_hz(2) = nyquist_frequency;
end

if hr_range_hz(1) >= hr_range_hz(2)
    error('The current sampling frequency is insufficient to cover the specified heart-rate band.');
end

%% ==================== 3. Detrend, Window, and Compute the Single-Sided Spectrum ====================

z_detrended = detrend(z_uniform(:));
n_samples = numel(z_detrended);

% Generate the Hann window manually to avoid depending on the
% Signal Processing Toolbox.
if n_samples > 1
    sample_index = (0:n_samples-1).';
    hann_window = 0.5 - 0.5*cos( ...
        2*pi*sample_index/(n_samples-1));
else
    hann_window = 1;
end

z_windowed = z_detrended .* hann_window;

nfft = 2^nextpow2(n_samples * 4);
fft_result = fft(z_windowed, nfft);

spectrum = abs(fft_result(1:floor(nfft/2)+1)).^2;
frequency_hz = (0:numel(spectrum)-1).' * ...
    sampling_frequency / nfft;

%% ==================== 4. Find the Largest Spectral Peak Within the Heart-Rate Band ====================

search_mask = frequency_hz >= hr_range_hz(1) & ...
    frequency_hz <= hr_range_hz(2);

if ~any(search_mask)
    error('There are no spectral samples within the specified heart-rate band.');
end

candidate_frequencies = frequency_hz(search_mask);
candidate_spectrum = spectrum(search_mask);

[~, local_peak_index] = max(candidate_spectrum);
peak_frequency_hz = candidate_frequencies(local_peak_index);
heart_rate_bpm = peak_frequency_hz * 60;

peak_frequency_index = find(search_mask, 1, 'first') + ...
    local_peak_index - 1;

% Keep spectrum as the power spectrum, and additionally output the raw
% FFT amplitude spectrum and the peak amplitude.
amplitude_spectrum = sqrt(max(spectrum, 0));
peak_power = spectrum(peak_frequency_index);
peak_amplitude = amplitude_spectrum(peak_frequency_index);

% Divide the FFT amplitude by the coherent gain of the window to convert
% it back to the spatial-domain amplitude.
% If z(t)=A*sin(2*pi*f*t+phi), the main peak of the single-sided spectrum
% satisfies |X(f)| ≈ A*sum(window)/2, hence A≈2*|X(f)|/sum(window).
window_sum = sum(hann_window);
spatial_amplitude_spectrum = amplitude_spectrum / window_sum;

if mod(nfft,2) == 0
    spatial_amplitude_spectrum(2:end-1) = ...
        2 * spatial_amplitude_spectrum(2:end-1);
else
    spatial_amplitude_spectrum(2:end) = ...
        2 * spatial_amplitude_spectrum(2:end);
end

spatial_peak_amplitude = ...
    spatial_amplitude_spectrum(peak_frequency_index);
spatial_peak_to_peak_amplitude = 2 * spatial_peak_amplitude;
spatial_rms_amplitude = spatial_peak_amplitude / sqrt(2);

%% ==================== 5. Visualize the Trajectory, Spectrum, and Selected Heart-Rate Peak ====================

spectrum_figure = [];

if opt.ShowPlot

    if isempty(opt.FrequencyDisplayMax)
        frequency_display_max = min( ...
            nyquist_frequency, ...
            max(hr_range_hz(2)*1.5, hr_range_hz(2)+2));
    else
        frequency_display_max = min( ...
            opt.FrequencyDisplayMax, nyquist_frequency);
    end

    plot_title = char(opt.PlotTitle);

    if isempty(plot_title)
        figure_title = 'Heart-rate spectral analysis of the M-wave z-trajectory';
    else
        figure_title = plot_title;
    end

    spectrum_figure = figure( ...
        'Color', 'w', ...
        'Name', figure_title, ...
        'Position', [120,100,1000,720]);

    % Top panel: original trajectory and uniformly interpolated trajectory.
    ax_time = subplot(2,1,1);
    plot(ax_time, time_values, z_trajectory, ...
        'o-', ...
        'Color', [0.35,0.35,0.35], ...
        'MarkerSize', 3, ...
        'LineWidth', 0.8);
    hold(ax_time, 'on');
    plot(ax_time, time_uniform, z_uniform, ...
        '-', ...
        'Color', [0.10,0.45,0.85], ...
        'LineWidth', 1.2);
    grid(ax_time, 'on');
    box(ax_time, 'on');
    xlabel(ax_time, 'Time (s)');
    ylabel(ax_time, 'z position');
    title(ax_time, 'M-wave z-trajectory used for heart-rate estimation');
    legend(ax_time, {'Original trajectory','Uniformly interpolated trajectory'}, ...
        'Location', 'best');

    % Bottom panel: spectrum, heart-rate search range, and selected peak.
    ax_spectrum = subplot(2,1,2);
    plot(ax_spectrum, frequency_hz, spectrum, ...
        '-', ...
        'Color', [0.10,0.35,0.75], ...
        'LineWidth', 1.2);
    hold(ax_spectrum, 'on');
    grid(ax_spectrum, 'on');
    box(ax_spectrum, 'on');

    xlim(ax_spectrum, [0, frequency_display_max]);

    finite_spectrum = spectrum(isfinite(spectrum));

    if isempty(finite_spectrum)
        y_limits = [0,1];
    else
        y_max = max(finite_spectrum);

        if y_max <= 0
            y_max = 1;
        end

        y_limits = [0, y_max*1.15];
    end

    ylim(ax_spectrum, y_limits);

    % The heart-rate search band is highlighted with a light background.
    range_patch = patch(ax_spectrum, ...
        [hr_range_hz(1), hr_range_hz(2), ...
        hr_range_hz(2), hr_range_hz(1)], ...
        [y_limits(1), y_limits(1), ...
        y_limits(2), y_limits(2)], ...
        [0.20,0.75,0.35], ...
        'FaceAlpha', 0.12, ...
        'EdgeColor', 'none');
    uistack(range_patch, 'bottom');

    % Selected main peak.
    plot(ax_spectrum, peak_frequency_hz, ...
        spectrum(peak_frequency_index), ...
        'o', ...
        'MarkerSize', 9, ...
        'MarkerFaceColor', [0.90,0.10,0.10], ...
        'MarkerEdgeColor', 'k', ...
        'LineWidth', 1.0);

    xline(ax_spectrum, peak_frequency_hz, ...
        '--', ...
        'Color', [0.90,0.10,0.10], ...
        'LineWidth', 1.2);

    text(ax_spectrum, peak_frequency_hz, ...
        spectrum(peak_frequency_index), ...
        sprintf(['  %.3f Hz = %.1f BPM\n', ...
        '  Power = %.4g, FFT amplitude = %.4g\n', ...
        '  Spatial peak amp = %.4g, peak-to-peak = %.4g'], ...
        peak_frequency_hz, heart_rate_bpm, ...
        peak_power, peak_amplitude, ...
        spatial_peak_amplitude, spatial_peak_to_peak_amplitude), ...
        'Color', [0.70,0.05,0.05], ...
        'FontWeight', 'bold', ...
        'VerticalAlignment', 'bottom');

    if opt.ShowHarmonic

        harmonic_frequency = 2 * peak_frequency_hz;

        if harmonic_frequency <= frequency_display_max
            xline(ax_spectrum, harmonic_frequency, ...
                ':', ...
                'Color', [0.80,0.45,0.05], ...
                'LineWidth', 1.2);
            text(ax_spectrum, harmonic_frequency, ...
                y_limits(2)*0.90, '  2nd harmonic', ...
                'Color', [0.80,0.45,0.05]);
        end
    end

    xlabel(ax_spectrum, 'Frequency (Hz)');
    ylabel(ax_spectrum, 'Power spectrum magnitude');
    title(ax_spectrum, sprintf( ...
        'Heart-rate spectrum: selected %.3f Hz (%.1f BPM)', ...
        peak_frequency_hz, heart_rate_bpm));

    legend(ax_spectrum, ...
        {'Power spectrum','Heart-rate search range','Selected heart-rate peak'}, ...
        'Location', 'best');

    if ~isempty(plot_title)
        sgtitle(spectrum_figure, plot_title, ...
            'Interpreter', 'none');
    end

    drawnow;
end

%% ==================== 6. Output Detailed Information ====================

hr_info = struct();
hr_info.time_original = time_values;
hr_info.z_original = z_trajectory;
hr_info.time_uniform = time_uniform;
hr_info.z_uniform = z_uniform;
hr_info.z_detrended = z_detrended;
hr_info.sampling_frequency = sampling_frequency;
hr_info.nyquist_frequency = nyquist_frequency;
hr_info.hr_range_hz = hr_range_hz;
hr_info.hr_range_bpm = hr_range_hz * 60;
hr_info.peak_frequency_index = peak_frequency_index;
hr_info.peak_power = peak_power;
hr_info.peak_amplitude = peak_amplitude;
hr_info.amplitude_spectrum = amplitude_spectrum;
hr_info.window_sum = window_sum;
hr_info.spatial_peak_amplitude = spatial_peak_amplitude;
hr_info.spatial_peak_to_peak_amplitude = ...
    spatial_peak_to_peak_amplitude;
hr_info.spatial_rms_amplitude = spatial_rms_amplitude;
hr_info.spatial_amplitude_spectrum = spatial_amplitude_spectrum;
hr_info.spectrum_figure = spectrum_figure;
hr_info.options = opt;

if opt.Verbose
    fprintf(['Estimated heart rate: %.2f BPM (dominant frequency %.4f Hz, ', ...
        'spatial peak amplitude %.6g, peak-to-peak amplitude %.6g)\n'], ...
        heart_rate_bpm, peak_frequency_hz, ...
        spatial_peak_amplitude, spatial_peak_to_peak_amplitude);
end

end