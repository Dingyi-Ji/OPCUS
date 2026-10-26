% M-wave analysis code for OPCUS imaging results

% Using Mouse1 data as an example
data1 = load("..\..\..\..\data\hypoxia experiments\mouse1\OPCUS dynamic imaging result\0min\mouse1_0min_frames1_20.mat").reconstruction_all_frames;
tmp = load("..\..\..\..\data\hypoxia experiments\mouse1\OPCUS dynamic imaging result\0min\mouse1_0min_frames21_40.mat").reconstruction_all_frames;
data1 = cat(1, data1, tmp);

data2 = load("..\..\..\..\data\hypoxia experiments\mouse1\OPCUS dynamic imaging result\10min\mouse1_10min_frames1_20.mat").reconstruction_all_frames;
tmp = load("..\..\..\..\data\hypoxia experiments\mouse1\OPCUS dynamic imaging result\10min\mouse1_10min_frames21_40.mat").reconstruction_all_frames;
data2 = cat(1, data2, tmp);

data3 = load("..\..\..\..\data\hypoxia experiments\mouse1\OPCUS dynamic imaging result\20min\mouse1_20min_frames1_20.mat").reconstruction_all_frames;
tmp = load("..\..\..\..\data\hypoxia experiments\mouse1\OPCUS dynamic imaging result\20min\mouse1_20min_frames21_40.mat").reconstruction_all_frames;
data3 = cat(1, data3, tmp);

data4 = load("..\..\..\..\data\hypoxia experiments\mouse1\OPCUS dynamic imaging result\30min\mouse1_30min_frames1_20.mat").reconstruction_all_frames;
tmp = load("..\..\..\..\data\hypoxia experiments\mouse1\OPCUS dynamic imaging result\30min\mouse1_30min_frames21_40.mat").reconstruction_all_frames;
data4 = cat(1, data4, tmp);

data5 = load("..\..\..\..\data\hypoxia experiments\mouse1\OPCUS dynamic imaging result\40min\mouse1_40min_frames1_20.mat").reconstruction_all_frames;
tmp = load("..\..\..\..\data\hypoxia experiments\mouse1\OPCUS dynamic imaging result\40min\mouse1_40min_frames21_40.mat").reconstruction_all_frames;
data5 = cat(1, data5, tmp);

data6 = load("..\..\..\..\data\hypoxia experiments\mouse1\OPCUS dynamic imaging result\50min\mouse1_50min_frames1_20.mat").reconstruction_all_frames;
tmp = load("..\..\..\..\data\hypoxia experiments\mouse1\OPCUS dynamic imaging result\50min\mouse1_50min_frames21_40.mat").reconstruction_all_frames;
data6 = cat(1, data6, tmp);






% The loaded data is in [t,x,y,z] order; the following code converts it to [x,y,z,t]
data1 = permute(data1,[2,3,4,1]);
data2 = permute(data2,[2,3,4,1]);
data3 = permute(data3,[2,3,4,1]);
data4 = permute(data4,[2,3,4,1]);
data5 = permute(data5,[2,3,4,1]);
data6 = permute(data6,[2,3,4,1]);


% Flip the z-axis upside down
data1 = data1(:, :, end:-1:1, :);
data2 = data2(:, :, end:-1:1, :);
data3 = data3(:, :, end:-1:1, :);
data4 = data4(:, :, end:-1:1, :);
data5 = data5(:, :, end:-1:1, :);
data6 = data6(:, :, end:-1:1, :);


data1_tackled = data1.^0.5;
data2_tackled = data2.^0.5;
data3_tackled = data3.^0.5;
data4_tackled = data4.^0.5;
data5_tackled = data5.^0.5;
data6_tackled = data6.^0.5;


% Compute the M-wave of each dataset
data_sets = {data1_tackled, data2_tackled, data3_tackled, data4_tackled, data5_tackled,data6_tackled};
[mmode_all, info] = ...
    single_point_multiDataMWave( ...
    data_sets, 1, ...
    'FrameRate', 26.67, ...
    'TimeScale', 20, ...            % Temporal interpolation factor: 20x
    'DepthScale', 5, ...            % Z-direction interpolation factor; no interpolation by default
    'InterpMethod', 'pchip', ...    % Interpolation
    'SmoothSigma', [0,0], ...,
    'NormalizeEachFrame', false,...
    'Labels', {'0min','10min','20min','30min','40min','50min'} ,'SelectedPoint', [83,130]) ;


% Perform trajectory tracking on the data
[z_trajectory_set, peak_to_valley_distance, tracking_info] = ...
    track_single_point_mmode_grid_zroi( ...
    mmode_all, ...                          % Cell array of single-point multi-dataset M-waves; each cell is a [z,time] matrix
    'Labels', info.data_labels, ...   % Name of each dataset, used for figure titles and x-axes
    'Time', info.time_output_s, ...   % Interpolated time coordinates, corresponding to M-wave columns
    'ZCoordinates', info.z_output, ...% Interpolated z-coordinates, corresponding to M-wave rows
    'ZRange', [50*5,100*5], ...                 % Z-direction ROI used for trajectory tracking
    'ZRangeUnits', 'index', ...             % ZRange is interpreted as M-wave row indices
    'TopPercent', 5, ...                    % At each time point, select the top 5% brightest pixels within the ROI
    'BackgroundPercentile', 20, ...         % Estimate and subtract the background using the 20th percentile
    'WeightPower', 1, ...                   % Brightness weighting exponent; 1 = linear weighting
    'MaxStep', 10, ...
    'ShowSummaryPlot', false, ...
    'PeakToValleyMetric', 'robust');                     % Maximum allowed trajectory jump between adjacent time points



% Heart rate calculation
hr_range_hz = [4,12];  % Expected heart rate range
[nona, heart_rate_bpm, nonb, nonc] = ...
    compare_heart_rate_peak_amplitudes( ...
    z_trajectory_set, ...                    % Z-direction motion trajectory of each dataset
    tracking_info.time_values, ...           % Time axis corresponding to the M-wave trajectory, in seconds
    hr_range_hz, ...                         % Heart-rate search band, in Hz
    'Labels', info.data_labels, ...          % Use dataset names as figure titles
    'ShowIndividualPlots', true, ...         % Show the spectrum plot of each individual dataset
    'ShowHarmonic', true, ...                % Mark the second-harmonic position in the spectrum
    'ShowSummaryPlot', true);                % Plot the peak-amplitude line chart across the six datasets


disp('Heart rate of each dataset (BPM):');
disp(heart_rate_bpm);

disp('Corresponding RR intervals (ms):');
rr_interval_ms = 60000 ./ heart_rate_bpm;
disp(rr_interval_ms);
