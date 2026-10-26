%% ============================================================
%  ECG Feature Extraction and Dynamics Analysis
%  ============================================================
% clear; clc;



%% ========== 1. Parameter Settings ==========

% Path to the mouse data file
 csvFile   = "./Mouse6.csv";


Fs       = 2000;            % Sampling rate (Hz)
tStart   = 1;               % Analysis start time (s)
tEnd     = 5000;            % Analysis end time (s)

leadIdx  = 2;               % ECG lead index
polarity = 'pos';           % R-peak polarity
k        = 20;              % Moving-average window length (beats)

% --- Feature output at specified times ---

% Mouse1
% delta = 10;
% targetTimes = delta + [ 0 10 20 30 40 50 ]*60;

% Mouse2
% delta = 10;
% targetTimes = delta + [ 0 5 15 20 30 35 ]*60;

% Mouse3
% delta = 10;
% targetTimes = delta + [ 0 10 15 25 30 40 ]*60;

% Mouse4
% delta = 10;
% targetTimes = delta + [ 0 10 20 30 40 50]*60;

% Mouse5
% delta =10;
% targetTimes = delta + [ 0 5 10 20 25 30]*60;  % Times to output (s), editable

% Mouse6
delta = 10;
targetTimes = delta + [ 0 10 20 30 40 50]*60;



% --- R-peak detection parameters ---
minRR        = 0.05;        % Minimum RR interval (s), prevents duplicate detections

% --- Waveform search-window parameters (s); tune to your signal morphology ---
qWinPre   = 0.005;          % Q wave: search for a valley within qWinPre before R
sWinPost  = 0.02;           % S wave: search for a valley within sWinPost after R
stWinPost = 0.06;           % T/ST-T: take the second valley/peak within stWinPost after R
basePre   = 0.01;           % Baseline: mean of the window of length basePre before the Q point
%% ============================================================

%% 2. Load Data and Extract Analysis Segment
data = readmatrix(csvFile);
nSamples = size(data, 1);
totalDuration = nSamples / Fs;
fprintf('Total duration: %.3f s | Sampling rate: %d Hz\n', totalDuration, Fs);

if tEnd > totalDuration, tEnd = totalDuration; end
idx = (round(tStart*Fs)+1) : round(tEnd*Fs);
t   = (idx-1) / Fs;                       % Time axis (s)

xRaw = data(idx, leadIdx);
xRaw = xRaw(:);


%% 3. Signal Preprocessing
xDet   = lowpass(xRaw, 100, Fs);

xMorph = xDet;

% Automatically determine R-peak polarity (compare positive vs negative extrema)
if strcmpi(polarity, 'auto')
    if max(xDet) >= abs(min(xDet)), polarity = 'pos'; else, polarity = 'neg'; end
end
if strcmpi(polarity, 'neg'), xDet = -xDet; end
fprintf('R-peak polarity: %s\n', polarity);

%% 4. R-Peak Detection (thresholding + peak picking)


thresh = 400;

[~, rLoc] = findpeaks(xDet, ...
    'MinPeakHeight', thresh, ...
    'MinPeakDistance', round(minRR * Fs));
nR = length(rLoc);
fprintf('Detected R peaks: %d\n', nR);

%% 5. Locate Q / S / T / ST-T for Each R Peak
valid  = false(nR, 1);
qLoc   = nan(nR,1);  sLoc = nan(nR,1);
tLoc   = nan(nR,1);  sttLoc = nan(nR,1);
sttAmp = nan(nR,1);
tAmp   = nan(nR,1);

N = length(xMorph);
for i = 1:nR
    r = rLoc(i);

    % --- Q: nearest local minimum before R within the search window ---
    qRange = max(r - round(qWinPre*Fs), 1) : r;
    if length(qRange) < 3, continue; end
    seg = xMorph(qRange);
    promThresh = 10;
    [~, vIdx] = findpeaks(-seg, 'MinPeakProminence', promThresh);
    if ~isempty(vIdx)
        q = qRange(vIdx(end));                     % Last valley = closest to R
    else
        [~, ii] = min(seg);
        q = qRange(ii);
    end

    % --- S: nearest local minimum after R within the search window ---
    sRange = r : min(r + round(sWinPost*Fs), N);
    if length(sRange) < 3, continue; end
    seg = xMorph(sRange);
    promThresh = 10;
    [~, vIdx] = findpeaks(-seg, 'MinPeakProminence', promThresh);
    if ~isempty(vIdx)
        s = sRange(vIdx(1));                       % First valley = closest to R
    else
        [~, ii] = min(seg);
        s = sRange(ii);
    end
    if s <= q, continue; end

    % --- Amplitude baseline: mean of the window before the Q point ---
    baseRange = max(q - round(basePre*Fs), 1) : max(q - round(0.002*Fs), 1);
    base = mean(xMorph(baseRange));

    % --- T: second local minimum after R within the search window (the first is S) ---
    stRange = r : min(r + round(stWinPost*Fs), N);
    if length(stRange) < 3, continue; end
    seg = xMorph(stRange);
    promThresh = 0.01* (max(seg) - min(seg));

    [~, vIdx] = findpeaks(-seg, 'MinPeakProminence', promThresh);
    if length(vIdx) >= 2
        tLoc(i) = stRange(vIdx(2));                % Second valley = T
        tAmp(i) = xMorph(tLoc(i)) - base;          % T amplitude (relative to baseline; negative = downward valley)
    end

    % --- ST-T: second local maximum after R within the search window ---
    % Note: R is the first sample of stRange, and findpeaks never detects
    % endpoints, so R is automatically excluded
    [~, pIdx] = findpeaks(seg, 'MinPeakProminence', promThresh);
    if length(pIdx) >= 2
        sttLoc(i) = stRange(pIdx(2));              % Second peak = ST-T
        sttAmp(i) = xMorph(sttLoc(i)) - base;      % ST-T amplitude (relative to baseline)
    end

    qLoc(i) = q; sLoc(i) = s;
    valid(i) = true;
end

% Discard beats for which localization failed
rLoc = rLoc(valid); qLoc = qLoc(valid);
sLoc = sLoc(valid);
tLoc = tLoc(valid); sttLoc = sttLoc(valid);
sttAmp = sttAmp(valid);
tAmp = tAmp(valid);
nB = length(rLoc);
fprintf('Valid beats: %d (T detected in %d, ST-T detected in %d)\n', ...
    nB, sum(~isnan(tLoc)), sum(~isnan(sttLoc)));

%% 6. Feature Time Series
tR   = (rLoc - 1) / Fs;
qrsDur = (sLoc - qLoc) / Fs * 1000;        % QRS width (ms)
RR     = diff(tR);                          % RR intervals (s)
tRR    = tR(2:end);                         % Times corresponding to RR intervals

% k-point moving average (omitnan: beats with failed T/ST-T detection
% do not contaminate the mean)
mv = @(v) movmean(v, k, 'omitnan', 'Endpoints', 'discard');
qrsDur_ma = mv(qrsDur);
sttAmp_ma = mv(sttAmp);
tAmp_ma   = mv(tAmp);
RR_ma     = mv(RR);

% RR STD: std of every k RR intervals as one data point (non-overlapping blocks)
nBlock = floor(length(RR) / k);
rrStd  = nan(nBlock, 1);
rrStdT = nan(nBlock, 1);
rrStdStart = nan(nBlock, 1);   % Start time of each RR block
rrStdEnd   = nan(nBlock, 1);   % End time of each RR block
for b = 1:nBlock
    blk = (b-1)*k + 1 : b*k;
    rrStd(b)  = std(RR(blk));
    rrStdT(b)     = mean(tRR(blk));
    rrStdStart(b) = tRR(blk(1));
    rrStdEnd(b)   = tRR(blk(end));
end

fprintf('Mean heart rate: %.1f bpm | Mean RR: %.1f ms | QRS: %.1f ms\n', ...
    60/mean(RR), mean(RR)*1000, mean(qrsDur));

%% 7. Figure 1: Detection Quality Check (signal + fiducial points)
figure('Color','w','Position',[80 80 1100 420]);
plot(t, xMorph, 'Color',[0.6 0.6 0.6]); hold on; grid on;
plot(t(rLoc), xMorph(rLoc), 'rv', 'MarkerFaceColor','r');
plot(t(qLoc), xMorph(qLoc), 'bs', 'MarkerFaceColor','b');
plot(t(sLoc), xMorph(sLoc), 'ms', 'MarkerFaceColor','m');

tPlot   = tLoc(~isnan(tLoc));                  % Drop undetected points (NaN cannot be used as an index)
sttPlot = sttLoc(~isnan(sttLoc));
plot(t(tPlot),   xMorph(tPlot),   'c^', 'MarkerFaceColor','c');
plot(t(sttPlot), xMorph(sttPlot), 'd', 'MarkerFaceColor',[1 0.5 0], 'MarkerEdgeColor',[1 0.5 0]);

xlabel('Time (s)'); ylabel('Amplitude');
legend('ECG','R','Q','S','T','ST-T','Location','best');
title(sprintf('Detection Check - Lead %d (%.1f ~ %.1f s)', leadIdx, tStart, tEnd));

%% 8. Figure 2: Feature Dynamics over Time
figure('Color','w','Position',[80 80 1000 800]);

subplot(4,1,1);
plot(tR, qrsDur, '.', 'Color',[0.7 0.7 0.7]); hold on; grid on;
plot(tR(k:end), qrsDur_ma, 'b-', 'LineWidth', 1.5);
ylabel('QRS (ms)');
title(sprintf('QRS Duration  (k = %d moving average)', k));

% Optional ST-T amplitude panel (switch the layout to subplot(5,1,*) if enabled):
% subplot(5,1,3);
% plot(tR, sttAmp, '.', 'Color',[0.7 0.7 0.7]); hold on; grid on;
% plot(tR(k:end), sttAmp_ma, 'b-', 'LineWidth', 1.5);
% ylabel('ST-T Amp');
% title('ST-T Amplitude');

subplot(4,1,2);
plot(tR, tAmp, '.', 'Color',[0.7 0.7 0.7]); hold on; grid on;
plot(tR(k:end), tAmp_ma, 'b-', 'LineWidth', 1.5);
ylabel('T Amp');
title('T-wave Amplitude');

subplot(4,1,3);
plot(tRR, RR*1000, '.', 'Color',[0.7 0.7 0.7]); hold on; grid on;
plot(tRR(k:end), RR_ma*1000, 'b-', 'LineWidth', 1.5);
ylabel('RR (ms)');
title('RR Interval');

subplot(4,1,4);
plot(rrStdT, rrStd*1000, '-o', 'LineWidth', 1.5, 'MarkerSize', 4); grid on;
ylabel('RR STD (ms)'); xlabel('Time (s)');
title(sprintf('RR Interval STD (one data point per %d RRs)', k));

sgtitle(sprintf('ECG Feature Dynamics - Lead %d, %.1f ~ %.1f s', leadIdx, tStart, tEnd));

%% 9. Feature Extraction and Output at Specified Times
% QRS, T amplitude, RR interval: values are taken strictly from the
% k-point moving-average curves in Fig. 2. If a specified time has no
% matching point on the curve, the nearest valid curve values on its
% left and right are averaged.
% RR STD: Fig. 2 uses non-overlapping blocks of k RRs; adjacent RR-STD
% values are never averaged. The block containing the specified time is
% used first; if the time falls outside all blocks, the nearest RR-STD
% point in Fig. 2 is used instead.

tQRS_ma = tR(k:end);      % Time axis of the QRS moving-average curve in Fig. 2
tT_ma   = tR(k:end);      % Time axis of the T-amplitude moving-average curve in Fig. 2
tRR_ma  = tRR(k:end);     % Time axis of the RR moving-average curve in Fig. 2
timeTol = max(0.5/Fs, 1e-6);  % Tolerance for treating a curve point as a match

targetTimes = targetTimes(:);
nTarget = length(targetTimes);
qrsAtTarget  = nan(nTarget, 1);
tAmpAtTarget = nan(nTarget, 1);
rrAtTarget   = nan(nTarget, 1);
rrStdAtTarget = nan(nTarget, 1);
qrsMethod   = strings(nTarget, 1);
tAmpMethod  = strings(nTarget, 1);
rrMethod    = strings(nTarget, 1);
rrStdMethod = strings(nTarget, 1);

for i = 1:nTarget
    [qrsAtTarget(i), qrsMethod(i)] = samplePlottedCurve( ...
        tQRS_ma, qrsDur_ma, targetTimes(i), timeTol, 'QRS duration');

    [tAmpAtTarget(i), tAmpMethod(i)] = samplePlottedCurve( ...
        tT_ma, tAmp_ma, targetTimes(i), timeTol, 'T amplitude');

    % RR is plotted in ms in Fig. 2, so convert to ms here as well
    [rrAtTarget(i), rrMethod(i)] = samplePlottedCurve( ...
        tRR_ma, RR_ma*1000, targetTimes(i), timeTol, 'RR interval');

    % RR STD: no left/right averaging; directly use the value of the
    % RR block containing the specified time
    if isempty(rrStd)
        rrStdMethod(i) = "No RR STD available";
    else
        blockIdx = find(targetTimes(i) >= rrStdStart & ...
                        targetTimes(i) <= rrStdEnd, 1, 'first');
        if ~isempty(blockIdx)
            rrStdAtTarget(i) = rrStd(blockIdx) * 1000;
            rrStdMethod(i) = "Containing RR block";
        else
            [~, nearestIdx] = min(abs(rrStdT - targetTimes(i)));
            rrStdAtTarget(i) = rrStd(nearestIdx) * 1000;
            rrStdMethod(i) = "Nearest Fig. 2 point";

        end
    end
end

featureTable = table(targetTimes, qrsAtTarget, tAmpAtTarget, ...
    rrAtTarget,  ...
    'VariableNames', {'Time_s', 'QRS_ms', 'T_Amp', 'RR_ms'});

fprintf('\n========== ECG Features at Specified Times ==========\n');
disp(featureTable)








% Sample a plotted curve at a target time; if there is no matching point,
% average the nearest valid values on its left and right
function [valueAtTarget, pickMethod] = samplePlottedCurve(curveTime, curveValue, targetTime, timeTol, featureName)
    valid = isfinite(curveTime) & isfinite(curveValue);
    curveTime = curveTime(valid);
    curveValue = curveValue(valid);

    if isempty(curveTime)
        valueAtTarget = NaN;
        pickMethod = "No valid curve point";
        warning('%s: the Fig. 2 curve has no valid data points.', featureName);
        return;
    end

    [curveTime, sortIdx] = sort(curveTime);
    curveValue = curveValue(sortIdx);

    % Prefer a curve point coinciding with the target time (within tolerance)
    [nearestDiff, nearestIdx] = min(abs(curveTime - targetTime));
    if nearestDiff <= timeTol
        valueAtTarget = curveValue(nearestIdx);
        pickMethod = "Exact curve point";
        return;
    end

    % If there is no matching point, average the nearest valid curve
    % values on the left and right of the target time
    leftIdx  = find(curveTime < targetTime - timeTol, 1, 'last');
    rightIdx = find(curveTime > targetTime + timeTol, 1, 'first');

    if ~isempty(leftIdx) && ~isempty(rightIdx)
        valueAtTarget = (curveValue(leftIdx) + curveValue(rightIdx)) / 2;
        pickMethod = "Mean of adjacent points";
    else
        valueAtTarget = NaN;
        pickMethod = "Outside curve range";
        warning('%s: t = %.3f s is outside the Fig. 2 curve range; left/right averaging is not possible.', ...
            featureName, targetTime);
    end
end