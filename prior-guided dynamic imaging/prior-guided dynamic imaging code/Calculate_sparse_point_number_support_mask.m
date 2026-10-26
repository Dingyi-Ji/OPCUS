%% Calculate_sparse_point_number_support_mask.m
% Support-mask-derived TV sparsity -> transmit-event budget

clear; close all; clc;

%% ====================== USER CONFIGURATION ======================

% Binary / denoised support mask obtained from the preliminary full-field scan.
CFG.mask_path = 'mask_result.mat';
CFG.mask_var  = '';  % empty -> auto-detect largest 3-D numeric/logical variable

% If the loaded volume is not already logical / binary, threshold its
% normalized magnitude to produce the support mask.
CFG.mask_threshold_fraction = 0.10;

% Full candidate transmit grid.
CFG.tx_grid_size = [101, 101];
N_full = prod(CFG.tx_grid_size);

% Empirical mapping from support-mask TV sparsity to transmit-event fraction:
%
%   rho_tv = K_tv * log(e*N_grad/K_tv) / N_grad
%   rho_budget = budget_gain * rho_tv
%
% K_tv is the NUMBER of nonzero 3-D finite-difference coefficients of the
% binary support mask, rather than the sum of gradient magnitudes.
CFG.budget_gain = 1.3;

% Absolute minimum number of transmit events.
% IMPORTANT: there is NO 2% lower-bound term in this version.
CFG.min_sources_absolute = 100;

% Optional practical upper bound for sparse acquisition.
% Set to 1.00 to disable the upper cap.
CFG.max_sampling_fraction = 0.20;

% Output file.
CFG.output_mat = 'sparse_point_number_support_mask_result.mat';

%% ====================== 1. LOAD SUPPORT MASK ======================

fprintf('Loading support mask: %s\n', CFG.mask_path);

S = load(CFG.mask_path);
mask_raw = load_largest_3d_numeric(S, CFG.mask_var, 'support mask');
mask_raw = squeeze(mask_raw);

if ndims(mask_raw) ~= 3
    error('The support mask must be a 3-D array.');
end

if islogical(mask_raw)
    support_mask = mask_raw;
else
    mask_raw = double(mask_raw);
    mask_raw(~isfinite(mask_raw)) = 0;

    % If already binary-like in [0,1], keep all positive voxels.
    unique_vals = unique(mask_raw(:));
    is_binary_like = numel(unique_vals) <= 2 && ...
                     all(unique_vals >= 0) && all(unique_vals <= 1);

    if is_binary_like
        support_mask = mask_raw > 0;
    else
        ma = abs(mask_raw);
        ma_max = max(ma(:));

        if ma_max <= 0
            error('Loaded mask volume is all zero.');
        end

        support_mask = ma > CFG.mask_threshold_fraction * ma_max;
    end
end

if ~any(support_mask(:))
    error('Support mask is empty.');
end

[Nx, Ny, Nz] = size(support_mask);
P = numel(support_mask);
N_omega = nnz(support_mask);
support_fraction = N_omega / P;

fprintf('\n========== SUPPORT STATISTICS ==========\n');
fprintf('Reconstruction grid:                 %d x %d x %d = %d voxels\n', ...
    Nx, Ny, Nz, P);
fprintf('Support voxels N_omega:              %d\n', N_omega);
fprintf('Support fraction:                    %.4f (%.2f%%)\n', ...
    support_fraction, 100*support_fraction);

%% ====================== 2. SUPPORT-MASK TV SPARSITY ======================
%
% The binary support mask is treated as the structural prior.
%
% Instead of:
%   K = sum(|grad(mask)|)
%
% we count the number of NONZERO finite-difference coefficients:
%
%   K_tv = ||D_x mask||_0 + ||D_y mask||_0 + ||D_z mask||_0
%
% This is closer to a true sparsity count in the discrete TV-gradient domain.

mask_double = double(support_mask);

Dx = diff(mask_double, 1, 1);
Dy = diff(mask_double, 1, 2);
Dz = diff(mask_double, 1, 3);

Kx = nnz(Dx);
Ky = nnz(Dy);
Kz = nnz(Dz);

K_tv = Kx + Ky + Kz;

% Number of available finite-difference coefficients in the 3-D gradient domain.
N_grad = numel(Dx) + numel(Dy) + numel(Dz);

if K_tv <= 0
    warning('The support mask has no detected boundary changes; using the minimum budget.');
    tv_sparsity_index = 0;
else
    % Dimensionless CS-inspired complexity index.
    % It is used only as an empirical monotonic complexity measure.
    tv_sparsity_index = ...
        K_tv * log(exp(1) * N_grad / K_tv) / N_grad;
end

tv_sparsity_index = min(max(tv_sparsity_index, 0), 1);

% Boundary mask is used only for visualization / reporting.
if exist('bwperim', 'file') == 2
    boundary_mask = bwperim(support_mask, 6);
else
    % Fallback without Image Processing Toolbox.
    [gy, gx, gz] = gradient(mask_double);
    boundary_mask = sqrt(gx.^2 + gy.^2 + gz.^2) > 0;
end

fprintf('\n========== SUPPORT-MASK TV SPARSITY ==========\n');
fprintf('Nonzero x-difference coefficients Kx: %d\n', Kx);
fprintf('Nonzero y-difference coefficients Ky: %d\n', Ky);
fprintf('Nonzero z-difference coefficients Kz: %d\n', Kz);
fprintf('Total TV sparsity count K_tv:         %d\n', K_tv);
fprintf('Total gradient coefficients N_grad:   %d\n', N_grad);
fprintf('Normalized TV sparsity index:         %.6f\n', tv_sparsity_index);

%% ====================== 3. MAP SPARSITY TO SOURCE NUMBER ======================

rho_budget_raw = CFG.budget_gain * tv_sparsity_index;

% Only an upper fraction cap is applied here.
rho_budget_capped = min(rho_budget_raw, CFG.max_sampling_fraction);

N_select_raw = round(N_full * rho_budget_capped);

% Absolute minimum only: 100 points.
N_select = max(N_select_raw, CFG.min_sources_absolute);
N_select = min(N_select, N_full);

sampling_fraction = N_select / N_full;

fprintf('\n========== ADAPTIVE TRANSMIT-EVENT BUDGET ==========\n');
fprintf('Full transmit grid:                  %d x %d = %d\n', ...
    CFG.tx_grid_size(1), CFG.tx_grid_size(2), N_full);
fprintf('Budget gain:                         %.4f\n', CFG.budget_gain);
fprintf('Raw sparsity-derived fraction:       %.6f\n', rho_budget_raw);
fprintf('Upper-capped fraction:               %.6f\n', rho_budget_capped);
fprintf('Raw source number before minimum:    %d\n', N_select_raw);
fprintf('Absolute minimum source number:      %d\n', CFG.min_sources_absolute);
fprintf('Final sampling fraction:             %.6f (%.2f%%)\n', ...
    sampling_fraction, 100*sampling_fraction);
fprintf('Recommended transmit-event number:   %d\n', N_select);

%% ====================== 4. VISUALIZATION ======================

figure('Name', 'Support-mask-derived TV sparsity', ...
       'Position', [100 100 1250 380]);

subplot(1,3,1);
imagesc(squeeze(max(support_mask, [], 3)));
axis image;
set(gca, 'YDir', 'normal');
colormap(gca, gray);
colorbar;
title(sprintf('Binary support (%.2f%%)', 100*support_fraction));
xlabel('x'); ylabel('y');

subplot(1,3,2);
imagesc(squeeze(max(boundary_mask, [], 3)));
axis image;
set(gca, 'YDir', 'normal');
colormap(gca, gray);
colorbar;
title(sprintf('Support boundary, K_{TV} = %d', K_tv));
xlabel('x'); ylabel('y');

subplot(1,3,3);
bar([Kx, Ky, Kz]);
set(gca, 'XTickLabel', {'K_x','K_y','K_z'});
ylabel('Nonzero finite differences');
title(sprintf('\\rho_{TV} = %.4f, N_{select} = %d', ...
    tv_sparsity_index, N_select));
grid on;

%% ====================== 5. SAVE ======================

config_used = CFG;

save(CFG.output_mat, ...
    'N_select', ...
    'sampling_fraction', ...
    'tv_sparsity_index', ...
    'K_tv', ...
    'Kx', ...
    'Ky', ...
    'Kz', ...
    'N_grad', ...
    'N_omega', ...
    'support_fraction', ...
    'support_mask', ...
    'boundary_mask', ...
    'config_used');

fprintf('\nSaved result to: %s\n', CFG.output_mat);

%% ====================== LOCAL FUNCTION ======================

function x = load_largest_3d_numeric(S, requested_name, label)

    if ~isempty(requested_name)
        if ~isfield(S, requested_name)
            error('Requested %s variable "%s" was not found.', ...
                label, requested_name);
        end
        x = S.(requested_name);
        return;
    end

    names = fieldnames(S);
    best_name = '';
    best_numel = -inf;

    for ii = 1:numel(names)
        v = S.(names{ii});

        if (isnumeric(v) || islogical(v)) && ndims(v) >= 3
            if numel(v) > best_numel
                best_numel = numel(v);
                best_name = names{ii};
            end
        end
    end

    if isempty(best_name)
        error('No suitable 3-D numeric/logical %s variable was found.', label);
    end

    fprintf('Auto-selected %s variable: %s\n', label, best_name);
    x = S.(best_name);
end
