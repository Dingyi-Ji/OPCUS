% Denoise and filter the full-field compressed-sensing reconstruction result
% mask is the final data; save it as the variable mask_result.mat


% Load the data
ALL_field_recon_result = squeeze( load("..\..\..\data\prior-guided dynamic imaging\reconstruction_allfield.mat").reconstruction_all_frames );

ALL_field_recon_result(ALL_field_recon_result<0.69)=0.69;

% ALL_field_recon_result = squeeze(reconstruction_all_frames);
volumeViewer(squeeze(ALL_field_recon_result))

% Normalization
ALL_field_recon_result = (ALL_field_recon_result-min(ALL_field_recon_result(:)))/(max(ALL_field_recon_result(:))-min(ALL_field_recon_result(:)));

ALL_field_recon_result = ALL_field_recon_result.^4;
volumeViewer(ALL_field_recon_result)



mask = ALL_field_recon_result;
mask = imgaussfilt3(mask, 2);
mask(mask<0.4)=0;
volumeViewer(mask)