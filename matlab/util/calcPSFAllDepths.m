
function [psfWAVE_STACK] = calcPSFAllDepths(Camera, Resolution)
% calcPSFAllResolution.depths Computes PSF for all Resolution.depths
%   Computes PSF for all Resolution.depths, exploiting the symetry of Resolution.depths at
%   the same absolute distance to the zero plane

% Offsets the depths in the case of a LF 2.0 setup, offset_fobj is zero for
% 1.0 setup
Resolution.depths = Resolution.depths + Camera.offsetFobj;

psfWAVE_STACK = zeros(length(Resolution.yspace), length(Resolution.xspace), length(Resolution.depths));
disp('Computing PSF for main objective:');
disp('...');
for i = 1: length(Resolution.depths)
    compute_psf = 1;
    idx = 0;
    % Check if the abs(depth) was previoulsy computed, as depths symetric
    % to the zero are just conjugates.
    if i>1 && Camera.usePhaseMask == 0
        idx = find(abs(Resolution.depths(1:i-1)) == abs(Resolution.depths(i)));
        if ~isempty(idx)
            compute_psf = 0;
        end
    end
    
    % If depth has not been computed, compute it
    if compute_psf==1
        tic
        if Camera.usePhaseMask == 0
            psfWAVE = calcPSF(0, 0, Resolution.depths(i), Camera, Resolution);
        else
            psfWAVE = calcPSFwithPhaseMask(0, 0, Resolution.depths(i), Camera, Resolution);
        end
        disp(['PSF: ',num2str(i),'/',num2str(length(Resolution.depths)),' in ',num2str(toc),'s']);
    else
        % if it's exactly the same depth just copy
        if Resolution.depths(i)==Resolution.depths(idx)
            psfWAVE = psfWAVE_STACK(:,:,idx);
        else
            % if it's the negative, conjugate
            psfWAVE = conj(psfWAVE_STACK(:,:,idx));
        end
        disp(['PSF: ',num2str(i),'/',num2str(length(Resolution.depths)),' already computed for depth ',num2str(Resolution.depths(idx))]);
    end
    psfWAVE_STACK(:,:,i)  = psfWAVE;
    
end
disp('...');

end
