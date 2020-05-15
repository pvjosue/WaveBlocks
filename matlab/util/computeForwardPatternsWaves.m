function H = computeForwardPatternsWaves(psfWAVE_STACK, MLARRAY, Camera, Resolution)
% ComputeForwardPatternsWaves: Compute the forward projection for every source point (aa,bb,c) of our square shaped patch around the central microlens
% Take the PSF at native plane (psfWAVE_STACK), pass it through the
% microlens array (MLARRAY), and finally propagate it to the sensor

% for regular grids compute the psf for only one quarter of coordinates (due to symmetry)
if strcmp(Camera.range, 'quarter')
    coordsRange  = Resolution.TexNnum_half;
else
    coordsRange  = Resolution.TexNnum;
end

Nnum_half_coord = Resolution.TexNnum_half ./ Resolution.texScaleFactor;
sensorRes = Resolution.sensorRes;
Nnum = Resolution.Nnum;

% Resolution.texScaleFactor(1/2) is actually (Resolution.texRes(1/2) * M / Resolution.sensorRes(1/2))^-1  
H = cell( coordsRange(1), coordsRange(2), length(Resolution.depths) ); 
for c = 1:length(Resolution.depths)
    psfREF = psfWAVE_STACK(:,:,c);
    tic
    for aa_tex = 1:coordsRange(1)
            aa_sensor = aa_tex / Resolution.texScaleFactor(1);
        for bb_tex = 1:coordsRange(2)
            bb_sensor = bb_tex / Resolution.texScaleFactor(2);
            % shift the native plane PSF at every (aa, bb) position (native plane PSF is shift invariant)
            MLARRAYSHIFT = imShift2(MLARRAY, round(-(aa_sensor-Nnum_half_coord(1))), round(-(bb_sensor-Nnum_half_coord(2))) );
            MLARRAYSHIFT = MLARRAYSHIFT(Nnum(1)+1:end-Nnum(1),Nnum(2)+1:end-Nnum(2));
           	psfMLA = psfREF.*MLARRAYSHIFT;
            
            % phase mask
            if isfield(Camera,'SLMscale')
                if Camera.SLMscale~=0
                    psfMLA = RelaySLM(psfMLA, Camera);
                end
            end
            % propagate the response to the sensor via Fresnel diffraction
            LFpsfAtSensor = fresnel2DBilinear(psfMLA, sensorRes, Camera.mla2sensor, Camera.WaveLength, 1);
            
            % shift the response back to center (we need the patterns centered for convolution)
            LFpsf = LFpsfAtSensor;%imShift2(LFpsfAtSensor, round(-(aa_sensor-Nnum_half_coord(1))), round(-(bb_sensor-Nnum_half_coord(2))) );
            
            % Crop central part of LFpsf to avoid having empty border from
            % translating the image
%             LFpsf = LFpsf(Nnum(1)+1:end-Nnum(1),Nnum(2)+1:end-Nnum(2));
            
            % store the response pattern 
            psf = abs(double(LFpsf).^2);
%             max_slice = max(psf(:));
            % Clamp values smaller than tol 
%             psf(psf < max_slice*0.05) = 0;
            H{aa_tex,bb_tex,c} = sparse(psf);
        end
    end
    toc
    disp(['Forward Patterns, depth: ', num2str(c), '/', num2str(length(Resolution.depths))]);
end
