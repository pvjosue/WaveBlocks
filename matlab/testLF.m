%% %%%%%%%% Create PSFs and transmitance image, export it to Python
addpath(genpath('util/'));
depthRange = [-25,25];
depthStep = 25;
dephts = depthRange(1):depthStep:depthRange(2);
nDepths = length(dephts);
imgSize = 151;
fishes = h5read('../config_files/fish_phantom.h5','/fish_phantom');
phantomFish = imresize3(fishes, [imgSize, imgSize, nDepths]);
phantomFish = phantomFish-min(phantomFish(:));
volume = phantomFish./max(phantomFish(:));
% volume = zeros(size(volume));
% volume(77,60) = 1;
% volume = repmat(phantom,1,1,nDepths);% ones(265,265, nDepths);

sensorRes = 3*3.45;
lambda = 0.63;
M = 20;
NA = 0.45;
ftl = 165000;
fm = 2500;
tl2mla = ftl;
lensPitch = 112;
mla2sensor = 2000;

spacingPx = floor(lensPitch/sensorRes); % distance between lenstels centers in pixels
spacingPx = spacingPx + (1- mod(spacingPx,2));
Camera = setCameraParams(M, NA, 'reg', 'single', 2, ftl, tl2mla, fm,...
    mla2sensor, lensPitch, sensorRes, spacingPx, 1, lambda*10^3, 0, 0);

NewLensletGridModel = setGridModel(ceil(spacingPx), 1, 150, 150,...
    0, 0, 0, 'horz', 'reg');

[TextureGridModel] = setGridModel(spacingPx, 1, 150, 150,...
    0, 0, 0, 'horz', 'reg');

Depth.depthStep = depthStep;
Depth.depthRange = depthRange;
Resolution = computeResolution(NewLensletGridModel, TextureGridModel, Camera, Depth, 0);
lensletCenters = computeLensCenters(NewLensletGridModel, TextureGridModel, Resolution.sensorRes, Camera.focus, Camera.gridType);

% Object/Sensor and ML space
PSFsize = 3;
IMGsize_half = max( Resolution.Nnum(2)*(PSFsize), 2*Resolution.Nnum(2)); % PSF size in pixels
disp(['Size of PSF IMAGE = ' num2str(IMGsize_half*2+1) 'X' num2str(IMGsize_half*2+1) ' [Pixels]']);

% yspace/xspace are grid points on the sensor for which the PSF is going to be computed.
% yMLspace/xMLspace are the pixel positions inside a single ML.
yspace = Resolution.sensorRes(1)*[-IMGsize_half:1:IMGsize_half];
xspace = Resolution.sensorRes(2)*[-IMGsize_half:1:IMGsize_half];
yMLspace = Resolution.sensorRes(1)* [- Resolution.Nnum_half(1) + 1 : 1 : Resolution.Nnum_half(1) - 1];
xMLspace = Resolution.sensorRes(2)* [- Resolution.Nnum_half(2) + 1 : 1 : Resolution.Nnum_half(2) - 1];

% Compute Patterns
% setup parallel pool
pool = gcp('nocreate');
if isempty(pool)
    parpool;
end

depths = depthRange(1) : depthStep : depthRange(2); % depths
Resolution.depths = depths;
Resolution.yspace = yspace;
Resolution.xspace = xspace;
Resolution.yMLspace = yMLspace;
Resolution.xMLspace = xMLspace;
Resolution.maskFlag = 1;
% compute native plane PSF for every depth
psfWAVE_STACK = calcPSFAllDepths(Camera, Resolution);
% Compute MLA transmitance pattern
Resolution.usedLensletCenters = getUsedCenters(size(volume,1), lensletCenters);
ulensPattern = ulensTransmittance(Camera, Resolution);
MLARRAY = mlaTransmittance(Camera, Resolution, ulensPattern);

H = computeForwardPatternsWaves(psfWAVE_STACK, MLARRAY, Camera, Resolution);
H = NormLFPSF(H);

% Write to disk
outPath = '../config_files/';
psfFile = 'psfLFGT.h5';
% Psf and transmittance file name
% accomodate complex into last dimension
PSFWaveStackOut = zeros(size(psfWAVE_STACK,1),size(psfWAVE_STACK,2),size(psfWAVE_STACK,3),2);
PSFWaveStackOut(:,:,:,1) = single(real(psfWAVE_STACK));
PSFWaveStackOut(:,:,:,2) = single(imag(psfWAVE_STACK));

% same with transmitanse
MLARRAYOut = zeros(size(MLARRAY,1),size(MLARRAY,2),2);
MLARRAYOut(:,:,1) = single(real(MLARRAY));
MLARRAYOut(:,:,2) = single(imag(MLARRAY));

Resolution.M = Camera.M;
fwd = forwardProjectACC( H, volume, lensletCenters, Resolution, [imgSize,imgSize], 'quarter');

volOut = single(permute(volume,[2,1,3]));


% Run recon
S = @(img) real(sum(img(:)))+imag(sum(img(:)));
Si = @(M,fobj,So) M*(fobj+ M*fobj-M*So);

figure;imagesc(fwd)
%%
delete([outPath,psfFile]);
h5create([outPath,psfFile], '/PSFWaveStack', size(PSFWaveStackOut),'Datatype','single')
h5create([outPath,psfFile], '/MLATransmitance', size(MLARRAYOut))
h5create([outPath,psfFile], '/volume', size(volOut),'Datatype','single');
h5create([outPath,psfFile], '/sensorRes', 2);
h5write([outPath,psfFile], '/sensorRes', single(Resolution.sensorRes));
h5create([outPath,psfFile], '/Nnum', 2);
h5write([outPath,psfFile], '/Nnum', single(Resolution.Nnum));
h5create([outPath,psfFile], '/mla2sensor', 1);
h5write([outPath,psfFile], '/mla2sensor', single(mla2sensor));
h5create([outPath,psfFile], '/NA', 1);
h5write([outPath,psfFile], '/NA', single(Camera.NA));
h5create([outPath,psfFile], '/M', 1);
h5write([outPath,psfFile], '/M', single(Camera.M));
h5create([outPath,psfFile], '/fm', 1);
h5write([outPath,psfFile], '/fm', single(Camera.fm));
h5create([outPath,psfFile], '/ftl', 1);
h5write([outPath,psfFile], '/ftl', single(Camera.ftl));
h5create([outPath,psfFile], '/wavelenght', 1);
h5write([outPath,psfFile], '/wavelenght', single(lambda));
h5write([outPath,psfFile], '/PSFWaveStack', PSFWaveStackOut)
h5write([outPath,psfFile], '/MLATransmitance', MLARRAYOut)
h5write([outPath,psfFile], '/volume', single(volOut));
h5create([outPath,psfFile], '/LFImage', size(fwd'));
h5write([outPath,psfFile], '/LFImage', single(fwd'));
