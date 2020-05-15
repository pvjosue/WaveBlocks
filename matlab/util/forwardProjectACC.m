function Projection = forwardProjectACC( H, realSpace, lensCenters, Resolution, imgSize, range)
% forwardProjectACC: Forward projects a volume into a lenslet image,
% simulating the behavior of the microscope.

TexNnum = Resolution.TexNnum;
TexNnum_half = Resolution.TexNnum_half;

%% Retrieve sensor image and 3D scene containers sizes
nDepths = size(H,3);
texSize = [size(realSpace,1), size(realSpace,2)];

% imgSize = floor(texSize./Resolution.texScaleFactor);
% imgSize = imgSize + (1-mod(imgSize,2));

% offset centers to match the image and 3D backprojection
offsetImg = ceil(imgSize./2);
offsetVol = ceil(texSize./2);
lensYvox = lensCenters.metric(:,:,1)/(Resolution.texRes(1)*Resolution.M);% + offsetVol(1);
lensXvox = lensCenters.metric(:,:,2)/(Resolution.texRes(1)*Resolution.M);% + offsetVol(2);

%% Precompute positions in the vol/image where to apply the different PSFs patterns
indicesTex = cell(TexNnum(1),TexNnum(2));
indicesImg = cell(TexNnum(1),TexNnum(2));

for aa_tex = 1:TexNnum(1)
    for bb_tex = 1:TexNnum(2)
        % Find voxels to sample relative to the lenslet centers in the
        % volume
        lensXvoxCurrent = round(lensYvox - TexNnum_half(1) + aa_tex); % why x becomes y here?
        lensYvoxCurrent = round(lensXvox - TexNnum_half(2) + bb_tex);
        
        % convert lenses to img space
        lensXpxCurrent = lensXvoxCurrent;% - offsetVol(1);
        lensXpxCurrent = round(lensXpxCurrent/Resolution.texScaleFactor(1));
        lensXpxCurrent = lensXpxCurrent + offsetImg(1);
        
        lensYpxCurrent = lensYvoxCurrent;% - offsetVol(2);
        lensYpxCurrent = round(lensYpxCurrent/Resolution.texScaleFactor(2));
        lensYpxCurrent = lensYpxCurrent + offsetImg(2);
        
        lensXvoxCurrent = lensXvoxCurrent + offsetVol(1);
        lensYvoxCurrent = lensYvoxCurrent + offsetVol(2);
        
        % check for out of image and texture
        validLens = (lensXvoxCurrent <= texSize(1)) & (lensXvoxCurrent > 0) & ...
            (lensYvoxCurrent <= texSize(2)) & (lensYvoxCurrent > 0) & ...
            (lensXpxCurrent <= imgSize(1)) & (lensXpxCurrent > 0) & ...
            (lensYpxCurrent <= imgSize(2)) & (lensYpxCurrent > 0);
        
        lensXvoxCurrent = lensXvoxCurrent(validLens);
        lensYvoxCurrent = lensYvoxCurrent(validLens);
        indicesTex{aa_tex,bb_tex} = sub2ind(texSize,lensXvoxCurrent, lensYvoxCurrent);
        
        lensXpxCurrent = lensXpxCurrent(validLens);
        lensYpxCurrent = lensYpxCurrent(validLens);
        indicesImg{aa_tex,bb_tex} = sub2ind(imgSize,lensXpxCurrent, lensYpxCurrent);
    end
end

% Forwardproject
zeroSpace = zeros(imgSize);
if gpuDeviceCount>0
    Projection = gpuArray(zeros(size(zeroSpace)));
else
    Projection = zeroSpace;
end

for cc = 1:nDepths
%     tic
    disp(['FP depth: ', num2str(cc), '/', num2str(nDepths)]);
    realspaceCurrentDepth = realSpace(:,:,cc);
    
    if sum(realspaceCurrentDepth(:)) == 0
        continue;
    end
    for aa_tex = 1:TexNnum(1)
        
        % Precompute pattern index (for regular grids we computed the fwdprojection patterns only for one quarter of coordinates (due to symmetry))
        aa_new = aa_tex;
        flipX = 0;
        if (strcmp(range, 'quarter') && aa_tex > TexNnum_half(1))
            aa_new = TexNnum(1) - aa_tex + 1;
            flipX = 1;
        end
        
        % Slice PSF's needed for parallel computing
        HcurrentAACC = H(aa_new,:,cc);
%         tic;
        try
            parfor bb_tex = 1:TexNnum(2)
                
                % Forward project from every point once. Avoid overlap (hex grid)
                if Resolution.texMask(aa_tex, bb_tex) == 0
                    continue;
                end
                
                % Precompute pattern index (for regular grids compute the backprojection patterns only for one quarter of coordinates (due to symmetry))
                bb_new = bb_tex;
                flipY = 0;
                if (strcmp(range, 'quarter') && bb_tex > TexNnum_half(2))
                    bb_new = TexNnum(2) - bb_tex + 1;
                    flipY = 1;
                end
                
                % Fetch corresponding PSF for given coordinate behind the
                % lenslet
                Hs = HcurrentAACC{bb_new};
                tempspace = zeros(imgSize);
                tempspace(indicesImg{aa_tex,bb_tex}) = realspaceCurrentDepth(indicesTex{aa_tex,bb_tex});
                
                % Accumulate result
                if sum(tempspace(:)) > 0
                    % Check if computations where made on GPU
                    projectedPattern = sconv2Flip(tempspace, Hs, flipX, flipY, 'same');
                    Projection = Projection + gather(projectedPattern);
%                     disp([num2str(aa_tex),' ', num2str(bb_tex)]);
                    
                end
            end
        
            % Check for GPU errors and sugest a solution
        catch ME
            if (strcmp(ME.identifier,'parallel:gpu:array:OOM'))
                msg = ['Error: reconstuction too big. try with a smaller imager or depthrange. Or run in C++ implementation'];
                causeException = MException('parallel:gpu:array:OOM',msg);
                ME = addCause(ME,causeException);
            end
            rethrow(ME)
        end
%         toc
    end
end
    if gpuDeviceCount > 0
        Projection = gather(Projection);
    end
