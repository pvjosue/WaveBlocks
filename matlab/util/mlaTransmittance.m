function MLARRAY = mlaTransmittance(Camera, Resolution, ulensPattern)

%% Compute the ML array as a grid of phase/amplitude masks corresponding to mlens
ylength = length(Resolution.yspace);
xlength = length(Resolution.xspace);

% build a slightly bigger array to make sure there are no empty borders (due to the hexagonal packing)
ylength_extended = length(Resolution.yspace) + 2*length(Resolution.yMLspace);
xlength_extended = length(Resolution.xspace) + 2*length(Resolution.xMLspace);

% offset centers
usedLensletCentersOff(:,:,1) = round(Resolution.usedLensletCenters.px(:,:,1)) + ceil(ylength_extended / 2);
usedLensletCentersOff(:,:,2) = round(Resolution.usedLensletCenters.px(:,:,2)) + ceil(xlength_extended / 2);
 
% in case of multi focus arrays, lens center also have a type
if (size(Resolution.usedLensletCenters.px, 3) == 3)
    usedLensletCentersOff(:,:,3) = Resolution.usedLensletCenters.px(:,:,3);
end
% activate lenslet centers, e.g. set to 1
MLspace = zeros( ylength_extended, xlength_extended );
MLcenters = MLspace;
if (size(Resolution.usedLensletCenters.px, 3) == 3)
    MLcenters_types = MLspace;
end
for a = 1:size(usedLensletCentersOff,1) %ceil(size(usedLensletCentersOff,1)/2) % TODO: check this thing for the MLA_zero case
    for b = 1:size(usedLensletCentersOff,2) % ceil(size(usedLensletCentersOff,2)/2) 
        if( (usedLensletCentersOff(a,b,1)<1) || (usedLensletCentersOff(a,b,1) > ylength_extended) || ...
              (usedLensletCentersOff(a,b,2)<1) || (usedLensletCentersOff(a,b,2) > xlength_extended)  )
          continue
        end
        MLcenters( usedLensletCentersOff(a,b,1), usedLensletCentersOff(a,b,2)) = 1;
        if (size(Resolution.usedLensletCenters.px, 3) == 3)
            MLcenters_types( usedLensletCentersOff(a,b,1), usedLensletCentersOff(a,b,2)) = ...
                usedLensletCentersOff(a,b,3);
        end
    end
end

% apply the mlens pattern at every ml center
MLARRAY_extended = MLspace;
for j = 1 : length(Camera.fm)
    tempSlice = zeros(size(MLcenters));
    if (size(Resolution.usedLensletCenters.px, 3) == 3)
        tempSlice(MLcenters_types == j) = 1;
    else
        tempSlice = MLcenters;
    end
    MLARRAY_extended  = MLARRAY_extended +  conv2(tempSlice, ulensPattern(:,:,j), 'same');
end

% get back the center part of the array (ylength, xlength)
inner_rows = [ceil(ylength_extended/2) - floor(ylength/2) : ceil(ylength_extended/2) + floor(ylength/2)];
inner_cols = [ceil(xlength_extended/2) - floor(xlength/2) : ceil(xlength_extended/2) + floor(xlength/2)];
MLARRAY = MLARRAY_extended;%(inner_rows, inner_cols);
