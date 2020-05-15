function ulensPattern = ulensTransmittance(Camera, Resolution)

% compute lens transmittance function for one micro-lens (consitent with Advanced Optics Theory book and Broxton's paper)
ulensPattern = zeros( length(Resolution.yMLspace), length(Resolution.xMLspace), length(Camera.fm) );
for j = 1: length(Camera.fm)
    for a=1:length(Resolution.yMLspace)
        for b=1:length(Resolution.xMLspace)
            x1 = Resolution.yMLspace(a);
            x2 = Resolution.xMLspace(b);
            xL2norm = x1^2 + x2^2;
            if Camera.useMLAPhaseMask
                %position on phase mask is normalized by the actual size
                pmShift = -1i*5*(x1^3+x2^3)/abs(Camera.lensPitch/2)^3;
                ulensPattern(a,b,j) = exp(-1i*Camera.k/(2*Camera.fm(j))*xL2norm+pmShift);
            else
                ulensPattern(a,b,j) = exp(-1i*Camera.k/(2*Camera.fm(j))*xL2norm);
            end
        end
    end
end
% Resolution.mask the pattern, to avoid overlaping when applying it to the whole image
if (Resolution.maskFlag == 0)
    [x,y] = meshgrid(Resolution.yMLspace,Resolution.xMLspace);
end
for j = 1: length(Camera.fm)
    patternML_single = ulensPattern(:,:,j);
    if (Resolution.maskFlag == 1)
        patternML_single(Resolution.sensMask == 0) = 0;
    else
        % todo: something :-?
        patternML_single((sqrt(x.*x+y.*y) >= Camera.lensPitch/2  - 4)) = 0;
    end
    ulensPattern(:,:,j) = patternML_single;
end