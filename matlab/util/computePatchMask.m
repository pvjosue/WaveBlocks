function mask = computePatchMask(spacing, gridType, res, patchRad, Nnum)

ysensorspace = [(-floor(Nnum(1)/2)):1:floor(Nnum(1)/2)];
xsensorspace = [(-floor(Nnum(2)/2)):1:floor(Nnum(2)/2)];
[x,y] = meshgrid(res(1)*ysensorspace, res(2)*xsensorspace);
mask = sqrt(x.*x+y.*y) < patchRad;

% Resolve for holes and overlaps
mask = fixMask(mask, spacing, gridType);