function usedLensletCenters = getUsedCenters(PSFsize, lensletCenters)

    usedLens = [PSFsize + 3, PSFsize + 3]; 
    centerOfMatrix = round([size(lensletCenters.px,1)/2,size(lensletCenters.px,2)/2]); 
    usedLensletIndeces_y = centerOfMatrix(1)-usedLens(1):centerOfMatrix(1)+usedLens(1);
    usedLensletIndeces_x = centerOfMatrix(2)-usedLens(2):centerOfMatrix(2)+usedLens(2);
    
    % make sure indices are integer (when psf size is bigger than U/Vmax)
    usedLensletIndeces_y = usedLensletIndeces_y((usedLensletIndeces_y >= 1) & (usedLensletIndeces_y <=size(lensletCenters.px,1)));
    usedLensletIndeces_x = usedLensletIndeces_x((usedLensletIndeces_x >= 1) & (usedLensletIndeces_x <=size(lensletCenters.px,2)));
    
    usedLensletCenters.px = lensletCenters.px(usedLensletIndeces_y,usedLensletIndeces_x,:);
    usedLensletCenters.vox = lensletCenters.vox(usedLensletIndeces_y,usedLensletIndeces_x,:);
