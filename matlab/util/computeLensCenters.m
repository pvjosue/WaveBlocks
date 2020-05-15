function lensletCenters = computeLensCenters(LensletGridModel, TextureGridModel, sensorRes, focus, gridType)   

    centersPixels = LFBuildGrid(LensletGridModel, gridType);
    centerOfSensor = round([size(centersPixels,1)/2,size(centersPixels,2)/2]); 
    
    if(strcmp(focus, 'multi'))
        centersPixels = addLensTypes(centersPixels, centerOfSensor);
    end
    
    % Note: new_centers_pixels contains x coords in dim 1 and y coords in
    % dim 2; so we interchange those to account for the convention: 
    % 1 = vertical = row = y = aa
    
    % lenslets centers on the sensor in pixels
    lensletCenters.px = centersPixels;    
    centerOffset = [centersPixels(centerOfSensor(1), centerOfSensor(2),2), centersPixels(centerOfSensor(1), centerOfSensor(2),1)];
    lensletCenters.offset = centerOffset;
    
    lensletCenters.px(:,:,1) = (centersPixels(:,:,2) - centerOffset(1));
    lensletCenters.px(:,:,2) = (centersPixels(:,:,1) - centerOffset(2));
    
    % lenslets centers on the sensor in (um) needed for C++
    lensletCenters.metric = centersPixels;
    lensletCenters.metric(:,:,1) = lensletCenters.px(:,:,1) * sensorRes(1);
    lensletCenters.metric(:,:,2) = lensletCenters.px(:,:,2) * sensorRes(2);
    
    % centers of repetition texture patches in voxels (different of the lenslets center when sensor/texture reesolutions are different)
    centersVoxels = LFBuildGrid(TextureGridModel, gridType);
    centerOfTexture = round([size(centersVoxels,1)/2,size(centersVoxels,2)/2]); 
    centerOffset = [centersVoxels(centerOfTexture(1), centerOfTexture(2),2), centersVoxels(centerOfTexture(1), centerOfTexture(2),1)];
    lensletCenters.vox(:,:,1) = (centersVoxels(:,:,2) - centerOffset(1));
    lensletCenters.vox(:,:,2) = (centersVoxels(:,:,1) - centerOffset(2));
    
    
%     lensletCenters.vox = lensletCenters.metric;
%     lensletCenters.vox(:,:,1) = lensletCenters.metric(:,:,1) / texRes(1);
%     lensletCenters.vox(:,:,2) = lensletCenters.metric(:,:,2) / texRes(2);
    
