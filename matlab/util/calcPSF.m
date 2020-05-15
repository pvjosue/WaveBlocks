function psf = calcPSF(p1, p2, p3, Camera, Resolution)

%% Camputes the PSF at the Camera.NAtive image plane for a source point (p1, p2, p3) using wave optics theory
% fobj -> main lens focal length
% Camera.NA -> main lens numerical aperture
% yspace, xspace -> sensor space coordinates
% Camera.WaveLength -> wavelength
% M -> objective magnification
% n -> refractive index
k = 2*pi*Camera.n/Camera.WaveLength; % wave number
alpha = asin(Camera.NA/Camera.n); % maximal half-angle of the cone of light entering the lens
demag = 1/Camera.M;

ylength = length(Resolution.yspace);
xlength = length(Resolution.xspace);
centerPT = ceil(length(Resolution.yspace)/2);
pattern = zeros(centerPT, centerPT);
zeroline = zeros(1, centerPT); 

yspace = Resolution.yspace(1:centerPT);
xspace = Resolution.xspace(1:centerPT);

d1 = Camera.dof - p3; %% as per Advanced optics book (3.4.2)

% compute the PSF for one quarted on the sensor area and replicate (symmetry)
u = 4*k*(p3*1)*(sin(alpha/2)^2);
Koi = demag/((d1*Camera.WaveLength)^2)*exp(-1i*u/(4*(sin(alpha/2)^2)));

parfor a = 1:centerPT
    patternLine = zeroline;
        y = yspace(a);         
    for b = a:centerPT 
        x = xspace(b); 
        xL2normsq = (((y+Camera.M*p1)^2+(x+Camera.M*p2)^2)^0.5)/Camera.M;        
       
        % radial and axial optical coodinates
        v = k*xL2normsq*sin(alpha);    
        
        % compute PSF;
        % integrate over theta: 0->alpha
        intgrand = @(theta) (sqrt(cos(theta))) .* (1+cos(theta))  .*  (exp((1i*u/2)* (sin(theta/2).^2) / (sin(alpha/2)^2))) ...
            .*  (besselj(0, sin(theta)/sin(alpha)*v))  .*  (sin(theta)); % zeroth order Bessel function of the first kind
        I0 = integral(@(theta)intgrand (theta),0,alpha);          
        patternLine(1,b) = Koi*I0;
    end
    pattern(a,:) = patternLine;    
end
% setup the whole(sensor size) PSF
patternA = pattern;
patternAt = fliplr(patternA);
pattern3D = zeros(xlength,ylength, 4);
pattern3D(1:centerPT,1:centerPT,1) = pattern;
pattern3D( (1:centerPT), (centerPT:end),1 ) = patternAt;
pattern3D(:,:,2) = rot90( pattern3D(:,:,1) , -1);
pattern3D(:,:,3) = rot90( pattern3D(:,:,1) , -2);
pattern3D(:,:,4) = rot90( pattern3D(:,:,1) , -3);

% pattern = max(pattern3D,[],3);
% for the zero plane as there's no phase the result is real, 
% then max grabs the zeros instead of the negative values
% This for loops are used instead
% now only two diagonal lines overlap, so only evaluate the diagonals
pattern = sum(pattern3D,3);
for i=1:size(pattern3D,1)
    % first diagonal
    [~,index] = max(abs(pattern3D(i,i,:)));
    pattern(i,i) = pattern3D(i,i,index);
    % second diagonal
    [~,index] = max(abs(pattern3D(i,1+size(pattern,2)-i,:)));
    pattern(i,1+size(pattern,2)-i) = pattern3D(i,1+size(pattern,2)-i,index);
end
psf = pattern;
