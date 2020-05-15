function psf = RelaySLMTest(U, depth, LensFocal, LensDiameter, phasemask, pmNPix, pmPixSize, Camera)
% U0 = psfWAVE_STACK(:,:,1);
% LensFocal = 150000;
% LensDiameter = 50800;    % diameter of lens
% pmNPix = [1080,1920];
% pmPixSize = 8.5;

lambda = Camera.wavelenght;

kn = 2*pi/lambda*Camera.n; %where we have water dipping objective

U0 = fresnel2DBilinear(U, Camera.sensorRes, -depth, lambda, 1);

N = Camera.pmNsamples; % image samples
mid = (N+1)/2;
Ninput = size(U,1);
%physical length of sampled input field at U_0 in micrometer
LU0 = size(U0,1) * Camera.sensorRes(1);
%x and y points
x=linspace(-LU0/2, LU0/2, N);
y=x; 
[X,Y]=meshgrid(x,y);


pmSize = pmPixSize*pmNPix(1); % lateral size of phase mask
% pmSizePixels = round(pmSize/Resolution.sensorRes(1));

%pupil function of lens
circ = @(x,y,r) (x.^2.+y.^2.<r.^2)*1.0;

% Propagate until lens
[U1minus, LU1] = lensProp(U0, LU0, Camera.wavelenght, LensFocal);

% Resample U1minus
U1minus = imresize(U1minus, size(X),'method','bilinear');
phasemaskUp = imresize(phasemask, size(X));

coeffU1minus = -1i*exp(1i*kn*LensFocal)/Camera.wavelenght/LensFocal;
U1minus = coeffU1minus.*U1minus;

%shift due to the phase mask
%we divide by L1 and multiply by L2 to rescale our x positions 
%the argument x and y of our phase mask are normalized to the real phase
%masks dimension
%we choose here a 20mmx20mm phase mask so divide by 10mm
% pmscale = LU1/LU0/(pmSize/2);


% pmshiftOrig = ones(size(X));

% sign PM
% a = Camera.pmShift;
% k = Camera.pmK;
% pmshiftx = exp(1i*2*pi*a.*sign(X).*abs(X./LensDiameter/2).^k);
% pmshifty = exp(1i*2*pi*a.*sign(Y).*abs(Y./LensDiameter/2).^k);
% pmshiftOrig = pmshiftx.*pmshifty;

% Spiral PM
% a = Camera.pmShift;
% pmshiftOrig = a * (X.^2+Y.^2) .* atan2(Y,X);

% Put phase mask with SLM pixel size = 
% pmshiftTemp = imresize(pmshiftOrig, Camera.pixelPitch/pmPixSize);
% pmshift = imresize(pmshiftTemp, size(pmshiftOrig),'box');
% z = fspecial('gaussian', [N N], 70);
% pmshift = pmshift.*z;

%we also cut the field in the middle because of the back aperture of the
%objective
U1plus = (U1minus.*phasemaskUp).*circ(X./LU0.*LU1, Y./LU0.*LU1, LensDiameter./2);

%propagation through second lens
[U2minus, LU2] = lensProp(U1plus, LU1, lambda, LensFocal);
coeffU2minus = -1i*exp(1i*kn*LensFocal)/lambda/LensFocal;
U2minus = coeffU2minus.*U2minus;

%we have to reshape our U2minus since it is larger and has more points
%than our specified output format of xspace and yspace
cut = round((LU0/2)/(LU2/2)*(N+1)/2);
outImg = U2minus(mid-cut:mid+cut, mid-cut:mid+cut);

%downsample with bicubic interpolation our psf at the back focal plane of
%tube lens
psf = imresize(outImg,[Ninput Ninput], 'method', 'bilinear');

% figure(33);
% subplottight(2,2,1);
% imagesc(abs(U0));
% subplottight(2,2,2);
% imagesc(imag(pmshift));
% subplottight(2,2,3);
% imagesc(abs(outImg));