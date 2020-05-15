function [psf] = calcPSFwithPhaseMask(p1, p2, p3, Camera, Resolution)
% fobj -> main lens focal length
% NA -> main lens numerical aperture
% yspace, xspace -> sensor space coordinates
% lambda -> wavelength
% M -> objective magnification
% n -> refractive index
fobj = Camera.fobj;
lambda = Camera.WaveLength;
M = Camera.M;
n = Camera.n;
ftl = fobj*M;

k = 2*pi/lambda; % wave number for air
kn = 2*pi/lambda*n; %where we have water dipping objective

dobj = 2*fobj*Camera.NA;
A = 1; %some amplitude
%choose unevent number of points to have a reald midpoint at (2048,2048)
N = 4095; %samples
mid = (N+1)/2;
Ninput = length(Resolution.xspace);
%physical length of sampled input field at U_0 in micrometer
LU0 = 500;%34.1091;
%x and y points
x=linspace(-LU0/2, LU0/2, N);
y=x; 
[X,Y]=meshgrid(x,y);

%pupil function of lens
circ = @(x,y,r) (x.^2.+y.^2.<r.^2)*1.0;

%if input is at front focal plane we've got a delta peak
if p3 == 0
    U0 = zeros(N,N);
    xpos = mid+round((p1/(LU0/2))*N/2);
    ypos = mid+round((p2/(LU0/2))*N/2);
    U0(xpos,ypos) = 1;%*(LU0/N)^2;
else
    %distance from point source to point on lens
    r = sqrt((X-p1).^2.+(Y-p2).^2.+p3.^2);
    
    %p3>0: then we propagate back to the lens.
    if p3 > 0
        r = -1*r;
    end
    %wave at Positon U_0
    U0 = -1i*A*kn/2/pi./r .* exp(1i.*kn.*r);
end

%due to the FFT we've got at scaling factor and LU1 is the length of the
%field at U1minus (so after first lens)
[U1minus, LU1] = lensProp(U0, LU0, lambda, fobj, dobj./2);

coeffU1minus = -1i*exp(1i*k*fobj)/lambda/fobj;
U1minus = coeffU1minus.*U1minus;

%shift due to the phase mask
%we divide by L1 and multiply by L2 to rescale our x positions 
%the argument x and y of our phase mask are normalized to the real phase
%masks dimension
%we choose here a 20mmx20mm phase mask so divide by 10mm
pmscale = LU1/LU0/(10e3);
pmshift = exp(1i*117*(X.^3+Y.^3).*pmscale.^3);

% a = 150000000/pi;
% k = 3;
% 
% pmshiftx = exp(1i*2*pi*a.*sign(x).*abs(x./dobj/2).^k);
% pmshifty = exp(1i*2*pi*a.*sign(y).*abs(y./dobj/2).^k);
% pmshift = pmshiftx'*pmshifty;
%we also cut the field in the middle because of the back aperture of the
%objective
U1plus = U1minus.*circ(X./LU0.*LU1, Y./LU0.*LU1, dobj./2);%.*pmshift;

%figure()
%plot(abs(U1plus(mid,:)).^2)
%propagation through second lens
[U2minus, LU2] = lensProp(U1plus, LU1, lambda, ftl, 25000);
coeffU2minus = -1i*exp(1i*k*ftl)/lambda/ftl;
U2minus = coeffU2minus.*U2minus;

%we have to reshape our U2minus since it is larger and has more points
%than our specified output format of xspace and yspace
cut = round(Resolution.xspace(end)/(LU2/2)*(N+1)/2);
psf = U2minus(mid-cut:mid+cut, mid-cut:mid+cut);

%downsample with bicubic interpolation our psf at the back focal plane of
%tube lens
psf = imresize(psf,[Ninput Ninput], 'bicubic');
