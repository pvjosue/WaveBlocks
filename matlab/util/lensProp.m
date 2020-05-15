function[u2,L2]=lensProp(u1,L1,lambda,z,appDiam,seidelCoeffs, seidelAxis)
%% this function propagates from a incoming field to outcoming field
%% we use it for lens propagation
%% it's based on the function propFF out of the book "Computational Fourier
%% Optics. A MATLAB Tutorial". There you can find more information.

% assumes uniform sampling
% u1 - source plane field
% L1 - source plane side length
% lambda - wavelength
% z - propagation distance
% L2 - observation plane side length
% u2 - observation plane field
%
%get input field array size
[M,N] = size(u1);
%source sample interval
dx1 = L1/M;

fu=-1/(2*dx1):1/L1:1/(2*dx1)-(1/L1); %image freq coords
[Fu,Fv]=meshgrid(fu,fu);

wxp=appDiam/2; % apperture radius
lz=lambda*z;
% Pupil function
P=circ(sqrt(Fu.^2+Fv.^2)*lz/wxp);

% compute aberrated pupil
if exist('seidelCoeffs','var')
    k=2*pi/lambda; %wavenumber
    W=seidel_5(seidelAxis(1),seidelAxis(2),-lz*Fu/wxp,-lz*Fv/wxp,...
        seidelCoeffs(1),seidelCoeffs(2),seidelCoeffs(3),...
        seidelCoeffs(4),seidelCoeffs(5),seidelCoeffs(6));
    P = P.*exp(-j*k*W);
end

%obs sidelength
L2 = lambda*z/dx1;

%obs sample interval
dx2 = lambda*z/L1;

% mask input by pupil function
u1 = u1.*P;

u1(isnan(u1)) = 0;
%output field
%for odd length this
u2 = fftshift(fft2(ifftshift(u1))).*dx1.^2;

%this works for even length
%u2 = ifftshift(fft2(fftshift(u1))).*dx1.^2;

end
