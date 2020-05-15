function [f1,H] = fresnel2DBilinear(f0, sensorRes, z, lambda, idealSampling)
%% Computes the final Lightfield PSF
if z==0
    f1 = f0;
    h = zeros(size(f0));
    H = zeros(size(f0));
    rho = zeros(size(f0));
    return;
end
if (idealSampling)
    %% Ideal sampling rate (compute the impulse response h) -> computational Fourier Optics book
    paddAmount = 0;%ceil(size(f0,1));
    f0 = padarray(f0,[paddAmount,paddAmount],0);
    Lx = size(f0,1)*sensorRes(1);
    Ly = size(f0,2)*sensorRes(2);
    k = 2*pi/lambda;

    ideal_rate = abs([lambda*z/Lx, lambda*z/Ly]);
    ideal_samples_no = min([2500,2500],[Lx Ly]./ideal_rate);%ceil([Lx Ly]);%size(f0);%min(ceil([Lx Ly]./ideal_rate), Lx/0.5);
    ideal_samples_no = ideal_samples_no + (1-mod(ideal_samples_no,2));
    rate = [Lx, Ly]./ideal_samples_no;
    % 
    u = [sort(0:-rate(1):-Lx/2), rate(1):rate(1):Lx/2];
    v = [sort(0:-rate(2):-Ly/2), rate(2):rate(2):Ly/2];
    [x, y] = meshgrid(v,u);
%     h = exp(1i*k*z)/(1i*lambda*z)*exp(1i * k/(2*z)*((x).^2+y.^2)); % Fresnel IR, check out Rayleigh IR
    
% Rayleight-Sommerfield approach, exact even for shorter distances
    rho = sqrt(x.^2 + y.^2 + z.^2);
    h = z./(1i*lambda.*rho.^2).* exp(1i*k.*sign(z)*rho);
%     h = imgaussfilt(real(h),1) + 1i*imgaussfilt(imag(h),1);
    H = fft2(h)*rate(1)*rate(2);
%     figure(1);
%     subplot(2,2,3); imagesc(abs(h));
%     subplot(2,2,4); imagesc(abs(H));
    f1 = ifftshift(ifft2( fft2(imresize(f0, ideal_samples_no, 'bilinear', 'Antialiasing',false)) .* H )); 
    H = f1;
    f1 = imresize(f1, size(f0), 'bilinear', 'Antialiasing',false);
    if paddAmount~=0
        f1 = f1(paddAmount:end-paddAmount-1,paddAmount:end-paddAmount-1);
    end
else
    %% Original (compute the Transfer Fucntion H)
    Nx = size(f0,1);
    Ny = size(f0,2);
    k = 2*pi/lambda;
%     sensorRes = abs([lambda*z/(Nx*single(sensorRes(1))), lambda*z/(Ny*single(sensorRes(1)))]);
    % spacial frequencies in x and y direction
    du = 1./(Nx*single(sensorRes(1)));
    u = [ceil(-Nx/2):-1 0:ceil(Nx/2)-1]*du; 
    dv = 1./(Ny*sensorRes(2));
    v = [ceil(-Ny/2):-1 0:ceil(Ny/2)-1 ]*dv; 

    % transfer function for Fresnel diffraction integral
%     H = exp(-1i*2*pi^2*(repmat(u',1,length(v)).^2+repmat(v,length(u),1).^2)*z/k); 

    % transfer function for Rayleigh diffraction integral
    H = exp(1i*sqrt(1-lambda^2*(repmat(u',1,length(v)).^2+repmat(v,length(u),1).^2))*z*k); 
    rho = 0;
    % final Lightfield PSF -> sensor image
    f1 = fftshift(exp(1i*k*z)*ifftshift(ifft2(fftshift(fft2((f0))) .* H )));
end 