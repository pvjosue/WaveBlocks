function f1 = fresnel2D(f0, sensorRes, z, lambda, idealSampling)
%% Computes the final Lightfield PSF
if z==0
    f1 = f0;
    return;
end
if (idealSampling)
    %% Ideal sampling rate (compute the impulse response h) -> computational Fourier Optics book
    Lx = size(f0,1)*sensorRes(1);
    Ly = size(f0,2)*sensorRes(2);
    k = 2*pi/lambda;

    ideal_rate = abs([lambda*z/Lx, lambda*z/Ly]);
    ideal_samples_no = ceil([Lx Ly]./ideal_rate);
    ideal_samples_no = ideal_samples_no + (1-mod(ideal_samples_no,2));
    rate = [Lx, Ly]./ideal_samples_no;
    % 
    u = [sort(0:-rate(1):-Lx/2), rate(1):rate(1):Lx/2];
    v = [sort(0:-rate(2):-Ly/2), rate(2):rate(2):Ly/2];
    [x, y] = meshgrid(v,u);
    h = exp(1i*k*z)/(1i*lambda*z)*exp(1i * k/(2*z)*((x).^2+y.^2)); % Fresnel IR, check out Rayleigh IR

    H = fft2((h))*rate(1)*rate(2);
    f1 = (ifftshift(ifft2( fft2((imresize(f0, ideal_samples_no, 'bicubic'))) .* H ))); 
    f1 = imresize(f1, size(f0), 'bicubic');
else
    %% Original (compute the Transfer Fucntion H)
    Nx = size(f0,1);
    Ny = size(f0,2);
    k = 2*pi/lambda;

    % spacial frequencies in x and y direction
    du = 1./(Nx*single(sensorRes(1)));
    u = [0:ceil(Nx/2)-1 ceil(-Nx/2):-1]*du; 
    dv = 1./(Ny*sensorRes(2));
    v = [0:ceil(Ny/2)-1 ceil(-Ny/2):-1]*dv; 

    % transfer function for Fresnel diffraction integral
    % H = exp(-1i*2*pi^2*(repmat(u',1,length(v)).^2+repmat(v,length(u),1).^2)*z/k); 

    % transfer function for Rayleigh diffraction integral
    H = exp(1i*sqrt(1-lambda^2*(repmat(u',1,length(v)).^2+repmat(v,length(u),1).^2))*z*k); 

    % final Lightfield PSF -> sensor image
    f1 = exp(1i*k*z)*(ifft2(fft2((f0)) .* H ));
end 