function Camera = setCameraParams(M, NA, gridType, focus, plenoptic, ftl, tube2mla, fm, mla2sensor, lensPitch, pixelPitch, spacingPx, n, waveLength, usePhaseMask, useMLAPhaseMask)
% TODO: make defaults; switch for single/multi focus; microscope/Lytro
Camera = struct;

% microles grid type
Camera.gridType = gridType;
Camera.focus = focus;

% objective params
Camera.M = M; % objective magnification
Camera.NA = NA; % objective aperture
Camera.ftl = ftl; % focal length of tube lens (only for microscopes)
Camera.fobj = Camera.ftl/Camera.M;  %% focal length of objective lens
Camera.Delta_ot = Camera.ftl + Camera.fobj;

% ml array params
Camera.lensPitch = lensPitch; %127; %lenslet pitch
Camera.pixelPitch = pixelPitch; %6.5; %senosr pixel pitch
Camera.spacingPx = spacingPx; %25; %lensPitch/pixelPitch; hardcoded here due to the magnification of relay system (not perfectly 1:1)
Camera.fm = fm; %3000; %focal length of the lenslets
Camera.mla2sensor = mla2sensor; %Camera.fm; % array to sensor distance

% light characteritics
Camera.n = n; % refraction index (1 for air)
Camera.WaveLength = waveLength*10^-3; % wavelenght of the the emission light
Camera.k = 2*pi*Camera.n/Camera.WaveLength; %% wave number
Camera.k0 = 2*pi*1/Camera.WaveLength; %% k for air

objRad = Camera.fobj * NA; % objective radius
% uRad = lensPitch/2; % microimage radius

if (plenoptic == 2)
    % check if a tube2mla is provided by user, and compute 2.0 with
    
    delta_ot = Camera.fobj + Camera.ftl; %% objective to tl distance
    if tube2mla == 0
        tube2mla = computeTube2MLA(lensPitch, mla2sensor, delta_ot, objRad, ftl); %mla2sensor * 2*objRad/lensPitch; % - mla2sensor; %% Favaro
    end
    dot = Camera.ftl * tube2mla/(tube2mla-Camera.ftl); %% depth focused on the mla by the tube lens (object side of tl)
    dio = delta_ot - dot; %% image side of the objective
    dof = Camera.fobj * dio/(dio - Camera.fobj); %% object side of the objective -> dof is focused on the ml
    if isnan(dof) 
        dof = Camera.fobj;
    end
%     M_mla = tube2mla/dof;
    M_mla = M; %% in 4f systems magnification does not change with depth ?
else
    tube2mla = Camera.ftl;
    dof = Camera.fobj; %% object side of the objective -> dof is focused on the mla
    M_mla = M; % magnification up to the mla position
end
tubeRad = tube2mla*lensPitch/mla2sensor/2;
uRad =  tubeRad*mla2sensor/tube2mla; %% Favaro
offsetFobj = dof - Camera.fobj;

Camera.plenoptic = plenoptic;
Camera.objRad = objRad;
Camera.uRad = uRad;
Camera.tube2mla = tube2mla;
Camera.dof = dof;
Camera.offsetFobj = offsetFobj;
Camera.M_mla = M_mla;
Camera.tubeRad = tubeRad;
Camera.usePhaseMask = 0;
Camera.useMLAPhaseMask = 0;

if exist('usePhaseMask')
    if usePhaseMask == 1
        Camera.usePhaseMask = 1;
    end
end

if exist('useMLAPhaseMask')
    if useMLAPhaseMask == 1
        Camera.useMLAPhaseMask = 1;
    end
end

% If phasemask active or hex grid, quarter computation of patterns cannot
% be used.

if Camera.usePhaseMask==1 || Camera.useMLAPhaseMask==1 || strcmp(gridType, 'hex') || strcmp(focus, 'multi')
    Camera.range = 'full';
else
    Camera.range = 'quarter';
end
