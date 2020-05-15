function [GridCoords] = LFBuildGrid( LensletGridModel, grid_type )
% Adapted after: (to accomodate for regular grid arrays)
% LF Toolbox v0.4 released 12-Feb-2015
% Copyright (c) 2013-2015 Donald G. Dansereau

RotCent = eye(3);
RotCent(1:2,3) = [LensletGridModel.HOffset, LensletGridModel.VOffset];

ToOffset = eye(3);
ToOffset(1:2,3) = [LensletGridModel.HOffset, LensletGridModel.VOffset];

R = ToOffset * RotCent * LFRotz(LensletGridModel.Rot) * RotCent^-1;


[vv,uu] = ndgrid((0:LensletGridModel.VMax-1).*LensletGridModel.VSpacing, (0:LensletGridModel.UMax-1).*LensletGridModel.HSpacing);

if(strcmp(grid_type, 'hex'))
    uu(LensletGridModel.FirstPosShiftRow:2:end,:) = uu(LensletGridModel.FirstPosShiftRow:2:end,:) + 0.5.*LensletGridModel.HSpacing;
end

%% TODO: double check this out 
if(strcmp(grid_type, 'reg'))
%     uu(LensletGridModel.FirstPosShiftRow:1:end,:) = uu(LensletGridModel.FirstPosShiftRow:1:end,:) + 0.5.*LensletGridModel.HSpacing;
end

GridCoords = [uu(:), vv(:), ones(numel(vv),1)];
GridCoords = (R*GridCoords')';

GridCoords = reshape(GridCoords(:,1:2), [LensletGridModel.VMax,LensletGridModel.UMax,2]);

end
