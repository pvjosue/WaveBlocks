function NewLensletGridModel = setGridModel(SpacingPx, FirstPosShiftRow, UMax, VMax, HOffset, VOffset, Rot, Orientation, GridType)

% todo: defaults
if(strcmp(GridType, 'hex'))
    Spacing = [SpacingPx*cosd(30), SpacingPx];
    Spacing = ceil(Spacing);
    Spacing = ceil(Spacing/2)*2;
end

if(strcmp(GridType, 'reg'))
    Spacing = [SpacingPx, SpacingPx];
end

NewLensletGridModel.HSpacing = Spacing(2);
NewLensletGridModel.VSpacing = Spacing(1);

NewLensletGridModel.HOffset = HOffset;
NewLensletGridModel.VOffset = VOffset;
NewLensletGridModel.Rot = Rot;
NewLensletGridModel.UMax = UMax;
NewLensletGridModel.VMax = VMax;
NewLensletGridModel.Orientation = Orientation;
NewLensletGridModel.FirstPosShiftRow = FirstPosShiftRow;