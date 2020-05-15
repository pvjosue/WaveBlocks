function writeedf(V, arg1, arg2)
% function writeedf(V, path)
% function writeedf(V, spacing, path)
%
% Writes the given data volume to a file at the given path. The data is
% written in ESRF Data Format (EDF) with a padded header (KB-boundary) to
% be compatible with legacy implementations. Compression is not available.
% All non-complex primitive types are supported.
%
% Arguments:
% V             Volume to write.
% spacing       (Optional) spacing.
% path          Destination file.
%
% History:
% 2013-11-01    Initial version [jakob.vogel@cs.tum.edu]
% 2013-12-04    Added (non-standard) spacing [jakob.vogel@cs.tum.edu]

	if nargin < 2
		error('at least two arguments required');
	elseif nargin < 3
		spacing = [];
		path = arg1;
	else
		% make sure that 'spacing' is a row vector
		spacing = arg1(:)';
		path = arg2;
	end

	% check that non-empty spacing matches to volume dimension
	if ~isempty(spacing) && numel(spacing) ~= ndims(V)
		error('spacing vector does not match to volume dimension');
	end

	switch class(V)
		case 'int8'
			datatype = 'SignedByte';
			datatype_bytes = 1;
		case 'uint8'
			datatype = 'UnsignedByte';
			datatype_bytes = 1;
		case 'int16'
			datatype = 'SignedShort';
			datatype_bytes = 2;
		case 'uint16'
			datatype = 'UnsignedShort';
			datatype_bytes = 2;
		case 'int32'
			datatype = 'SignedInteger';
			datatype_bytes = 4;
		case 'uint32'
			datatype = 'UnsignedInteger';
			datatype_bytes = 4;
		case 'int64'
			datatype = 'SignedLong';
			datatype_bytes = 8;
		case 'uint64'
			datatype = 'UnsignedLong';
			datatype_bytes = 8;
		case 'single'
			datatype = 'FloatValue';
			datatype_bytes = 4;
		case 'double'
			datatype = 'DoubleValue';
			datatype_bytes = 8;
		otherwise
			error(['can not write ' class(V)]);
	end
	
	% open header
	header = sprintf('{\n');

	% write id and type information
	header = [header sprintf('HeaderID = %s;\n', 'EH:000001:000000:000000')];
	header = [header sprintf('Image = %d;\n', 1)];
	header = [header sprintf('ByteOrder = %s;\n', 'LowByteFirst')];
	header = [header sprintf('DataType = %s;\n', datatype)];

	% write dimension and size
	for i=1:ndims(V)
		header = [header sprintf('Dim_%d = %d;\n', i, size(V,i))]; %#ok<AGROW>
	end
	header = [header sprintf('Size = %d;\n', datatype_bytes * numel(V))];

	% write date
	header = [header sprintf('Date = %s;\n', date)];

	% write spacing
	if ~isempty(spacing)
		header = [header sprintf('Spacing = %s;\n', num2str(spacing, ' %g'))];
	end

	% pad the header by adding spaces such that the header ends on a
	% kilobyte boundary
	n = 1024;
	while n < (length(header) + 3)
		n = n + 1024;
	end
	n = n - length(header) - 3;
	header((end+1):(end+n)) = ' ';

	% close header
	header = [header sprintf('\n}\n')];

	% open the file, and write everything
	fid = fopen(path, 'w', 'l');
	fwrite(fid, header);
	fwrite(fid, V(:), class(V));
	fclose(fid);
end
