function [V, spacing] = readedf(path)
% function [V] = readedf(path)
% function [V, spacing] = readedf(path)
%
% Reads the file at the given path into a data volume. The file is expected
% to be in ESRF Data Format (EDF). All non-complex primitive types are
% supported. Compression is not available.
%
% Arguments:
% path          Source file.
% V             Parsed volume.
% spacing       Parsed spacing, empty if not found in header.
%
% History:
% 2013-11-01    Initial version [jakob.vogel@cs.tum.edu]
% 2013-12-04    Added (non-standard) spacing [jakob.vogel@cs.tum.edu]

	fid = fopen(path, 'r');
	if fid < 0
		error(['failed opening "' path '"']);
	end

	% read a single character
	if fread(fid, 1) ~= '{'
		fclose(fid);
		error('failed reading opening header marker');
	end

	byteorder = 'n';
	datatype = '';
	datatype_bytes = 0;
	size = 0;
	dim = [];
	spacing = [];

	% read name/value pairs
	while ~feof(fid)
		% skip white-space characters
		while ~feof(fid)
			pos = ftell(fid);
			chr = fread(fid, 1);
			if chr ~= 10 && chr ~= 13 && chr ~= 32 && chr ~= 9 && sum(isspace(chr)) == 0
				fseek(fid, pos, 'bof');
				break
			end
		end

		% abort if the header is closed
		pos = ftell(fid);
		chr = fread(fid, 1);
		if feof(fid) || chr == '}'
			break
		end
		fseek(fid, pos, 'bof');

		% extract the property assignment
		quotesSingle = false; quotesDouble = false;
		assignment = '';
		while ~feof(fid)
%			pos = ftell(fid);
			chr = fread(fid, 1);

			if chr == ';' && ~(quotesSingle || quotesDouble)
				break
			end
			
			% check for quote characters
			if chr == ''''
				quotesSingle = ~quotesSingle;
			end
			if chr == '"'
				quotesDouble = ~quotesDouble;
			end
			
			assignment = [assignment chr]; %#ok<AGROW>
		end

		% split the assignment, and remove quotes (if they exist)
		delim = strfind(assignment, '=');
		name = strtrim(assignment(1:(delim-1)));
		value = strtrim(assignment((delim+1):end));
		if value(1) == value(end) && (value(1) == '''' || value(1) == '"')
			value = value(2:(end-1));
		end

		if length(name) >= 5 && strcmpi(name(1:4), 'Dim_')
			component = str2double(name(5:end));
			dim(component) = str2double(value); %#ok<AGROW>
		elseif strcmpi(name, 'Spacing')
			spacing = str2num(value); %#ok<ST2NM>
		elseif strcmpi(name, 'DataType')
			if strcmpi(value, 'SignedByte')
				datatype = 'int8';
				datatype_bytes = 1;
			elseif strcmpi(value, 'UnsignedByte')
				datatype = 'uint8';
				datatype_bytes = 1;
			elseif strcmpi(value, 'SignedShort')
				datatype = 'int16';
				datatype_bytes = 2;
			elseif strcmpi(value, 'UnsignedShort')
				datatype = 'uint16';
				datatype_bytes = 2;
			elseif strcmpi(value, 'SignedInteger')
				datatype = 'int32';
				datatype_bytes = 4;
			elseif strcmpi(value, 'UnsignedInteger')
				datatype = 'uint32';
				datatype_bytes = 4;
			elseif strcmpi(value, 'SignedLong')
				datatype = 'int64';
				datatype_bytes = 8;
			elseif strcmpi(value, 'UnsignedLong')
				datatype = 'uint64';
				datatype_bytes = 8;
			elseif strcmpi(value, 'Float') || strcmpi(value, 'Real') || strcmpi(value, 'FloatValue')
				datatype = 'single';
				datatype_bytes = 4;
			elseif strcmpi(value, 'Double') || strcmpi(value, 'DoubleValue')
				datatype = 'double';
				datatype_bytes = 8;
			else
				fclose(fid);
				error(['unsupported data type: ' value]);
			end
		elseif strcmpi(name, 'Compression')
			fclose(fid);
			error('compression not supported');
		elseif strcmpi(name, 'Size')
			size = str2double(value);
		elseif strcmpi(name, 'Image')
			img = str2double(value);
			if img ~= 1
				fclose(fid);
				error('image not set to 1');
			end
		elseif strcmpi(name, 'ByteOrder')
			if strcmpi(value, 'LowByteFirst')
				byteorder = 'l';
			elseif strcmpi(value, 'HighByteFirst')
				byteorder = 'b';
			else
				fclose(fid);
				error(['unknown byte order: ' value]);
			end
		end

		% extract further chars until the end of the line is reached
		while ~feof(fid)
			chr = uint8(fread(fid, 1));
			% abort on line feed or carriage return
			if chr == 10 || chr == 13
				break
			end
		end
	end
	
	if feof(fid) || fread(fid, 1) ~= 10
		fclose(fid);
		error('failed reading closing header marker');
	end

	% store the position, and close the file
	pos = ftell(fid);
	fclose(fid);

	% check the size
	dim_serialized = prod(dim);
	if dim_serialized * datatype_bytes ~= size
		error('size inconsistency');
	end
	
	% re-open the file with the correct byte order (from original
	% physicist's code)
    fid = fopen(path, 'r', byteorder);
    fseek(fid, pos, 'bof');

	[V, count] = fread(fid, dim_serialized, [datatype '=>' datatype]);
	if count ~= dim_serialized
		fclose(fid);
		error('data incomplete');
	end
	if length(dim) > 1
		V = reshape(V, dim);
	end
	fclose(fid);
end
