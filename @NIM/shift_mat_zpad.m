function Xshifted = shift_mat_zpad( X, shift, dim )
%
% Xshifted = shift_mat_zpad( X, shift, <dim> )
%
% Takes a vector or matrix and shifts it along dimension dim by amount
% shift using zero-padding. Positive shifts move the matrix right or down

% Default to appropriate dimension if X is one-dimensional
if nargin < 3
	[a b] = size(X);
	if a == 1
		dim = 2;
	else
		dim = 1;
	end
end

sz = size(X);
if dim == 1
	if shift >= 0
		Xshifted = [zeros(shift,sz(2)); X(1:end-shift,:)];
	else
		Xshifted = [X(-shift+1:end,:); zeros(-shift,sz(2))];
	end
elseif dim == 2
	if shift >= 0
		Xshifted = [zeros(sz(1),shift) X(:,1:end-shift)];
	else
		Xshifted = [X(:,-shift+1:end) zeros(sz(1),-shift)];
	end
end

