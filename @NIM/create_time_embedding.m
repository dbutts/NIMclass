function Xmat = create_time_embedding(stim,params)
%
% Xmat = create_time_embedding(stim,params)
%
% Takes a Txd stimulus matrix and creates a time-embedded matrix of size Tx(d*L),
% where L is the desired number of time lags. If stim is a 3d array 
% the 'spatial dimensions are folded into the 2nd dimension. Assumes zeros-padding.
% Optional up-sampling of stimulus and tent-basis representation for filter
% estimation. Note that Xmatrix is formatted so that adjacent time lags 
% are adjacent within a time-slice of the Xmatrix. Thus X(t,1:nLags) gives
% all time lags of the first spatial pixel at time t.
%
% INPUTS:
%       stim: stimulus matrix (time must be in the first dim.)
%       params: struct of stimulus params (see NIM_create_stim_params)
% OUTPUTS: 
%       Xmat: Time embedded stim matrix

%%
sz = size(stim);

%if there are two spatial dims, fold them into one
if length(sz) > 2
    stim = reshape(stim,sz(1),prod(sz(2:end)));
end
%no support for more than two spatial dims
if length(sz) > 3
    disp('More than two spatial dimensions not supported, but creating Xmatrix anyways...');
end

%check that the size of stim matches with the specified stim_params
%structure
[NT,Npix] = size(stim);
if prod(params.dims(2:end)) ~= Npix
    error('Stimulus dimension mismatch');
end

%up-sample stimulus if required
if params.up_fac > 1
    stim = stim(floor((0:(NT*params.up_fac)-1)/params.up_fac)+1,:);
    NT = size(stim,1); %new data dur
end

%if using a tent-basis representation
if ~isempty(params.tent_spacing)
    tbspace = params.tent_spacing;
    %create a tent-basis (triangle) filter
    tent_filter = [(1:tbspace)/tbspace 1-(1:tbspace-1)/tbspace]/tbspace;
    
    %apply to the stimulus
    filtered_stim = zeros(size(stim));
    for i = 1:length(tent_filter)
        filtered_stim = filtered_stim + shift_mat_zpad(stim,i-tbspace,1)*tent_filter(i);
    end
    
    stim = filtered_stim; 
    lag_spacing = tbspace;
else
    lag_spacing = 1;
end

%for temporal only stimuli (this method can be faster if you're not using
%tent-basis rep
if Npix == 1 && isempty(params.tent_spacing)
    Xmat = toeplitz(stim,[stim(1) zeros(1,params.dims(1) - 1)]);
else
    %otherwise loop over lags and manually shift the stim matrix
    Xmat = zeros( NT, prod(params.dims));
    for n = 1:params.dims(1)
        Xmat(:,n-1+(1:params.dims(1):(Npix*params.dims(1)))) = shift_mat_zpad( stim, lag_spacing*(n-1), 1);
    end    
end
