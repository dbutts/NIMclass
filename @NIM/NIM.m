classdef NIM
    
% Class implementation of 'nonlinear-input model' (NIM). 
%     Models consist of a set of linear-nonlinear processing
%     'subunits', which act on different sets of predictors
%     ('stimuli'). The summed output of these subunits is then
%     transformed by a 'spiking nonlinearity' function to generate a
%     predicted firing rate. Model parameters are optimized based on
%     the penalized Log-likelihood. Several different noise models can
%     be used.
%   
% Reference: 
%   McFarland JM, Cui Y, Butts DA (2013) Inferring nonlinear neuronal computation based
%        on physiologically plausible inputs. PLoS Computational Biology 9(7): e1003142
% 
% Original: Created by James McFarland, September 2015
% Modified: By NeuroTheory Lab at University of Maryland (Jan 2016-) 

%% PROPERTIES
properties
	spkNL;          % struct defining the spiking NL function
	subunits;       % array of subunit objects
	stim_params;    % struct array of parameters characterizing the stimuli that the model acts on, must have a .dims field
	noise_dist;     % noise distribution class specifying the noise model
	spk_hist;       % class defining the spike-history filter properties
	fit_props;      % struct containing information about model fit evaluations
	fit_history;    % struct containing info about history of fitting
end

properties (Hidden)
	init_props;         % struct containing details about model initialization
	allowed_reg_types = {'nld2','d2xt','d2x','d2t','l2','l1'}; % set of allowed regularization types
	allowed_spkNLs = {'lin','rectpow','exp','softplus','logistic'}; % set of NL functions currently implemented
	allowed_noise_dists = {'poisson','bernoulli','gaussian'}; % allowed noise distributions
	version = '1.0';    % source code version used to generate the model
	create_on = date;   % date model was generated
	min_pred_rate = 1e-50; % minimum predicted rate (for non-negative data) to avoid NAN LL values
	opt_check_FO = 1e-2;   % threshold on first-order optimality for fit-checking
end
    
%% METHODS DEFINED IN SEPARATE FILES
methods
	nim = fit_filters( nim, Robs, Xstims, varargin );             % filter model stim-filters 
	%[bnims,LLs] = fit_filters_boost( nim, Robs, Xstims, varargin );             % filter model stim-filters 
	%nim = fit_filters_boost_sub( nim, Robs, Xstims, varargin );             % filter model stim-filters 
	nim = fit_upstreamNLs( nim, Robs, Xstims, varargin );         % fit model upstream NLs
	nim = fit_spkNL( nim, Robs, Xstims, varargin );               % fit parameters of spkNL function
	nim = fit_NLparams( nim, Robs, Xstims, varargin );            % fit parameters of (parametric) upstream NL functions
	nim = fit_weights( nim, Robs, Xstims, varargin );             % fit linear weights on each subunit
	nim = reg_path( nim, Robs, Xs, Uindx, XVindx, varargin );     % determine optimal regularization using cross-val data
	nim = reg_path2( nim, Robs, Xs, Uindx, XVindx, varargin );     % determine optimal regularization using cross-val data
	fig_handle = display_model( nim, Robs, Xstims, varargin );    % display current model
	[] = display_model_dab( nim, Robs, Xstims, varargin );        % display current model (originally modified version)
	[] = display_model_jmm( nim, Robs, Xstims, varargin );        % display current model (original version)
end
methods (Static)
	Xmat = create_time_embedding( stim, params ) % make time-embedded stimulus
	spikebins = convert_to_spikebins( binned_spikes ) % make vector of spk indices from binned spike histogram
	Xshifted = shift_mat_zpad( X, shift, dim ) % shift matrix along given dimension
end
methods (Static, Hidden)
	Tmat = create_Tikhonov_matrix( stim_params, reg_type ); % make regularization matrices
end

%% ADDITIONAL METHODS DEFINED HERE in NIM.m
% Constructor:
%   function nim = NIM( stim_params, NLtypes, mod_signs, varargin ) 
% Setting Methods:
%   function nim = set_reg_params( nim, varargin )
%		function nim = set_stim_params( nim, varargin )
% Getting Methods:
%   function lambdas = get_reg_lambdas( nim, varargin )
%   function filtKs = get_filtKs( nim, sub_inds )
%   function NLtypes = get_NLtypes( nim, sub_inds )
%   function [filt_penalties,NL_penalties] = get_reg_pen( nim, Tmats )
% Helper Methods:
%   function nim = add_subunits( nim, NLtypes, mod_signs, varargin )
%   function nim = init_spkhist( nim, n_bins, varargin )
%   function nim = init_nonpar_NLs( nim, Xstims, varargin )
%   function [LL, pred_rate, mod_internals, LL_data] = eval_model( nim, Robs, Xstims, varargin )
%   function [LLs,LLnulls] = eval_model_reps( nim, RobsR, Xstims, varargin )
%   function [filt_SE,hessMat] = compute_filter_SEs( nim, Robs, Xstims, varargin )
% Display Methods:
%   function [] = display_spkNL( nim, G, varargin )
%   function [] = display_spike_history( nim, varargin )
% Hidden Methods:
%   function [G, fgint, gint] = process_stimulus( nim, Xstims, sub_inds, gain_funs )
%   function [LL,norm_fact] = internal_LL( nim, rPred, Robs )
%   function LL_deriv = internal_LL_deriv( nim, rPred, Robs )
%   function rate = apply_spkNL( nim, gen_signal )
%   function rate_deriv = apply_spkNL_deriv( nim, gen_signal, thresholded_inds )
%   function rate_grad = spkNL_param_grad( nim, params, x )
%   function Tmats = make_Tikhonov_matrices( nim )
%   function Xspkhst = create_spkhist_Xmat( nim, Robs )
%   function Tmat = make_NL_Tmat( nim )
%   function [] = check_inputs( nim, Robs, Xstims, sub_inds, gain_funs )
%   function nim = set_subunit_scales( nim, fgint )
% Static/Hidden:
%   function [train_inds,parsed_options,modvarargin] = parse_varargin( cellarray2parse, fields_to_remove, default_options  )
%   function arevalid = validate_parsed_options( parsed_struct, valid_list, return_error )
%   function percentiles = my_prctile( x, p )


%% ********************** Constructor *************************************
methods
    
	function nim = NIM( stim_params, NLtypes, mod_signs, varargin )
	% Usage: nim = NIM( stim_params, NLtypes, mod_signs, varargin ) 
	% constructor for class NIM 
	% INPUTS:
	%   stim_params: struct array defining parameters for each stimulus the model acts on.
	%                Must specify the .dims field for each stim 
	%   NLtypes: string or cell array of strings specifying the upstream NL type associated
	%            with each subunit. If it's a single string, we use the same NL type throughout.
	%            Allowed NLtypes: {'lin','quad','rectlin','rectpow','softplus','exp','nonpar'} 
	%   mod_signs: vector specifying the weight associated with each subunit (typically +/-1) 
	%   optional_flags:
	%   -> general subunit information
	%      ('NLparams',NLparams): vector of parameter values (or cell array) for the corresponding NL functions
	%      ('NLoffsets',NLoffsets): vector of initial NL offset terms (or single value)
	%      ('init_filts',init_filts): cell array of initial subunit filter values 
	%      ('Ksign_cons',Ksign_cons): vector specifying any constraints on the filter
	%         coefs of each subunit. [-1 for neg; +1 for pos; nan for no cons]
	%      ('Xtargets',Xtargs): vector specifying the index of the stimulus each subunit acts on (defaults to ones) 
	%   -> spiking nonlinearity and noise model]
	%      ('spkNL',spkNL): string specifying type of spkNL function: 'lin','rectpow','exp','softplus','logistic'
	%      ('noise_dist',noise_dist): string specifying type of noise distribution: 'poisson','bernoulli','gaussian'
	%   -> regularization
	%      (lambda_type,lambda_vals): specify type of regularization, and then a vector of values for each 
	%          subunit (or a scalar which is assumed the same for all units. Regularization types 
	%          currently supported are: l1, l2, d2xt, d2x, d2t, and nld2 (for non-parametric nonlinearity)
	%
	% OUTPUTS:
	%   nim: initialized model object

		if nargin == 0
			return % handle the no-input-argument case by returning a null model. This is important when initializing arrays of objects  
		end

		nStims = length(stim_params); % number of stimuli
		nim.stim_params = stim_params;

		nSubs = length(mod_signs); % number of subunits
		% If 'NLtypes' is specified as a single string, default to using this NLtype for all subunits
		if ~iscell(NLtypes) && ischar(NLtypes); NLtypes = cellstr(NLtypes); end;
		if length(NLtypes) == 1 && nSubs > 1; NLtypes = repmat(NLtypes,nSubs,1); end

		% Set defaults
		Xtargets = ones(nSubs,1);
		spkNL = 'softplus';
		noise_dist = 'poisson';
		init_filts = cell(nSubs,1);
		Ksign_cons = zeros(nSubs,1); %default no constraints on filters
		NLoffsets = zeros(nSubs,1); %default NL offsets to 0
		NLparams = cell(nSubs,1);

		% Parse input flags
		assert( mod(nargin-3,2) == 0, 'input format for optional flags must be in pairs: ''argname'',''argval' );
		j = 1; % initialize counter after required input args
		reg_types = {}; reg_vals = [];
		while j <= length(varargin)
			switch lower(varargin{j})
				case 'xtargets'
					Xtargets = varargin{j+1};
					assert(all(ismember(Xtargets,1:nStims)),'invalid Xtargets specified');
					if length(Xtargets) == 1 % if only one is specified, assume all subunits get the same
						Xtargets = repmat(Xtargets,1,nSubs);
					end
				case 'spknl'
					spkNL = lower(varargin{j+1});
					assert(ischar(spkNL),'spkNL must be a string');
				case 'noise_dist'
					noise_dist = lower(varargin{j+1});
					assert(ischar(noise_dist),'noise_dist must be a string');
				case 'init_filts'
					init_filts = varargin{j+1};
					assert(iscell(init_filts),'init_filts must be a cell array');
				case 'ksign_cons',
					Ksign_cons = varargin{j+1};
					assert(all(ismember(Ksign_cons,[-1 1 0])),'Ksign_cons must have values -1,0, or 1');
					if length(Ksign_cons) == 1 %if only one specified, assume all subunits get the same
						Ksign_cons = repmat(Ksign_cons,1,nSubs);
					end
				case 'nlparams'
					cur_NLparams = varargin{j+1};
					if ~iscell(cur_NLparams)
						[NLparams{1:nSubs}] = deal(cur_NLparams); %if specified as vector, assume all subunits get the same
					else
						NLparams = cur_NLparams;
					end
				case 'nloffsets'
					NLoffsets = varargin{j+1};
					if length(NLoffsets) == 1
						NLoffsets = repmat(NLoffsets,1,nSubs); % assume all filters get same offset if only one specified 
					end
					assert(length(NLoffsets) == nSubs,'NLoffsets must be vector of lenght nSubs');
				case nim.allowed_reg_types % if the flag is an allowed regularization type
					reg_types = cat(1,reg_types,lower(varargin{j}));
					cur_vals = varargin{j+1};
					reg_vals = cat(2,reg_vals, cur_vals(:)); % build up a [KxP] matrix, where K is the number of subunits and P is the number of reg types
				otherwise
					error('Invalid input flag'); 
			end
			j = j + 2;
		end
		if size(reg_vals,1) == 1 % if reg_vals are specified as scalars, assume we want the same for all subuntis
			reg_vals = repmat(reg_vals,nSubs,1);
		end
		if ~isempty(reg_vals); assert(size(reg_vals,1) == nSubs,'must specify a vector of regularization lambdas for each subunit or a scalar'); end;

		% Only use logistic spkNL for a bernoulli noise model
		if strcmp(noise_dist,'bernoulli') && ~strcmp('spkNL','logistic')
			spkNL = 'logistic';
			fprintf('Forcing spkNL to logistic for Bernoulli likelihood\n');  
		end
		
		% Check and create spk NL function
		assert(ismember(spkNL,nim.allowed_spkNLs),'not an allowed spk NL type');
		nim.spkNL.type = spkNL;
		nim.spkNL.theta = 0; % initialize offset term
    
		% Set default parameters for other spkNL parameters depending on the type
		switch nim.spkNL.type
			case 'lin'
				nim.spkNL.params = [1]; %defines [beta] in f(x) = beta*x
			case 'rectpow'
				nim.spkNL.params = [1 2]; %defines [beta, gamma] in f(x) = (beta*x)^gamma
			case 'exp'
				nim.spkNL.params = [1]; %defines [beta] in f(x; beta) = exp(beta*x)
			case 'softplus'
				nim.spkNL.params = [1 1]; %defines [beta, alpha] in f(x) = alpha*log(1+exp(beta*x))
			case 'logistic'
				nim.spkNL.params = [1]; %defines [beta] in f(x) = 1/(1+exp(-beta*x))    
		end
		
		% Check and create noise distribution
		assert(ismember(noise_dist,nim.allowed_noise_dists),'not an allowed noise distribution');
		nim.noise_dist = noise_dist;
		assert(length(Xtargets) == nSubs,'length of mod_signs and Xtargets must be equal');

		% Initialize subunits
		nim.init_props = rng(); %save state of RNG used for initializing filter weights
		for ii = 1:nSubs %loop initializing subunits (start from last to initialize object array)
			stimD = prod(nim.stim_params(Xtargets(ii)).dims); %dimensionality of the current filter
			if isempty(init_filts{ii})
				init_filt = randn(stimD,1)/stimD; % initialize fitler coefs with gaussian noise
			else
				init_filt = init_filts{ii};
			end
			nim.subunits = cat(1,nim.subunits,SUBUNIT(init_filt, mod_signs(ii), NLtypes{ii},Xtargets(ii),...
		NLoffsets(ii),NLparams{ii},Ksign_cons(ii)));
		end
		
		% If initial lambdas are specified       
		for ii = 1:length(reg_types)
			assert(all(reg_vals(:,ii)) >= 0,'regularization hyperparameters must be non-negative');
			for jj = 1:nSubs
				nim.subunits(jj).reg_lambdas.(reg_types{ii}) = reg_vals(jj,ii);
			end	
		end
		
		% Initialize WITHOUT spike history term
		nim.spk_hist.coefs = [];
		nim.spk_hist.bin_edges = [];
		nim.spk_hist.spkhstlen = 0;
	end

end

%% ********************** Setting Methods *********************************
methods
    
	function nim = set_reg_params( nim, varargin )
	% Usage: nim = nim.set_reg_params( varargin )
	% Sets a desired set of regularization parameters to specified values, apply to specified set of subunits
	%
	% INPUTS:
	%   optional flags:
	%       ('subs',sub_inds): set of subunits to apply the new reg_params for (default = ALL)
	%       ('lambda_type',lambda_val): first input is a string specifying the type of regularization,
	%            e.g. 'd2t' for temporal smoothness. This must be followed by a scalar or vector of
	%            length (Nsubs) giving the associated lambda values
	%
	% OUTPUTS:
	%   nim: NIM model object

		sub_inds = 1:length(nim.subunits); %default is to apply the change to all subunits

		% INPUT PARSING
		j = 1;
		reg_types = {}; reg_vals = [];
		while j <= length(varargin)
			switch lower(varargin{j})  
				case 'subs'
					sub_inds =  varargin{j+1};
					assert(all(ismember(sub_inds,1:length(nim.subunits))),'invalid target subunits specified');
				case nim.allowed_reg_types
					reg_types = cat(1,reg_types,lower(varargin{j}));
					cur_vals = varargin{j+1};
					reg_vals = cat(2,reg_vals, cur_vals(:)); % build up a [KxP] matrix, where K is the number of subunits and P is the number of reg types
				otherwise
					error('Invalid input flag');
			end
			j = j + 2;
		end
		
		if size(reg_vals,1) == 1 %if reg_vals are specified as scalars, assume we want the same for all subuntis
			reg_vals = repmat(reg_vals,length(sub_inds),1);
		end
		
		if isempty(reg_vals)
			warning('No regularization values specified, no action taken');
		end
		
		for ii = 1:length(reg_types)
			assert(all(reg_vals(:,ii) >= 0),'regularization hyperparameters must be non-negative');
			for jj = 1:length(sub_inds)
				nim.subunits(sub_inds(jj)).reg_lambdas.(reg_types{ii}) = reg_vals(jj,ii);
			end
		end	
	end

	function nim = set_stim_params( nim, varargin )
	% Usage: nim = nim.set_stim_params( varargin )
	% Sets values of the stim_params struct for a desired 'Xtarget'
	%
	% INPUTS:
	%   optional flags:
	%			'xtarg': index of stimulus to apply the new stim_params for [default is 1]
	%     'dims': dimensionality of stim: [Tdim, X1dim, X2dim] where Tdim is the number of temporal dimensions, etc.
	%     'boundary_conds': boundary conditions on each stim dimension. Inf means free boundaries, 0 means tied 
	%                       to 0, and -1 means periodic.
	%     'split_pts': specify an 'internal boundary' over which you dont want to smooth. Must be a vector of the form:
	%                  [direction split_ind boundary_cond], where direction is the index of the dimension of the 
	%                  stimulus you want to create a boundary (i.e. 1 for the temporal dimension), split_ind is the 
	%                  index value along that dimension after which you want to create the boundary, and boundary_cond
	%                  specifies the boundary conditions you want to use.
	%
	% OUTPUTS: 
	%   nim: updated NIM object

		Xtarg = 1; %default is to apply the change to stim 1
		allowed_flags = {'dims','boundary_conds','split_pts'}; %fields of the stim_params struct that we might want to set

		% INPUT PARSING
		j = 1; fields_to_set = {}; field_vals = {};
		while j <= length(varargin)
			switch lower(varargin{j})
				case 'xtarg'
					Xtarg = varargin{j+1};
				case allowed_flags
					fields_to_set = cat(1,fields_to_set,varargin{j});
					field_vals = cat(1,field_vals,varargin{j+1});
				otherwise
					error('Invalid input flag'); 
			end
			j = j + 2;  
		end
		
		if isempty(field_vals)
			warning('No stim_params values specified to change');
		end
		
		if Xtarg > length(nim.stim_params) % if we're creating a new stimulus
			dim_locs = find(ismember('dims',fields_to_set));
			assert(~isempty(dim_locs),'need to specify dims to initialize new stimulus');
			new_stim_params.dims = field_vals{dim_locs};
			new_stim_params = NIM.create_stim_params(new_stim_params.dims); % initialize default params for new stimulus
			nim.stim_params(Xtarg) = new_stim_params;
		end
		for ii = 1:length(fields_to_set) % assign new field values
			nim.stim_params(Xtarg).(fields_to_set{ii}) = field_vals{ii};
		end
		while length(nim.stim_params(Xtarg).dims) < 3 % pad dims with 1s for book-keeping
			nim.stim_params(Xtarg).dims = cat(2,nim.stim_params(Xtarg).dims,1);
		end	
	end

end

%% ********************** Getting Methods *********************************
methods
    
	function lambdas = get_reg_lambdas( nim, varargin )
	% Usage: lambdas = nim.get_reg_lambdas( varargin )
	% Gets regularization lambda values of specified type from a set of nim subunits
	%
	% INPUTS:
	%   optional flags:
	%      'subs': vector specifying which subunits to extract lambda values from
	%      'lambda_type': string specifying the regularization type
	% OUTPUTS:
	%   lambdas: [K,N] matrix of lambda values, K is the number of specified lambda_types and N is the number of subunits

		sub_inds = 1:length(nim.subunits); %default is to grab reg values from all subunits
		
		% INPUT PARSING    
		jj = 1;
		reg_types = {};
		while jj <= length(varargin)
			switch lower(varargin{jj})
				case 'subs'
					sub_inds = varargin{jj+1};
					assert(all(ismember(sub_inds,1:length(nim.subunits))),'invalid target subunits specified');
					jj = jj + 2;
				case nim.allowed_reg_types
					reg_types = cat(1,reg_types,lower(varargin{jj}));
					jj = jj + 1;
				otherwise
					error('Invalid input flag'); 
			end		
		end
		
		lambdas = nan(length(reg_types),length(sub_inds));
		if isempty(reg_types)
			warning( 'No regularization type specified, returning nothing' );
		end
		for ii = 1:length(reg_types)
			for jj = 1:length(sub_inds)
				lambdas(ii,jj) = nim.subunits(sub_inds(jj)).reg_lambdas.(reg_types{ii});  
			end
		end	
	end

	function filtKs = get_filtKs( nim, sub_inds )
	% Usage: filtKs = nim.get_filtKs( sub_inds )
	% Gets filters for specified set of subunits 
	%
	% INPUTS:
	% 	<sub_inds>: vector specifying which subunits to get filters from (default is all subs)
	% OUTPUTS:
	%   filtKs: Cell array of filter coefs
      
		Nsubs = length(nim.subunits);
		if nargin < 2
			sub_inds = 1:Nsubs; %default is to grab filters for all subunits
		end
		filtKs = cell(length(sub_inds),1);
		for ii = 1:length(sub_inds)
			filtKs{ii} = nim.subunits(sub_inds(ii)).get_filtK;
		end	
	end

	function NLtypes = get_NLtypes( nim, sub_inds )
	% Usage: NLtypes = nim.get_NLtypes( <sub_inds> )
	%  Gets cell array of strings specifying NLtype of each subunit
	%
	%  INPUTS: 
	%    <sub_inds>: set of subunits to get NL types from (default is all)
	%  OUTPUTS: 
	%    NLtypes: cell array of NLtypes

		Nsubs = length(nim.subunits);
		if nargin < 2
			sub_inds = 1:Nsubs; %default is to grab filters for all subunits
		end
		NLtypes = cell(length(sub_inds),1);
		for ii = 1:length(sub_inds)
			NLtypes{ii} = nim.subunits(sub_inds(ii)).NLtype;
		end	
	end

	function [filt_penalties,NL_penalties] = get_reg_pen( nim, Tmats )
	% Usage: [filt_penalties,NL_penalties] = nim.get_reg_pen( <Tmats> )
	% Calculates the regularization penalties on each subunit, separately for filter and NL regularization
	%
	% INPUTS: 
	%   <Tmats>: struct array of 'Tikhonov' regularization matrices
	% OUTPUTS: 
	%   filt_penalties: Kx1 vector of regularization penalties for each filter
	%   NL_penalties: Kx1 vector of regularization penalties for each upstream NL

		if nargin < 2 || isempty(Tmats) %if the Tmats are not precomputed and supplied, compute them here        
			Tmats = make_Tikhonov_matrices(nim);
		end
		
		Nsubs = length(nim.subunits);
		Xtargs = [nim.subunits(:).Xtarg];
		filtKs = nim.get_filtKs();
		filt_penalties = zeros(1,Nsubs);
    
		for ii = 1:length(Tmats) % loop over the derivative regularization matrices
			cur_subs = find(Xtargs == Tmats(ii).Xtarg); % set of subunits acting on the stimulus given by this Tmat
			cur_penalties = sum((Tmats(ii).Tmat * cat(2,filtKs{cur_subs})).^2);
			cur_lambdas = nim.get_reg_lambdas(Tmats(ii).type,'subs',cur_subs); % current lambdas
			filt_penalties(cur_subs) = filt_penalties(cur_subs) + cur_penalties.*cur_lambdas; % reg penalties for filters
		end
		
		l2_lambdas = nim.get_reg_lambdas('l2');
		if any(l2_lambdas > 0) % compute L2 penalties on the filter coefficients
			filt_penalties = filt_penalties + l2_lambdas.*cellfun(@(x) sum(x.^2),filtKs)';
		end
		nl_lambdas = nim.get_reg_lambdas('nld2');  % reg lambdas on the NL TB coefficients
		NL_penalties = zeros(1,Nsubs);
		
		if any(nl_lambdas > 0)
			Tmat = nim.make_NL_Tmat();
			nonpar_subs = find(strcmp(nim.get_NLtypes,'nonpar'))';
			for imod = nonpar_subs % compute the reg penalty on each subunit's NL
				NL_penalties(imod) = nl_lambdas(imod)*sum((Tmat*nim.subunits(imod).NLnonpar.TBy').^2);
			end
		end
	end

	function nrms = subunit_filter_norms( nim, Xstims )
	% Usage: nrms = nim.subunit_filter_norms( Xstims )
	% Returns array of subunit filter output magnitudes
	
		if nargin < 2
			Xstims = [];
		end
		
		Nsubs = length(nim.subunits);
		nrms = zeros(1,Nsubs);
		for nn = 1:Nsubs
			nrms(nn) = nim.subunits(nn).filter_norm( Xstims );
		end	
	end
	
end

%% ********************** Helper Methods **********************************
methods
    
	function nim = add_subunits( nim, NLtypes, mod_signs, varargin )
	% Usage: nim = nim.add_subunits( NLtypes, mod_signs, varargin )
	% Adds subunits to the model with specified properties. Can add multiple subunits in one call. 
	% Default is to initialize regularization lambdas to be equal to an existing subunit that acts on 
	% the same Xtarg (and has same NL type). Otherwise, they are initialized to 0.
	%
	% INPUTS: 
	%   NLtypes: string or cell array of strings specifying upstream NL types
	%						 allowed: 'lin','quad','rectlin','rectpow','softplus'
	%   mod_signs: vector of weights associated with each subunit (typically +/- 1)
	%   optional flags:
	%     'init_filts': cell array of initial filter values for each subunit
	%     'lambda_type': first input is a string specifying the type of
	%                    regularization (e.g. 'd2t' for temporal smoothness). This must be followed by a
	%                    scalar/vector giving the associated lambda value(s).
	%     'NLparams': cell array of upstream NL parameter vectors
	%     'xtargs': specify vector of of Xtargets for each added subunit
	% OUTPUTS: nim: updated nim object
     
		if ~iscell(NLtypes) && ischar(NLtypes);
			NLtypes = cellstr(NLtypes); % make NL types a cell array
		end
		nSubs = length(mod_signs); % number of subunits being added
		nStims = length(nim.stim_params);
		Xtargets = ones(nSubs,1); % default Xtargets to 1
		if length(NLtypes) == 1 && nSubs > 1
			NLtypes = repmat(NLtypes,nSubs,1); % if NLtypes is specified as a single string, assume we want this NL for all subunits
		end
		init_filts = cell(nSubs,1);
		NLparams = cell(nSubs,1);

		% Parse input flags
		j = 1; reg_types = {}; reg_vals = [];
		while j <= length(varargin)
			switch lower(varargin{j})
				case 'xtargs'
					Xtargets = varargin{j+1};
					assert(all(ismember(Xtargets,1:nStims)),'invalid Xtargets specified');
				case 'init_filts'
					if ~iscell(varargin{j+1}) % if init_filts are specified as a matrix, make them a cell array
						init_filts = cell(length(mod_signs),1);
						for ii = 1:length(mod_signs)
							init_filts{ii} = varargin{j+1}(:,ii);
						end
					else
						init_filts = varargin{j+1};
					end
				case 'nlparams'
					NLparams = varargin{j+1};
					assert(iscell(NLparams),'NLparams must be input as cell array');
				case nim.allowed_reg_types
					reg_types = cat(1,reg_types,lower(varargin{j}));
					cur_vals = varargin{j+1};
					reg_vals = cat(2,reg_vals, cur_vals(:));
				otherwise
					error('Invalid input flag'); 
			end
			j = j + 2;
		end		
		if size(reg_vals,1) == 1 % if reg_vals are specified as scalars, assume we want the same for all subuntis    
			reg_vals = repmat(reg_vals,nSubs,1);      
		end
		
		assert(length(Xtargets) == nSubs,'length of mod_signs and Xtargets must be equal');
    
		% Initialize subunits
		for ii = 1:nSubs % loop initializing subunits (start from last to initialize object array)
			stimD = prod(nim.stim_params(Xtargets(ii)).dims); % dimensionality of the current filter
			if isempty(init_filts{ii})
				init_filt = randn(stimD,1)/stimD; % initialize fitler coefs with gaussian noise
			else
				init_filt = init_filts{ii};
			end
			
			% Use the regularization parameters from the most similar subunit if we have one, otherwise use default init
			if ~isempty(nim.subunits)
				same_Xtarg = find([nim.subunits(:).Xtarg] == Xtargets(ii),1); % find any existing subunits with this same Xtarget
				same_Xtarg_and_NL = same_Xtarg(strcmp(nim.get_NLtypes(same_Xtarg),NLtypes{ii})); % set that also have same NL type
			else
				same_Xtarg = [];
				same_Xtarg_and_NL = [];  
			end
            
			if ~isempty(same_Xtarg_and_NL) 
				default_lambdas = nim.subunits(same_Xtarg_and_NL(1)).reg_lambdas;
			elseif ~isempty(same_Xtarg)
				default_lambdas = nim.subunits(same_Xtarg(1)).reg_lambdas;
			else
				default_lambdas = [];  
			end
			
			% Add new subunit
			nim.subunits = cat(1, nim.subunits(:), SUBUNIT(init_filt, mod_signs(ii), NLtypes{ii},Xtargets(ii),NLparams{ii}) ); 
           
			if ~isempty(default_lambdas)
				nim.subunits(end).reg_lambdas = default_lambdas;
			end
			for jj = 1:length(reg_types) % add in user-specified regularization parameters
				assert(reg_vals(ii,jj) >= 0,'regularization hyperparameters must be non-negative');
				nim.subunits(end).reg_lambdas.(reg_types{jj}) = reg_vals(ii,jj);
			end
		end	
	end
	
	function nim = init_spkhist( nim, n_bins, varargin )
	% Usage: nim = nim.init_spkhist( n_bins, varargin )
	% Adds a spike history term with specified parameters to an existing NIM.
	% INPUTS: 
	%   n_bins: number of coefficients in spike history filter
	%   optional flags:
	%     init_spacing: Initial spacing (in time bins) of piecewise constant 'basis functions' (default = 1)
	%     doubling_time: Make bin spacing logarithmically increasing, with given doubling time (default no doubling)
	%     negcon: If flag is included, constrain spike history filter coefs to be non-positive (default false)
	% OUTPUTS: nim: new nim object

		% Default inputs
		init_spacing = 1;
		doubling_time = n_bins;
		negCon = false;
		% Parse input flags
		j = 1; %initialize counter after required input args
		while j <= length(varargin)
			switch lower( varargin{j})
				case 'init_spacing'
					init_spacing = varargin{j+1};
					assert(init_spacing > 0,'invalid init_spacing');
					j = j + 2;
				case 'doubling_time'
					doubling_time = varargin{j+1};
					assert(doubling_time > 0,'invalid doubling time');
					j = j + 2;
				case 'negcon'
					negCon = true;
					j = j + 1;
				otherwise
					error('Invalid input flag'); 
			end	
		end
		
		% COMPUTE RECTANGULAR BASIS FUNCTIONS
		bin_edges = zeros(n_bins+1,1);
		inc = init_spacing; %bin increment size
		pos = 1; count = 0;
		for n = 1:n_bins+1 
			bin_edges(n) = pos; % current bin edge loc
			pos = pos + inc; % move pointer by inc
			count = count + 1; % increase counter for doubling
			if count >= doubling_time % if reach doubling time, reset counter and double inc
				count = 0; inc = inc * 2;
			end
		end
		
		% LOAD INTO A SPK_HIST STRUCT IN NIM
		nim.spk_hist.bin_edges = bin_edges;
		nim.spk_hist.coefs = zeros(n_bins,1); %init filter coefs to 0
		nim.spk_hist.negCon = negCon;  
		nim.spk_hist.spkhstlen = n_bins;  
	end
	
	function nim = init_nonpar_NLs( nim, Xstims, varargin )
	% Usage: nim = nim.init_nonpar_NLs( Xstims, varargin )
	% Initializes the specified model subunits to have nonparametric (tent-basis) upstream NLs
	%
	% INPUTS: 
	%   Xstims: cell array of stimuli
	%   optional flags:
	%     'subs': Index values of set of subunits to make nonpar (default is all)
	%     'lambda_nld2': specify strength of smoothness regularization for the tent-basis coefs
	%     'NLmon': Set to +1 to constrain NL coefs to be monotonic increasing and
	%              -1 to make monotonic decreasing. 0 means no constraint. Default here is +1 (monotonic increasing)
	%     'edge_p': Scalar that determines the locations of the outermost tent-bases 
	%               relative to the underlying generating distribution
	%     'n_bfs': Number of tent-basis functions to use 
	%     'space_type': Use either 'equispace' for uniform bin spacing, or 'equipop' for equipopulated bins
	%     'zeroX': which element of nonlin anchored to zero: 0 = center (default), 1 = edge, dep on monotonicity
	%     'NLrange': if nonlinearity contrained to be within range. Default = [], One value assumes lower bound
	%
	% OUTPUTS: nim: updated nim object
    
		Nsubs = length(nim.subunits);

		% Set defaults
		defaults.subs = 1:Nsubs; % defualt to fitting all subunits 
		defaults.NLmon = 1; % default monotonic increasing TB-coefficients
		defaults.edge_p = 0.05; % relative to the generating distribution (pth percentile) where to put the outermost tent bases
		defaults.n_bfs = 25; % default number of tent basis functions
		defaults.space_type = 'equispace'; % default uninimrm tent basis spacing
		defaults.lambda_nld2 = 0; % default no smoothing on TB coefs
		defaults.zeroX = 0;
		defaults.NLrange = [];
		
		[~,parsed_inputs] = NIM.parse_varargin( varargin, {}, defaults );
		subs = parsed_inputs.subs;

		% Store NL tent-basis parameters
		tb_params = struct( 'NLmon', parsed_inputs.NLmon, 'edge_p',parsed_inputs.edge_p, ...
							'n_bfs',parsed_inputs.n_bfs, 'space_type',parsed_inputs.space_type, ...
							'zeroX',parsed_inputs.zeroX, 'NLrange',parsed_inputs.NLrange );
		for ii = parsed_inputs.subs % load the TB param struct into each subunit we're making nonpar
			nim.subunits(ii).NLnonpar.TBparams = tb_params;
		end
		
		% Compute internal generating functions
		if ~iscell(Xstims)
			tmp = Xstims; clear Xstims
			Xstims{1} = tmp;
		end
		gint = nan(size(Xstims{1},1),Nsubs);
		for ii = 1:length(subs)
			gint(:,subs(ii)) = Xstims{nim.subunits(subs(ii)).Xtarg} * nim.subunits(subs(ii)).filtK;
		end
		
		prev_NL_types = nim.get_NLtypes(); % current NL types   
		for imod = parsed_inputs.subs     
			nim.subunits(imod).NLtype = 'nonpar'; % set the subunit NL type to nonpar  
			nim.subunits(imod) = nim.subunits(imod).rescale_nonparX( Xstims );
			TBx = nim.subunits(imod).NLnonpar.TBx;
			
% 			if strcmp(parsed_inputs.space_type,'equispace') % for equi-spaced bins        
% 				left_edge = NIM.my_prctile( gint(:,imod), parsed_inputs.edge_p );        
% 				right_edge = NIM.my_prctile( gint(:,imod), 100-parsed_inputs.edge_p );
% 				if left_edge == right_edge % if the data is constant over this range (e.g. with a 0 filter), just make the xrange unity
% 					left_edge = right_edge - 0.5;
% 					right_edge = right_edge + 0.5;  
% 				end
% 				spacing = (right_edge - left_edge)/parsed_inputs.n_bfs;
% 				% Adjust the edge locations so one of the bins lands at 0
% 				left_edge = ceil(left_edge/spacing)*spacing;
% 				right_edge = floor(right_edge/spacing)*spacing;
% 				TBx = linspace( left_edge, right_edge, parsed_inputs.n_bfs ); % equispacing
% 			elseif strcmp(parsed_inputs.space_type,'equipop') % for equi-populated binning
% 				if std(gint(:,imod)) == 0  % subunit with constant output
% 					TBx = mean(gint(:,imod)) + linspace(-0.5,0.5,parsed_inputs.n_bfs); % do something sensible
% 				else
% 					TBx = NIM.my_prctile( gint(:,imod), linspace(parsed_inputs.edge_p,100-parsed_inputs.edge_p,parsed_inputs.n_bfs) )'; % equipopulated
% 				end
% 			end
% 			% Set nearest tent basis to 0 so we can keep it fixed during fitting
% 			[~,nearest] = min(abs(TBx));
% 			TBx(nearest) = 0;
			
			% Initalize tent basis coefs   
			switch prev_NL_types{imod}      
				case 'lin'
					TBy = TBx;
				case 'rectlin'
					TBy = TBx + nim.subunits(imod).NLoffset;
					TBy(TBx < -nim.subunits(imod).NLoffset) = 0;
				case 'rectpow'
					TBy = (TBx + nim.subunits(imod).NLoffset).^nim.subunits(imod).NLparams(1);
					TBy(TBx < -nim.subunits(imod).NLoffset) = 0;
				case 'quad'
					TBy = (TBx + nim.subunits(imod).NLoffset).^2;
				case 'softplus'
					TBy = log(1 + exp(nim.subunits(imod).NLparams(1)*(TBx + nim.subunits(imod).NLoffset)));
				case 'nonpar'
					fprintf('upstream NL already set as nonparametric\n');
					return
				otherwise
					error('Unsupported NL type'); 
			end
			nim.subunits(imod).NLoffset = 0;
			nim.subunits(imod).NLnonpar.TBy = TBy;
			%nim.subunits(imod).NLnonpar.TBx = TBx;
			nim.subunits(imod).reg_lambdas.nld2 = parsed_inputs.lambda_nld2;
			nim.subunits(imod).TBy_deriv = nim.subunits(imod).get_TB_derivative(); % calculate derivative of Tent-basis coeffics
		end	
	end

	function [LL, pred_rate, mod_internals, LL_data] = eval_model( nim, Robs, Xstims, varargin )
	% Usage: [LL, pred_rate, mod_internals, LL_data] = nim.eval_model( Robs, Xstims, <eval_inds>, varargin )
	% Evaluates the model on the supplied data
	%
	%   INPUTS:
	%     Robs: vector of observed data (leave empty [] if not interested in LL)
	%     Xstims: cell array of stimuli
	%     <eval_inds>: optional vector of indices on which to evaluate the model
	%     optional flags:
	%       'gain_funs': [TxK] matrix specifying gain at each timepoint for each subunit
	%   OUTPUTS:
	%     LL: log-likelihood per spike
	%     pred_rate: predicted firing rates (in counts/bin)
	%     mod_internals: struct containing the internal components of the model prediction
	%       G: is the total generating signal (not including the constant offset theta). 
	%          This is the sum of subunit outputs (weighted by their subunit weights w)
	%       fgint: is the output of each subunit
	%       gint: is the output of each subunits linear filter
	%     LL_data: struct containing more detailed info about model performance:
	%       filt_pen: total regularization penalty on filter coefs 
	%       NL_pen total regularization penalty on filter upstream NLs
	%       nullLL: LL of constant-rate model

		Nsubs = length(nim.subunits); % number of subunits
      
		% PROCESS INPUTS
		[eval_inds,parsed_options] = NIM.parse_varargin( varargin );
		NIM.validate_parsed_options( parsed_options, {'gain_funs','fit_offsets'}, 0 );
		gain_funs = []; % default has no gain_funs
		if isfield( parsed_options, 'gain_funs' )
			gain_funs = parsed_options.gain_funs;
		end

		if ~iscell(Xstims)
			tmp = Xstims; clear Xstims
			Xstims{1} = tmp;
		end
		if isempty(Robs); Robs = zeros(size(Xstims{1},1),1); end  % if empty, make null list
		if size(Robs,2) > size(Robs,1); Robs = Robs'; end;  % make Robs a column vector
		nim.check_inputs(Robs,Xstims,eval_inds,gain_funs);  % make sure input format is correct
		if nim.spk_hist.spkhstlen > 0  % add in spike history term if needed
			Xspkhst = nim.create_spkhist_Xmat( Robs );
		else			
			Xspkhst = [];  
		end

		if ~isempty(eval_inds) % if specifying a subset of indices to train model params
			for nn = 1:length(Xstims)
				Xstims{nn} = Xstims{nn}(eval_inds,:); % grab the subset of indices for each stimulus element
			end
			Robs = Robs(eval_inds);
			if ~isempty(Xspkhst); Xspkhst = Xspkhst(eval_inds,:); end;
			if ~isempty(gain_funs); gain_funs = gain_funs(eval_inds,:); end;
		end
		[G, fgint, gint] = nim.process_stimulus( Xstims, 1:Nsubs, gain_funs );
		if nim.spk_hist.spkhstlen > 0 % add in spike history term if needed
			G = G + Xspkhst*nim.spk_hist.coefs(:);  
		end
		
		pred_rate = nim.apply_spkNL( G + nim.spkNL.theta ); % apply spiking NL
		[LL,norm_fact] = nim.internal_LL( pred_rate, Robs ); % compute LL
		LL = LL/norm_fact; % normalize by spikes (or time points for Gaussian noise distribution)
    
		if nargout > 2 % if outputting model internals
			mod_internals.G = G;
			mod_internals.fgint = fgint;
			mod_internals.gint = gint;
		end
		
		if nargout > 3  % if we want more detailed model evaluation info, create an LL_data struct
			LL_data.LL = LL;
			[filt_penalties,NL_penalties] = nim.get_reg_pen(); % get regularization penalty for each subunit
			LL_data.filt_pen = sum(filt_penalties); %/norm_fact; % normalize by number of spikes
			LL_data.NL_pen = sum(NL_penalties)/norm_fact;
			avg_rate = mean(Robs);
			null_prate = ones(length(Robs),1)*avg_rate;
			nullLL = nim.internal_LL(null_prate,Robs)/norm_fact;
			LL_data.nullLL = nullLL;
		end		
	end

	function [LLs,LLnulls,pred_rates] = eval_model_reps( nim, RobsR, Xstims, varargin )
	% Usage: [LLs,LLnulls,pred_rates] = nim.eval_model_reps( Robs, Xstims, <eval_inds>, <varargin> )
	% Evaluates the model on the supplied data. In this case RobsR would be a NT x Nreps matrix
   
		Nreps = size(RobsR,2);
		pred_rates = zeros(size(RobsR));
		LLs = zeros(Nreps,1); 	LLnulls = zeros(Nreps,1);
		for nn = 1:Nreps
			[LLs(nn),rs,~,LLdata] = eval_model( nim, RobsR(:,nn), Xstims, varargin{:} );
			if nargout > 2  
				pred_rates(1:length(rs),nn) = rs; % if eval_inds is included in varargin, rs will be smaller
			end
			LLnulls(nn) = LLdata.nullLL;
		end	
		if nargout > 2
			pred_rates = pred_rates(1:length(rs),:);
		end
	end

	function [filt_SE,hessMat] = compute_filter_SEs( nim, Robs, Xstims, varargin )
	% Usage: [filt_SE,hessMat] = compute_filter_SEs( nim, Robs, Xstims, <eval_inds>, varargin )
	% Computes standard error estimates for the filter coefficients of a set of subunits, 
	% based on the outer-product of gradients estimator of the fisher info matrix
	%
	% INPUTS:
	%   Robs: vector of observed data
	%   Xstims: cell array of stimuli
	%   <eval_inds>: optional vector of indices on which to evaluate the model
	%   optional flags:
	%     'gain_funs': [TxK] matrix specifying gain at each timepoint for each subunit
	%     'subs': index values of subunits to compute filter SEs for
	%   OUTPUTS:
	%     filt_SE: cell array of standard error estimates of the filter coefs
	%     hessMat: estimate of the second derivative of the log-posterior
   
		Nsubs = length(nim.subunits); % number of subunits
		NT = length(Robs); % number of time points

		% PROCESS INPUTS
		defaults.eval_inds = nan; % this default means evaluate on all data
		defaults.gain_funs = []; % default has no gain_funs 
		defaults.subs = 1:Nsubs; % default is all subunits
		[eval_inds,parsed_options] = NIM.parse_varargin( varargin, {}, defaults );
		validate_parsed_options( parsed_options, {'subs','gain_funs'} );
		sub_inds = parsed_options.subs;
		
		if size(Robs,2) > size(Robs,1); Robs = Robs'; end; % make Robs a column vector
    
		mod_weights = [nim.subunits(sub_inds).weight];    
		mod_Xtargs = [nim.subunits(sub_inds).Xtarg];

		[~,pred_rate,mod_internals] = nim.eval_model( Robs, Xstims, eval_inds, 'gain_funs', parsed_options.gain_funs );
		G = mod_internals.G + nim.spkNL.theta; % add in spkNL offset to generating signal

		r_deriv = nim.apply_spkNL_deriv(G);  % derivative of spkNL
		LL_deriv = nim.internal_LL_deriv( pred_rate, Robs ); % derivative of LL function
		residual = r_deriv.*LL_deriv; % F'[]*r'[]

		filtKs = nim.get_filtKs( sub_inds ); % all filter coefs as cell array
		filt_dims = cellfun(@(x) length(x), filtKs ); % number of coefs for each filter

		% Compute second derivative of log-posterior wrt L2 penalties
		Tmats = nim.make_Tikhonov_matrices(); % generate regularization matrices
		penHessMat = zeros( sum(filt_dims) ); % init hessian of penalty terms   
		for ii = 1:length(Tmats) % loop over the derivative regularization matrices
			cur_subs = find(mod_Xtargs == Tmats(ii).Xtarg); % set of subunits acting on the stimulus given by this Tmat
			for jj = 1:length(cur_subs)
				irange = sum(filt_dims(1:cur_subs(jj)-1)) + (1:filt_dims(cur_subs(jj))); % range of parameter indices corresponding to this subunits filter        
				pen_hess = 2*Tmats.Tmat' * Tmats(ii).Tmat;
				penHessMat(irange,irange) = penHessMat(irange,irange) + pen_hess*nim.get_reg_lambdas(Tmats(ii).type,'subs',sub_inds(cur_subs(jj)));      
			end	
		end
		l2_lambdas = nim.get_reg_lambdas( 'l2', 'subs',sub_inds );
		if any(l2_lambdas > 0) % now for straight L2 penalties
			l2_subs = find(l2_lambdas > 0);
			for ii = 1:length(l2_subs)     
				irange = sum(filt_dims(1:l2_subs(ii)-1)) + (1:filt_dims(l2_subs(ii))); % range of parameter indices corresponding to this subunits filter        
				pen_hess = 2*eye(length(irange));
				penHessMat(irange,irange) = penHessMat(irange,irange) + l2_lambdas(l2_subs(ii))*pen_hess;      
			end	
		end

		gradMat = zeros(NT,sum(filt_dims));
		for ii = 1:length(sub_inds) % compute gradient of LL wrt all filter coefs
			irange = sum(filt_dims(1:ii-1)) + (1:filt_dims(ii)); % range of parameter indices corresponding to this subunits filter
			filt_fd_ii = nim.subunits(sub_inds(ii)).apply_NL_deriv(mod_internals.gint(:,sub_inds(ii))); % first derivative of ith subunits upstream NL wrt its input arg
			gradMat(:,irange) = mod_weights(sub_inds(ii))*bsxfun(@times,Xstims{mod_Xtargs(ii)},filt_fd_ii.*residual); 
		end
		
		hessMat = gradMat'*gradMat; % use outer product of gradients to estimate Fisher info matrix (ref: Greene W. Econometric Analysis 7th ed. 2011. Eqn 14-18
		hessMat = hessMat + penHessMat; % add in penalty term    
		if min(eig(hessMat)) < 0; warning( 'hessian not positive semi-definite' ); end; % make sure it's positive semi-def    
		inv_hess = pinv(hessMat); % invert to get parameter covariance mat    
		allK_SE = sqrt(diag(inv_hess)); % take sqrt of diagonal component as SE

		% Parse into cell array
		filt_SE = cell(length(sub_inds),1);
		for ii = 1:length(sub_inds)       
			irange = sum(filt_dims(1:ii-1)) + (1:filt_dims(ii));    
			filt_SE{ii} = allK_SE(irange);    
		end
	end

end

%% ********************** Display Methods *********************************
methods

	function [] = display_spkNL( nim, varargin )
	% Usage: [] = nim.display_spikingNL( G, varargin )
	%
	% Plot spiking nonlinearity in subplot panel specified by plotloc. Will only plot in presence of 
	% total ouputs
	%
	% INPUTS:
	%   G: the generating potential that the spiking nonlinearity acts on
	%   Optional inputs:
	%     None currently
	
		CUTOFF = 0.5; % percentile under which to not display G distribution (and plot nonlinearity)
		
		n_hist_bins = 80; % internal parameter determining histogram resolution

		[G,parsed_options] = NIM.parse_varargin( varargin ); % first varargin must be G
		assert((nargin > 1) && ~isempty(G), 'G-distribution required in order to plot spiking nonlinearity.' )

		left_edge = NIM.my_prctile( G, CUTOFF );        
		right_edge = NIM.my_prctile( G, 100-CUTOFF );
		[Gdist_y,Gdist_x] = hist( G, left_edge:((right_edge-left_edge)/n_hist_bins):right_edge ); % histogram the generating signal
    
		% This is a hack to deal with cases where the threshold linear terms (creates a max value of G that doesn't dwarf rest)
		if Gdist_y(1) > 2*Gdist_y(2)
			Gdist_y(1) = 1.5*Gdist_y(2);
		end
		
		if strcmp(nim.spkNL.type,'logistic')
			NLx = linspace( Gdist_x(1), Gdist_x(end) + (Gdist_x(end)-Gdist_x(1))/4,100 );
		else
			NLx = Gdist_x;
		end

		NLy = nim.apply_spkNL(NLx + nim.spkNL.theta);
		if ~strcmp(nim.spkNL.type,'lin')
			NLy = NLy/nim.stim_params(1).dt; % convert to correct firing rate units
		end
		
		plot(NLx,NLy,'b','LineWidth',1)
		hold on
		plot(Gdist_x,Gdist_y/max(Gdist_y)*max(NLy)/4*3,'r')
		xlim([Gdist_x(1) Gdist_x(end)])
		ylim([min([0 min(NLy)]) max(NLy)]);
		xlabel('g')
		
		if strcmp(nim.spkNL.type,'lin')
			ylabel('Output','fontsize',12);
		else			
			ylabel('Firing rate (Hz)','fontsize',12);
		end
		%set(gca,'YTick',[]);
		title('Spiking NL','fontsize',12)
	end
	
	function [] = display_spike_history( nim, varargin )
	% Usage: [] = nim.display_spikingNL( varargin )
	%
	% Plot spike-history term, either scaled or not.
	% INPUTS:
	%   Optional inputs:
	%     'scaled': using actual time bins (default not)

		[~,parsed_options] = NIM.parse_varargin( varargin );

		if isfield(parsed_options,'scaled')
			stairs( nim.spk_hist.bin_edges(1:end-1)*nim.stim_params(1).dt, nim.spk_hist.coefs );
			xlim(nim.spk_hist.bin_edges([1 end])*nim.stim_params(1).dt)
		else
			plot( nim.spk_hist.coefs );
			xlim( [1 length(nim.spk_hist.coefs)] )
		end
		
    xl = xlim();
    line(xl,[0 0],'color','k','linestyle','--');
    xlabel('Time');
    title('Spike history','fontsize',8)	
	end	

	function [] = display_subunit_filters( nim, varargin )
	% Usage: [] = nim.display_subunit_filters( G, varargin )
	% Display only subunit filters, arranged in efficient grids. If one-d filters, then display on 
	% same plot
	
		figure
		Nsubs = length(nim.subunits);
		if prod(nim.stim_params(1).dims(2:3)) > 1
			Ncols = 2*ceil(sqrt(Nsubs/4));
			Nrows = ceil(Nsubs/Ncols*2);
		
			for nn = 1:Nsubs
				% Plot filters (allowing 2 positions for each display) 
				dims = nim.stim_params(nim.subunits(nn).Xtarg).dims;
				nim.subunits(nn).display_filter( dims, [Nrows Ncols 2*(nn-1)+1], varargin{:} );
				subplot( Nrows, Ncols, 2*(nn-1)+1 );
				if nim.subunits(nn).weight > 0
					stype = 'exc';
				else
					stype = 'sup';
				end
				title( sprintf( 'Sub #%d (%s)', nn, stype ) )
			end
		else
			clrs = 'kbcgrmkbcgrmkbcgrm';
			hold on
			for nn = 1:Nsubs
				plot(nim.subunits(nn).filtK, clrs(nn) )
				legend_list{nn} = sprintf('k%d', nn );
			end
			legend(legend_list)
		end
	end

end

%% ********************** Hidden Methods **********************************
methods (Hidden)

	function [G, fgint, gint] = process_stimulus( nim, Xstims, sub_inds, gain_funs )
	% Usage: [G, fgint, gint] = nim.process_stimulus( Xstims, sub_inds, gain_funs )
	%
	% Processes the stimulus with the subunits specified in sub_inds
	% INPUTS:
	%   Xstims: stimulus as cell array
	%   sub_inds: set of subunits to process
	%   gain_funs: temporally modulated gain of each subunit
	% OUTPUTS:
	%   G: summed generating signal
	%   fgint: output of each subunit
	%   gint: output of each subunit filter

		NT = size(Xstims{1},1);
		if isempty(sub_inds);
			[G,fgint,gint] = deal(zeros(NT,1));
			return  
		end
		Nsubs = length(sub_inds);
		Xtarg_set = [nim.subunits(sub_inds).Xtarg];
		un_Xtargs = unique(Xtarg_set); % set of Xtargets
		filter_offsets = [nim.subunits(sub_inds).NLoffset]; % set of filter offsets
		filtKs = cell(Nsubs,1);
		for ii = 1:Nsubs % loop over subunits, get filter coefs
			filtKs{ii} = nim.subunits(sub_inds(ii)).get_filtK();
		end
		gint = zeros(size(Xstims{1},1),Nsubs);
		for ii = 1:length(un_Xtargs) % loop over the unique Xtargs and compute the generating signals for all relevant filters
			cur_subs = find(Xtarg_set == un_Xtargs(ii)); % set of targeted subunits that act on this Xtarg
			gint(:,cur_subs) = Xstims{un_Xtargs(ii)} * cat(2,filtKs{cur_subs}); % apply filters to stimulus
		end
		gint = bsxfun(@plus,gint,filter_offsets); % add offsets to filter outputs
		fgint = gint; % init subunit outputs by filter outputs
		for ii = 1:Nsubs
			if ~strcmp(nim.subunits(sub_inds(ii)).NLtype,'lin')
				fgint(:,ii) = nim.subunits(sub_inds(ii)).apply_NL(gint(:,ii)); %apply upstream NL
			end
		end
		if ~isempty(gain_funs)
			fgint = fgint.*gain_funs(:,sub_inds); % apply gain modulation if needed
		end
		G = fgint*[nim.subunits(sub_inds).weight]';
	end

	function [LL,norm_fact] = internal_LL( nim, rPred, Robs )
	% Usage: [LL,norm_fact] = nim.internal_LL( rPred, Robs )
	% Internal evaluatation method for computing the total LL associated with the predicted rate rPred,
	% given observed data Robs returns total LL as well as an appropriate normalization factor 
      
		switch nim.noise_dist
			case 'poisson' % LL = Rlog(r) - r + C
				LL = sum(Robs .* log(rPred) - rPred);
				norm_fact = sum(Robs); % normalize by total number of spks
			case 'bernoulli' % LL = R*log(r) + (1-R)*log(1-r)
				LL = nansum(Robs.*log(rPred) + (1-Robs).*log(1-rPred));
				norm_fact = sum(Robs); % normalize by total number of spks
			case 'gaussian' % LL = (r-R)^2 + c
				LL = -sum((rPred - Robs).^2);
				norm_fact = length(Robs); % normalize by number of time points 
		end	
	end

	function LL_deriv = internal_LL_deriv( nim, rPred, Robs )
	% Usage: LL_deriv = internal_LL_deriv( nim, rPred, Robs )
	% Computes the derivative of the LL wrt the predicted rate at rPred, given Robs (as a vector over time)
    
		switch nim.noise_dist
			case 'poisson' %LL'[r] = R/r - 1
				LL_deriv = Robs./rPred - 1;                    
			case 'bernoulli' %LL'[r] = R/r - (1-R)/(1-r)
				LL_deriv = Robs./rPred - (1-Robs)./(1-rPred);
			case 'gaussian' %LL'[r] = 2*(r-R)
				LL_deriv = -2*(rPred - Robs);
		end
	end

	function rate = apply_spkNL( nim, gen_signal )
	% Usage: rate = apply_spkNL( nim, gen_signal )
	% Applies the spkNL function to the input gen_signal. 
	% NOTE: the offset term should already be added to gen_signal
          
		switch nim.spkNL.type
			case 'lin' %F[x;beta] = beta*x
				rate = gen_signal*nim.spkNL.params(1);
				
			case 'rectpow' %F[x; beta, gamma] = (beta*x)^gamma iff x > 0; else 0
				rate = (gen_signal*nim.spkNL.params(1)).^nim.spkNL.params(2);
				rate(gen_signal < 0) = 0;
				
			case 'exp' %F[x; beta] = exp(beta*x)
				rate = exp(gen_signal*nim.spkNL.params(1));
				
			case 'softplus' %F[x; beta, alpha] = alpha*log(1+exp(beta*x))
				max_g = 50; %to prevent numerical overflow
				gint = gen_signal*nim.spkNL.params(1);
				rate = nim.spkNL.params(2)*log(1 + exp(gint));
				rate(gint > max_g) = nim.spkNL.params(2)*gint(gint > max_g);
				
			case 'logistic'
				rate = 1./(1 + exp(-gen_signal*nim.spkNL.params(1))); 
				
		end
		if ismember(nim.noise_dist,{'poisson','bernoulli'}) %cant allow rates == 0 because LL is undefined
			rate(rate < nim.min_pred_rate) = nim.min_pred_rate; 
		end
		if strcmp(nim.noise_dist,'bernoulli') %cant allow rates == 1 because LL is undefined    
			rate(rate > (1 - nim.min_pred_rate)) = 1 - nim.min_pred_rate; 	
		end			
	end

	function rate_deriv = apply_spkNL_deriv( nim, gen_signal, thresholded_inds )
	% Usage: rate_deriv = apply_spkNL_deriv( nim, gen_signal, thresholded_inds )
	% Applies the derivative of the spkNL to the input gen_signal.
	% Again, gen_signal should have the offset theta already added in.
    
		if nargin < 3
			thresholded_inds = []; % this just specifies the index values where we've had to apply thresholding on the predicted rate to avoid Nan LLs    
		end
		switch nim.spkNL.type
       
			case 'lin' % F'[x; beta] = beta;
				rate_deriv = nim.spkNL.params(1)*ones(size(gen_signal));
        
			case 'rectpow' % F'[x; beta, gamma] = gamma*beta^gamma*x^(gamma-1) iff x > 0; else 0
				rate_deriv = nim.spkNL.params(2)*nim.spkNL.params(1)^nim.spkNL.params(2)*...
						gen_signal.^(nim.spkNL.params(2)-1);
				rate_deriv(gen_signal < 0) = 0;
				
			case 'exp' % F'[x; beta] = beta*exp(beta*x)
				rate_deriv = nim.spkNL.params(1)*exp(nim.spkNL.params(1)*gen_signal);
				
			case 'softplus' % F[x; beta, alpha] = alpha*beta*exp(beta*x)/(1+exp(beta*x))
				max_g = 50; % to prevent numerical overflow
				gint = gen_signal*nim.spkNL.params(1);
				rate_deriv = nim.spkNL.params(1)*nim.spkNL.params(2)*exp(gint)./(1 + exp(gint));
				rate_deriv(gint > max_g) = nim.spkNL.params(1)*nim.spkNL.params(2); % e^x/(1+e^x) => 1 for large x
				
			case 'logistic'
				rate_deriv = nim.spkNL.params(1)*exp(-gen_signal*nim.spkNL.params(1))./...
						(1 + exp(-gen_signal*nim.spkNL.params(1))).^2; % e^(-x)/(1+e^(-x))^2
		end
		if ismember(nim.noise_dist,{'poisson','bernoulli'}) % cant allow rates == 0 because LL is undefined
			rate_deriv(thresholded_inds) = 0; % if thresholding the rate to avoid undefined LLs, set deriv to 0 at those points  
		end
	end

	function rate_grad = spkNL_param_grad( nim, params, x )
	% Usage: rate_grad = nim.spkNL_param_grad( params, x )
	% Computes the gradient of the spkNL function with respect to its 
	% parameters (subroutine for optimizing the spkNL params)

		rate_grad = zeros(length(x),length(params));
		switch nim.spkNL.type      
			case 'lin' % F[x;beta,theta] = beta*(x+theta)
				rate_grad(:,1) = x; %dr/dbeta = x
				rate_grad(:,2) = ones(size(x)); %dr/dtheta = 1
			case 'rectpow' % F[x;beta,gamma,theta] = (beta*(x+theta))^gamma iff (x + theta) > 0 
				temp = params(1)*(x + params(3)); % (beta*x+theta)
				temp(temp < 0) = 0; % threshold at 0
				rate_grad(:,1) = params(2)*temp.^(params(2)-1).*(x + params(3)); % dr/dbeta 
				rate_grad(:,2) = temp.^params(2).*log(temp); % dr/dgamma
				rate_grad(temp == 0,2) = 0; % define this as 0
				rate_grad(:,3) = params(2)*temp.^(params(2)-1); % dr/dtheta
			case 'exp' % F[x;beta, theta] = exp(beta*(x+theta))
				temp = exp(params(1)*(x + params(end)));
				rate_grad(:,1) = (x + params(end)).*temp; % dr/dbeta = (x+theta)*exp(beta*(x+theta))
				rate_grad(:,2) = params(1).*temp; % dr/dtheta = beta*exp(beta*(x+theta))
			case 'softplus' % F[x;beta, alpha, theta] = alpha*log(1+exp(beta*(x+theta)))
				temp = params(2)*exp(params(1)*(x + params(3)))./(1 + exp(params(1)*(x + params(3)))); % alpha*exp(beta*(x+theta))/(1 + exp(beta*(x+theta)))
				rate_grad(:,1) = temp.*(x + params(3)); % dr/dbeta = temp*(x + theta)
				rate_grad(:,2) = log(1 + exp(params(1)*(x + params(3)))); % dr/dalpha = log[]
				rate_grad(:,3) = temp.*params(1); % dr/dtheta = temp*beta
			case 'logistic' % F[x;beta, theta] = 1/(1 + exp(-beta*(x+theta)))
				temp = exp(-params(1)*(x+params(2)))./(1 + exp(-params(1)*(x + params(2)))).^2; % exp(-beta*(x+theta))/(1+exp(-beta(x+theta)))^2
				rate_grad(:,1) = temp.*(x + params(2)); % dr/dbeta = temp*(x+theta)
				rate_grad(:,2) = temp.*params(1); % dr/dtheta = temp*beta
			otherwise
				error('unsupported spkNL type');    
		end            
	end
	
	function Tmats = make_Tikhonov_matrices( nim )
	% Usage: Tmats = nim.make_Tikhonov_matrices()
	% Creates a struct containing the Tikhonov regularization matrices, given the stimulus and regularization 
	% parameters specified in the nim
    
		Nstims = length(nim.stim_params); % number of unique stimuli 
		Xtargs = [nim.subunits(:).Xtarg];

		deriv_reg_types = nim.allowed_reg_types(strncmp(nim.allowed_reg_types,'d',1)); % set of regularization types where we need a Tikhonov matrix
		cnt = 1;
		Tmats = [];
		for ii = 1:Nstims % for each stimulus
			cur_subs = find(Xtargs == ii); % get set of subunits acting on this stimuls
			for jj = 1:length(deriv_reg_types) % check each possible derivative regularization type
				cur_lambdas = nim.get_reg_lambdas(deriv_reg_types{jj},'subs',cur_subs);
				if any(cur_lambdas > 0)
					cur_Tmat = NIM.create_Tikhonov_matrix(nim.stim_params(ii),deriv_reg_types{jj});
					Tmats(cnt).Tmat = cur_Tmat;
					Tmats(cnt).Xtarg = ii;
					Tmats(cnt).type = deriv_reg_types{jj};
					cnt = cnt + 1;
				end
			end
		end      
	end

	function Xspkhst = create_spkhist_Xmat( nim, Robs )
	% Usage: Xspkhst = nim.create_spkhist_Xmat( Robs )
	% Creates an X-matrix out of observed spike train Robs (using bin_edges defined in spkhist initialization)

		NT = length(Robs);
		bin_edges = nim.spk_hist.bin_edges;
		maxlag = max(bin_edges);
		spkbns = NIM.convert_to_spikebins(Robs); % spike bins

		Tmat = zeros(NT + maxlag,length(bin_edges)-1);
		for i = 1:length(spkbns)
			for j = 1:length(bin_edges)-1
				Tmat(spkbns(i)+(bin_edges(j):(bin_edges(j+1)-1)),j) = Tmat(spkbns(i)+(bin_edges(j):(bin_edges(j+1)-1)),j) + 1;
			end
		end

		Xspkhst =  Tmat(1:NT,:); % concatenate onto X_matrix
	end

	function Tmat = make_NL_Tmat( nim )
	% Usage: Tmat = nim.make_NL_Tmat()
	% Makes Tikhonov matrix for smoothness regularization of the TB NLs
        
		nonpar_set = find(strcmp(nim.get_NLtypes(),'nonpar'));
		assert(~isempty(nonpar_set),'no nonparametric NLs found');
		n_tbs = length(nim.subunits(nonpar_set(1)).NLnonpar.TBx); %number of TBx (assume this is the same for all subunits)!
		et = ones(n_tbs,1);
		et([1 end]) = 0; %free boundaries
		Tmat = spdiags([et -2*et et], [-1 0 1], n_tbs, n_tbs)';
	end

	function [] = check_inputs( nim, Robs, Xstims, sub_inds, gain_funs )
	% Usage: [] = nim.check_inputs( Robs, Xstims, sub_inds, gain_funs )
	% Checks the format of common inputs params
        
		if nargin < 4
			gain_funs = [];
		end
		if nargin < 5
			sub_inds = nan;  
		end
		Nsubs = length(nim.subunits);
		for n = 1:Nsubs % check that stimulus dimensions match
			[NT,filtLen] = size(Xstims{nim.subunits(n).Xtarg}); % stimulus dimensions
			assert( filtLen == prod(nim.stim_params(nim.subunits(n).Xtarg).dims), 'Xstim dims dont match stim_params' );  
		end
		assert( length(unique(cellfun(@(x) size(x,1),Xstims))) == 1, 'Xstim elements need to have same size along first dimension' );
		assert( size(Robs,2) == 1, 'Robs must be a vector' );
		assert( iscell(Xstims), 'Xstims must for input as a cell array' );
		if ~isempty(gain_funs)
			assert( size(gain_funs,1) == NT & size(gain_funs,2) == Nsubs, 'format of gain_funs is incorrect' );
		end
		if ~isnan(sub_inds)
			assert( min(sub_inds) > 0 & max(sub_inds) <= NT, 'invalid data indices specified' );
		end
	end

	function nim = set_subunit_scales( nim, fgint )
	% Usage: nim = nim.set_subunit_scales( fgint )
	% Sets the 'scale' of each subunit based on the SD of its output fgint

		fgint_SDs = std(fgint);
		for ii = 1:length(nim.subunits)
			nim.subunits(ii).scale = fgint_SDs(ii);
		end
	end
	
end

%% ********************** Static Methods **********************************
methods (Static)

	function stim_params = create_stim_params(dims,varargin)
	% Usage: stim_params = create_stim_params(stim_dims,<varargin>)
	% Creates a struct containing stimulus parameters
	%
	% INPUTS:
	%     dims: dimensionality of the (time-embedded) stimulus, in the
	%         form: [nLags nXPix nYPix]. For 1 spatial dimension use only nXPix
	%     optional_flags:
	%       ('stim_dt',stim_dt): time resolution (in ms) of Xmatrix (used only for plotting)
	%       ('stim_dx',stim_dx): spatial resolution (in deg) of Xmatrix (used only for plotting)
	%       ('upsampling',up_samp_fac): optional up-sampling of the stimulus from its raw form
	%       ('tent_spacing',tent_spacing): optional spacing of tent-basis functions when using a tent-basis
	%         representaiton of the stimulus (allows for the stimulus filters to be
	%         represented at a lower time resolution than other model components). 
	%         Default = []: no tent_bases
	%       ('boundary_conds',boundary_conds): vector of boundary conditions on each
	%           dimension (Inf is free, 0 is tied to 0, and -1 is periodi)
	%       ('split_pts',split_pts): specifies an internal boundary as a 3-element vector: [direction boundary_ind boundary_cond]
	% OUTPUTS:
	%     stim_params: struct of stimulus parameters
    
		% Set defaults
		stim_dt = 1; %cefault to unitless time
		stim_dx = 1; %default unitless spatial resolution
		up_samp_fac = 1; %default no temporal up-sampling
		tent_spacing = []; %default no tent-bases
		boundary_conds = [0 0 0]; %tied to 0 in all dims
		split_pts = []; %no internal boundaries
		
		% Parse inputs
		j = 1;
		while j <= length(varargin)
			switch lower(varargin{j})
				case 'stim_dt'
					stim_dt = varargin{j+1};
					j = j + 2;
				case 'stim_dx'
					stim_dx = varargin{j+1};
					j = j + 2;
				case 'upsampling'
					up_samp_fac = varargin{j+1};
					j = j + 2;
				case 'tent_spacing'
					tent_spacing = varargin{j+1};
					j = j + 2;
				case 'boundary_conds'
					boundary_conds = varargin{j+1};
					j = j + 2;
				case 'split_pts'
					split_pts = varargin{j+1};
					j = j + 2;
				otherwise
					error('Invalid input flag'); 
			end	
		end
		
		% Make sure stim_dims input has form [nLags nXPix nYPix] and concatenate 1's if necessary    
		while length(dims) < 3 %pad dims with 1s for book-keeping
			dims = cat(2,dims,1);
		end
		while length(boundary_conds) < 3
			boundary_conds = cat(2,boundary_conds,0); %assume free boundaries on spatial dims if not specified
		end
		
		dt = stim_dt/up_samp_fac; %model fitting dt
		stim_params = struct('dims',dims,'dt',dt,'dx',stim_dx,'up_fac',up_samp_fac,...
			'tent_spacing',tent_spacing,'boundary_conds',boundary_conds,'split_pts',split_pts);
	end 
	
	function Robs = Spks2Robs( spks, dt, NT )
	% Usage: Robs = Spks2Robs( spks, dt, NT )
	% Produces Robs given binsize dt and number of bins
	
		Robs = histc( spks, (0:(NT-1))*dt );
		Robs = Robs(:);
	end
	
	function RobsReps = Spks2Robs_reps( spksR, dt, NT )
	% Usage: RobsReps = Spks2Robs_reps( spksR, dt, NT )
	% makes NTxNRrep binned spike from spksR (saved as list of spike times with repeats separated
	% by -1s
	
		Rlocs = [0 find(spksR < 0)];
		Nreps = length(Rlocs)-1;
		RobsReps = zeros( NT, Nreps );
		for nn = 1:Nreps
			RobsReps(:,nn) = NIM.Spks2Robs( spksR((Rlocs(nn)+1):(Rlocs(nn+1)-1)), dt, NT );
		end		
	end
	
	function [Uindx,XVindx] = generate_XVfolds( NTXstim, Nfold, XVfolds )
	%	Usage: [Uindx,XVindx] = generate_XVfolds( NTXstim, <Nfold>, <XVfolds> )
	%	Generates Uindx and XVindx to use for fold-Xval.
	%
	%	INPUTS:
	%		NTXstim = size(Xstim,1);  number of time steps in the stimulus
	%		Nfold = number of folds (e.g., 5-fold). Default = 5
	%		XVfolds = which folds to set aside for X-val. Can be more than 1. Default = in the middle (3 for 5)
	%	OUTPUTS:
	%		Uindx = indices of design matrix (e.g., X-stim) to use for model fitting
	%		XVindx = indices of design matrix (e.g., X-stim) to use for cross-validation
									
		if (nargin < 3) || isempty(Nfold)
			Nfold = 5;
		end
		if (nargin < 4) 
			XVfolds = ceil(Nfold/2);
		end

		NT = NTXstim;
		XVindx = [];
		for nn = 1:length(XVfolds)
			XVindx = cat(1, XVindx, (floor((XVfolds(nn)-1)*NT/Nfold+1):floor(XVfolds(nn)*NT/Nfold))' );
		end
		Uindx = setdiff((1:NT)',XVindx);
	end
	
end

%% ********************** Static Hidden Methods ***************************
methods (Static, Hidden)

	function optim_params = set_optim_params( optimizer, input_params, silent )
	% internal function that checks stim_params struct formatting, and 
	% initializes default values for the given optimizer

		optim_params.maxIter = 500; %maximum number of iterations
		if ~isempty(input_params) %check if silent is specified in the input params, 
			if isfield(input_params,'silent')
				silent = input_params.('silent'); %if so, over-ride with that value
				input_params = rmfield(input_params,'silent'); %remove silent field to allow parsing of other input params
			end	
		end
		
		switch optimizer
      
			case 'fminunc'
				optim_params.TolX = 1e-7; % termination tolerance on X
				optim_params.TolFun = 1e-7; % termination tolerance on the function value
				optim_params.LargeScale = 'off'; % dont use large-scale method
				optim_params.HessUpdate = 'bfgs'; % update Hessian using BFGS
				optim_params.GradObj = 'on'; % use gradient
				if silent
					optim_params.Display = 'off';
				else
					optim_params.Display = 'iter';
				end
				
			case 'minFunc'
				optim_params.optTol = 1e-5; %[minFunc] termination tolerance on first order optimality (max(abs(grad))
				optim_params.progTol = 1e-8; %[minFunc] termination tolerance on function/parameter values
				optim_params.Method = 'lbfgs'; % [minFunc] method
				optim_params.verbose = 2; % display full iterative output
				if silent
					optim_params.Display = 'off';
				else
					optim_params.Display = 'iter';
				end
					
			case 'fmincon'
				optim_params.Algorithm = 'active-set';
				optim_params.GradObj = 'on';
				optim_params.TolX = 1e-7;
				optim_params.TolFun = 1e-7;
				if silent
					optim_params.Display = 'off';
				else
					optim_params.Display = 'iter';
				end
				
			case 'L1General_PSSas'
				optim_params.optTol = 1e-5;
				optim_params.progTol = 1e-8;
				optim_params.verbose = 2;
				if silent
					optim_params.verbose = 0;
				else
					optim_params.verbose = 2;
				end
				
			case 'minConf_TMP'
				optim_params.optTol = 1e-5;
				optim_params.progTol = 1e-8;
				optim_params.verbose = 2;
				if silent
					optim_params.verbose = 0;
				else
					optim_params.verbose = 2;
				end
				
			case 'fminsearch'
				if silent
					optim_params.Display = 'off';
				else
					optim_params.Display = 'iter';
				end
				
			otherwise
				error('unsupported optimizer');
		end
		
		% Load in specified parameters
		if ~isempty(input_params)
			spec_fields = fieldnames(input_params);
			for ii = 1:length(spec_fields)
				value = input_params.(spec_fields{ii});
				optim_params.(spec_fields{ii}) = value;
			end
		end	
	end

	function [train_inds,parsed_options,modvarargin] = parse_varargin( cellarray2parse, fields_to_remove, default_options  )
	% Usage: [train_inds parsed_options modvarargin] = parse_varargin( cellarray2parse, <fields_to_remove>, <default_options>  )
	% Parses the standard varargin into "train_inds" (if present) and a struct "parsed_options"
	% with options as seperate fields. Will also return a modified 'modvarargin' to pass to further
	% functions if 'fields_to_remove' is present. Finally, can pass in a struct of 'default_options' such
	% that this function will just overwrite the options that are set by varargin.
	%
	% INPUTS:
	%		cellarray2parse: standard varargin input to function as cell array 
	%		fields_to_remove: cell array of strings to not pass to modvarargin
	%		default_options: initial struct with pre-set values that varargin adds to
	% OUTPUTS:
	%		train_inds: pulls out vector (list of numbers) at beginning as common optional argument in most NIM function calls
	%		parsed_options: struct with fields corresponding to options of varargin
	%		modvarargin: cell array identical to varargin but with desired fields removed

		if nargin < 3
			default_options = [];
		end
		if nargin < 2
			fields_to_remove = {};
		end
		
		train_inds = [];
		parsed_options = default_options;
		modvarargin = {};
		if isempty(cellarray2parse)
			return
		end

		j = 1;      
		while j <= length(cellarray2parse)
			flag_name = cellarray2parse{j};      
			if ~ischar(flag_name) % if not a flag, it must be train_inds
				if ~isempty(flag_name)
					train_inds = flag_name;
					modvarargin{end+1} = train_inds; % implicitly add to modvarargin
				end
				j = j + 1; % only one argument here        
			else
				parsed_options.(flag_name) = cellarray2parse{j+1};
				if ~ismember(flag_name,fields_to_remove)
					modvarargin{end+1} = flag_name;
					modvarargin{end+1} = cellarray2parse{j+1};
				end
				j = j + 2;
			end			
		end
	end

	function arevalid = validate_parsed_options( parsed_struct, valid_list, return_error )
	% Usage: arevalid = validate_parsed_options( parsed_struct, valid_list, return_error )
	% Makes sure that fields in parsed_struct are valid options specified in 'valid_list' 
	% 
	% INPUTS:
	%   parsed_struct: output of parse_varargin: a struct with fields corresponding to inputed options
	%   valid list: cell array of strings saying valid options
	%   return_error: specifies whether to return error (default) or simply set 'arevalid' as false if invalid options
	% OUTPUTS:
	%   arevalid: Boolean: true if all fields in parsed_struct are on the valid_list
	
		if nargin < 3
			return_error = 1;
		end
		arevalid = 1;
		if isempty(parsed_struct)
			return
		end
		parsed_options = fields(parsed_struct);
		invalids = find(ismember(parsed_options, valid_list) == 0);
		if isempty(invalids)
			return
		end
		
		% Otherwise invalid elements of list
		for nn = 1:length(invalids)
			fprintf('\n  ''%s'' is not a valid option.\n', parsed_options{invalids(nn)} );
		end
		if return_error
			error( 'Invalid input arguments.' )
		else
			arevalid = 0;
		end
	end

	function percentiles = my_prctile( x, p )
	% Usage: percentiles = my_prctile( x, p )
	%
	% Calculate pth percentiles of x (p can be vector-valued).

		x = x(:);
		p = p(:);
		xlen = length(x);

		x = sort(x);
		un_ax = [0 100*(0.5:(xlen-0.5))./xlen 100]'; % uniform spacing from 0 to 100 of length xlen + 2
		x = [x(1); x; x(end)]; % make length xlen + 2
		percentiles = interp1q(un_ax,x,p); % find pth percentiles
	end
    
end
end