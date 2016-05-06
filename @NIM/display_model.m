	function  fig_handle = display_model( nim, varargin )
	% Usage: [] = nim.display_model( varargin )
	% Displays all parts of model as single plot (with multiple subplots)
	%
	% INPUTS:  
	%  Optional flags:
	%    'Xstims': to enter Xstims for calculation of G-distributions
	%    'Robs': to enter Robs, as required for G-distributions where there is a spike-history term
	%    'mod_outs': output of eval_model that gives required internal parameters in place of Xstims and Robs
	%    'colormap': to specify colormap for any image plots in filter displays
	%    'time_rev': time-reverse temporal plots
	%    'dt': enter to plot actual time versus time in binned units
	
		[~,parsed_options,modvarargin] = NIM.parse_varargin( varargin, {'Xstims','Robs','mod_outs'} );
		valid_list = {'Xstims','Robs','mod_outs','colormap','single','color','dt','time_rev','xt-separable','xt-spatial','sign','y_offset','no_axes_space','no_axes_time'};
		NIM.validate_parsed_options( parsed_options, valid_list );
		Nmods = length(nim.subunits);
		mod_outs = [];
		
		if isfield( parsed_options, 'Xstims' )
			if iscell(parsed_options.Xstims)
				Xstims = parsed_options.Xstims;
			else
				Xstims{1} = parsed_options.Xstims;
			end			
			Robs = [];
			if isfield( parsed_options, 'Robs' )  % Robs only makes sense to process when other 
				Robs = parsed_options.Robs;
			end
			[~,~,mod_outs] = nim.eval_model( Robs, Xstims );
		elseif isfield( parsed_options, 'mod_outs' )
			mod_outs = parsed_options.mod_outs;
		end
	
		extra_plots = [(nim.spk_hist.spkhstlen > 0) ~isempty(mod_outs)]; % will be spike-history or spkNL plot?
		
		if sum(extra_plots) == 0
			Nrows = Nmods;
			Ncols = 3;
		else
			% Then need extra column (and possibly extra row)
			Nrows = max([Nmods sum(extra_plots)]);
			Ncols = 4;
		end
		
		if nargout > 0
			fig_handle = figure;
		else
			figure;
		end
		% Plot Subunit info
		for nn = 1:Nmods
			nim.subunits(nn).display_filter( nim.stim_params(nim.subunits(nn).Xtarg).dims, [Nrows Ncols (nn-1)*Ncols+1], modvarargin{:} );
			subplot( Nrows, Ncols, (nn-1)*Ncols+3 );
			if isempty(mod_outs)
				nim.subunits(nn).display_NL();
			else
				nim.subunits(nn).display_NL( mod_outs.gint(:,nn) );
			end
		end
		
		% Plot spkNL
		if sum(extra_plots) == 0
			return
		end
		
		subplot( Nrows, Ncols, Ncols );
		if extra_plots(2) > 0
			nim.display_spkNL( mod_outs.G );
			title( 'Spk NL' )
			if extra_plots(1) > 0
				subplot( Nrows, Ncols, 2*Ncols );
			end
		end
		if extra_plots(1) > 0
			nim.display_spike_history();
		end
	end
	