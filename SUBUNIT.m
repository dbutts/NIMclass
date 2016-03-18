classdef SUBUNIT
% Class implementing the subunits comprising an NIM model
%
% James McFarland, September 2015
% Modified NTLab and UMD, Jan 2016-

%% Properties
properties
	filtK;       % filter coefficients, [dx1] array where d is the dimensionality of the target stimulus
	NLtype;      % upstream nonlinearity type (string)
	NLparams;    % vector of 'shape' parameters associated with the upstream NL function (for parametric functions)
	NLnonpar;    % struct of settings and values for non-parametric fit (TBx TBy TBparams)
	NLoffset;    % scalar offset value added to filter output
	weight;      % subunit weight (typically +/- 1)
	Xtarg;       % index of stimulus the subunit filter acts on
	reg_lambdas; % struct of regularization hyperparameters
	Ksign_con;   % scalar defining any constraints on the filter coefs [-1 is negative con; +1 is positive con; 0 is no con]
end
	
properties (Hidden)
	allowed_subunitNLs = {'lin','quad','rectlin','rectpow','softplus','exp','nonpar'}; % set of subunit NL functions currently implemented
	TBy_deriv;   % internally stored derivative of tent-basis NL
	scale;       % SD of the subunit output derived from most-recent fit
end
    
	
%% METHODS
methods
        
	%% ********************** Constructor *************************************
	function subunit = SUBUNIT(init_filt, weight, NLtype, Xtarg, NLoffset, NLparams, Ksign_con)
	% Usage: subunit = SUBUNIT(init_filt, weight, NLtype, <NLparams>, <Ksign_con>)
	%   constructor for SUBUNIT class.
	%   INPUTS:
	%			init_filt: vector of initial filter coefs
	%     weight: scalar weighting associated with subunit (typically +/-1)
	%     NLtype: string specifying the type of upstream NL
	%     <Xtarg>: scalar index specifying which stimulus element this subunit acts on
	%     <NLoffset>: scalar term added to filter output
	%     <NLparams>: vector of 'shape' parameters for the upstream NL
	%     <Ksign_con>: constraint on filter coefs [-1 = negative; +1 = positive; 0 is no con]
	%   OUTPUTS: subunit: subunit object
        
		if nargin == 0 
			return %handle the no-input-argument case by returning a null model. This is important when initializing arrays of objects
		end
		
		if (nargin < 4 || isempty(Xtarg)); Xtarg = 1; end % default Xtarget is 1
		if (nargin < 5 || isempty(NLoffset)); NLoffset = 0; end; % default NLoffset is 0
		if nargin < 6; NLparams = []; end;
		if (nargin < 7 || isempty(Ksign_con)); Ksign_con = 0; end; % default no constraints on filter coefs
            
		assert(length(weight) == 1,'weight must be scalar!');
		assert(ischar(NLtype),'NLtype must be a string');
		assert(isscalar(NLoffset),'NLoffset must be a scalar');      
		subunit.filtK = init_filt;
		subunit.weight = weight;   
		if ~ismember(weight,[-1 1])       
			warning('Best to initialize subunit weights to be +/- 1');      
		end
		subunit.Xtarg = Xtarg;    
		subunit.NLtype = lower(NLtype);
      
		assert( ismember(subunit.NLtype,subunit.allowed_subunitNLs), 'invalid NLtype!' );
                  
		% if using an NLtype that has parameters, check that input      
		% parameter vector is the right size, or initialize to default
		% values      
		switch subunit.NLtype
			case {'lin','quad','rectlin','exp'} %these NLs dont have any shape parameters
				assert(isempty(NLparams),sprintf('%s NL type has no shape parameters',subunit.NLtype));       
			case 'softplus'        
				if isempty(NLparams)      
					NLparams = [1]; %defines beta in f(x) = log(1 + exp(beta*x))        
				else
					assert(length(NLparams) == 1,'invalid NLparams vector');    
				end	
			case 'rectpow'   
				if isempty(NLparams)          
					NLparams = [2]; %defines gamma in f(x) = x^gamma iff x >= 0          
				else
					assert(length(NLparams) == 1,'invalid NLparams vector');    
				end		
		end
		subunit.NLparams = NLparams;	
		subunit.NLnonpar.TBx = [];		
		subunit.NLnonpar.TBy = [];
		subunit.NLnonpar.TBparams = [];    
		subunit.NLoffset = NLoffset;    
		subunit.reg_lambdas = SUBUNIT.init_reg_lambdas();  
		subunit.Ksign_con = Ksign_con;	
		subunit.scale = 1;
	end
				

	%% ********************** Calculating outputs *************************************
        
	function sub_out = apply_NL( subunit, gen_signal )
  % Usage: sub_out = apply_NL( subunit, gen_signal )
	%
	% Applies subunit NL to the input generating signal
            
		switch subunit.NLtype 
			case 'lin' %f(x) = x 
				sub_out = gen_signal;
                    
			case 'quad' %f(x) = x^2
				sub_out = gen_signal.^2;
                    
			case 'rectlin' %f(x) = x iff x >= 0; else x = 0
				sub_out = gen_signal;
				sub_out(gen_signal < 0) = 0;
                    
			case 'rectpow' %f(x;gamma) = x^gamma iff x >= 0; else x = 0
				sub_out = gen_signal.^subunit.NLparams(1);         
				sub_out(gen_signal < 0) = 0;            

			case 'exp' %f(x;gamma) = exp(x)
				sub_out = exp(gen_signal);         
				sub_out(gen_signal < 0) = 0;            

			case 'softplus' %f(x;beta) = log(1 + exp(beta*x))
				max_g = 50; %to prevent numerical overflow
				gint = subunit.NLparams(1)*gen_signal; %beta*gen_signal            
				expg = exp(gint); 
				sub_out = log(1 + expg);
				sub_out(gint > max_g) = gint(gint > max_g);
                            
			case 'nonpar' %f(x) piecewise constant with knot points TBx and coefficients TBy [note: no offset]            
				sub_out = zeros(size(gen_signal));           
				% Data where X < TBx(1) are determined entirely by the first tent basis
				left_edge = find(gen_signal < subunit.NLnonpar.TBx(1));            
				sub_out(left_edge) = sub_out(left_edge) + subunit.NLnonpar.TBy(1);
				% similarly for the right edge  
				right_edge = find(gen_signal >= subunit.NLnonpar.TBx(end));            
				sub_out(right_edge) = sub_out(right_edge) + subunit.NLnonpar.TBy(end);
				slopes = diff(subunit.NLnonpar.TBy)./diff(subunit.NLnonpar.TBx);            
				for j = 1:length(subunit.NLnonpar.TBy)-1
					cur_set = find(gen_signal >= subunit.NLnonpar.TBx(j) & gen_signal < subunit.NLnonpar.TBx(j+1));           
					sub_out(cur_set) = sub_out(cur_set) + subunit.NLnonpar.TBy(j) + slopes(j)*(gen_signal(cur_set) - subunit.NLnonpar.TBx(j));
				end		
		end
	end
		
	function sub_deriv = apply_NL_deriv(subunit,gen_signal)
	% Usage: sub_deriv = apply_NL_deriv( subunit, gen_signal )
	%
	% Applies derivative of subunit NL to input gen_signal          
	
		switch subunit.NLtype
                
			case 'lin' %f'(x) = 1 (shouldnt ever need to use this within the optimization...)
				sub_deriv = ones(size(gen_signal));
            
			case 'quad' %f'(x) = 2x
				sub_deriv = 2*gen_signal;
                    
			case 'rectlin' %f'(x) = 1 iff x >= 0; else 0
				sub_deriv = gen_signal >= 0;
                    
			case 'rectpow' %f'(x) = gamma*x^(gamma-1) iff x > =0; else 0
				sub_deriv = subunit.NLparams(1)*gen_signal.^(subunit.NLparams(1)-1);
				sub_deriv(gen_signal < 0) = 0;

			case 'exp' %f'(x) = exp(x)
				sub_deriv = exp(gen_signal);

			case 'softplus' %f'(x) = beta*exp(beta*x)/(1 + exp(beta*x))
				max_g = 50; % to prevent numerical overflow
				gint = subunit.NLparams(1)*gen_signal; % beta*gen_signal
				sub_deriv = subunit.NLparams(1)*exp(gint)./(1 + exp(gint));
				sub_deriv(gint > max_g) = 1; % for large gint, derivative goes to 1
               
			case 'nonpar'
				ypts = subunit.TBy_deriv; xedges = subunit.NLnonpar.TBx;
				sub_deriv = zeros(length(gen_signal),1);
				for n = 1:length(xedges)-1
					sub_deriv((gen_signal >= xedges(n)) & (gen_signal < xedges(n+1))) = ypts(n);			
				end
		end
	end
		
	function NLgrad = NL_grad_param( subunit, x )
	% Usage: NLgrad = subunit.NL_grad_param( x )
	%
	% Calculates gradient of upstream NL wrt vector of parameters at input
	% value x. Note, parameter vector is of the form [NLparams NLoffset]
	
		NT = length(x);        
		NLgrad = zeros(NT,length(subunit.NLparams));
		
		switch subunit.NLtype
    
			case 'rectlin' %f(x;c) = x+c iff x >= -c
				NLgrad = (x >= -subunit.NLoffset); %df/dc
        
			case 'rectpow' %f(x;gamma,c) = (x+c)^gamma iff x >= -c; else 0
				NLgrad(:,1) = (x + 0).^subunit.NLparams(1) .* log(x + 0); %df/dgamma      
				NLgrad(:,2) = subunit.NLparams(1)*(x + 0).^(subunit.NLparams(1) - 1); %df/dc
				NLgrad(x < 0,:) = 0;

			case 'exp'
				NLgrad = exp(x+subunit.NLoffset);
				
			case 'softplus' %f(x) = log(1 + exp(beta*(x + c)):
				temp = exp(subunit.NLparams(1)*(x + subunit.NLoffset))./ (1 + exp(subunit.NLparams(1)*(x + subunit.NLoffset)));
				NLgrad(:,1) = temp.*(x + subunit.NLoffset); %df/dbeta
				NLgrad(:,2) = temp.*subunit.NLparams(1); %df/dc		
		end
	end

end

%% ********************** Display functions *************************************
methods
	
	function [] = display_filter( subunit, dims, varargin )
	% Usage: [] = subunit.display_filter( dims, <plot_location>, varargin )
	% Plots subunit filter in a 2-row, 1-column subplot
	%
	% INPUTS:
	%	  plot_location: 3-integer list = [Fig_rows Fig_col Loc] arguments to subplot. Default = [1 2 1]
	%	  optional arguments (varargin)
	%     'color': enter to specify color of non-image-plots (default is blue). This could also have dashes etc
	%     'colormap': choose colormap for 2-D plots. Default is 'gray'
	%	    'dt': enter if you want time axis scaled by dt
	%	    'time_rev': plot temporal filter reversed in time (zero lag on right)
	%	    'xt_rev': plot 2-D plots with time on x-axis
	%     'single': 1-D plots have only best latency/spatial position instead of all
	%	    'notitle': suppress title labeling subunit type
	%     'xt-separable': do not plot x-t plot, but rather separable x and seperable t
	%     'xt-spatial': for xt-plots (1-D space), plot spatial instead of temporal as second subplot
	
		assert((nargin > 1) && ~isempty(dims), 'Must enter filter dimensions.' )

		[plotloc,parsed_options,modvarargin] = NIM.parse_varargin( varargin, {'notitle','xt-spatial','xt-separable'} );
		if isempty(plotloc)
			plotloc = [1 2 1];
		end
		assert( plotloc(3) < prod(plotloc(1:2)), 'Invalid plot location.' )
		titleloc = plotloc(3);
		
		if prod(dims(2:3)) == 1

			% then 1-dimensional filter
			subplot( plotloc(1), plotloc(2), plotloc(3)+[0 1] ); hold on
			subunit.display_temporal_filter( dims, modvarargin{:} );
			titleloc = plotloc(3)+[0 1];
			
		elseif dims(3) == 1
			
			% then space-time plot in first subplot unless 'xt-separable'
			subplot( plotloc(1), plotloc(2), plotloc(3) )
			if isfield( parsed_options, 'xt-separable' ) || (dims(1) == 1)
				subunit.display_spatial_filter( dims, modvarargin{:} );
			else
				k = reshape( subunit.get_filtK(), dims(1:2) );
				imagesc( 1:dims(1),1:dims(2), k, max(abs(k(:)))*[-1 1] )
				if isfield(parsed_options,'colormap')
					colormap(parsed_options.colormap);
				else
					colormap('gray');
				end
			end
				
			% Plot temporal (default) or spatial in second subplot
			subplot( plotloc(1), plotloc(2), plotloc(3)+1 );
			if isfield( parsed_options, 'xt-spatial' )
				subunit.display_spatial_filter( dims, modvarargin{:} );
			else
				subunit.display_temporal_filter( dims, modvarargin{:} );
			end
			
		else
			
			% 3-d filter
			subplot( plotloc(1), plotloc(2), plotloc(3) )
			subunit.display_temporal_filter( dims, modvarargin{:} );
			subplot( plotloc(1), plotloc(2), plotloc(3)+1 )
			subunit.display_spatial_filter( dims, modvarargin{:} );
		end
		
		if ~isfield( parsed_options, 'notitle' )
			subplot( plotloc(1), plotloc(2), titleloc ) % to put title back on the first
			if strcmp(subunit.NLtype,'lin')
				title(sprintf('Lin'),'fontsize',10);
			elseif subunit.weight == 1
				title(sprintf('Exc'),'fontsize',10);
			elseif subunit.weight == -1				
				title(sprintf('Sup'),'fontsize',10);			
			end	
		end
	end
		
	function [] = display_temporal_filter( subunit, dims, varargin )
	% Usage: [] = subunit.display_temporal_filter( dims, varargin )
	%
	% Plots temporal elements of filter in a 2-row, 1-column subplot
	% INPUTS:
	%	  plot_location: 3-integer list = [Fig_rows Fig_col Loc] arguments to subplot. Default = [1 2 1]
	%	  optional arguments (varargin)
	%	    'color': enter to specify color of non-image-plots (default is blue). This could also have dashes etc
	%	    'dt': enter if you want time axis scaled by dt
	%	    'time_rev': plot temporal filter reversed in time (zero lag on right)
	%	    'single': plot single temporal function at best spatial position
	
		assert((nargin > 1) && ~isempty(dims), 'Must enter filter dimensions.' )
		if dims(1) == 1
			%warning( 'No temporal dimensions to plot.' )
			return
		end

		[~,parsed_options] = NIM.parse_varargin( varargin );
		if isfield(parsed_options,'color')
			clr = parsed_options.color;
		else
			clr = 'b';
		end
		
		% Time axis details
		NT = dims(1);
		if isfield(parsed_options,'dt')
			dt = parsed_options.dt;
		else
			dt = 1;
		end
		if isfield(parsed_options,'time_rev')
			ts = -dt*(0:NT-1);
		else
			ts = dt*(1:NT);
			if isfield(parsed_options,'dt')
				ts = ts-dt;  % if entering dt into time axis, shift to zero lag
			end
		end

		L = max(subunit.get_filtK())-min(subunit.get_filtK());
		
		if prod(dims(2:3)) == 1
			% then 1-dimensional filter
			k = subunit.get_filtK();
		else
			% then 2-d or 3-d filter
			k = reshape( subunit.get_filtK(), [dims(1) prod(dims(2:3))] );
			if isfield(parsed_options,'single')
				% then find best spatial
				[~,bestX] = max(std(k,1,1));
				k = k(:,bestX);
			end
		end
		plot( ts, k, clr, 'LineWidth',0.8 );
		hold on
		plot([ts(1) ts(end)],[0 0],'k--')
		
		axis([min(ts) max(ts) min(subunit.filtK)+L*[-0.1 1.1]])
		if isfield(parsed_options,'time_rev')
			box on
		else
			box off
		end
	end			

	function [] = display_spatial_filter( subunit, dims, varargin )
	% Usage: [] = subunit.display_spatial_filter( dims, varargin )
	%
	% Plots subunit filter in a 1-row, 1-column subplot
	% INPUTS:
	%	  plot_location: 3-integer list = [Fig_rows Fig_col Loc] arguments to subplot. Default = [1 1 1]
	%	  optional arguments (varargin)
	%	    'single': plot single temporal function at best spatial position
	%	    'color': enter to specify color of non-image-plots (default is blue). This could also have dashes etc
	%			'colormap': choose colormap for 2-D plots. Default is 'gray'

		if prod(dims(2:3)) == 1
			warning( 'No spatial dimensions to plot.' )
			return
		end
		
		assert( (nargin > 1) && ~isempty(dims), 'Must enter filter dimensions.' )

		[~,parsed_options] = NIM.parse_varargin( varargin );
		if isfield(parsed_options,'color')
			clr = parsed_options.color;
		else
			clr = 'b';
		end
		if isfield(parsed_options,'colormap')
			clrmap = parsed_options.colormap;
		else
			clrmap = 'gray';
		end
		
		k = reshape( subunit.get_filtK(), [dims(1) prod(dims(2:3))] );
		if dims(3) == 1

			% then 1-dimensional spatial filter
			if isfield(parsed_options,'single')
				% then find best spatial
				[~,bestT] = max(std(k,1,2));
				k = k(bestT,:);
			end
			plot( k, clr, 'LineWidth',0.8 );
			hold on
			plot([1 dims(2)],[0 0],'k--')
			L = max(k(:))-min(k(:));
			axis([1 dims(2) min(k(:))+L*[-0.1 1.1]])
		else
			
			% then 2-dimensional spatial filter
			[~,bestlat] = max(max(abs(k')));
			Kmax = max(abs(k(:)));

			imagesc( reshape(k(bestlat,:)/Kmax,dims(2:3)), [-1 1] )								
			colormap(clrmap)

		end
	end
	
	function [] = display_NL( subunit, varargin )
	% Usage: [] = subunit.display_filter( <gint>, varargin )
	%
	% Plots subunit upstream NL
	% INPUTS:
	%	  plot_location: 3-integer list = [Fig_rows Fig_col Loc] arguments to subplot. Default = [1 1 1]
	%	  optional arguments (varargin)
	%	    'sign': plot upside-down if suppressive
	%     'y_offset': y-axis offset
		
		n_hist_bins = 80; % internal parameter determining histogram resolution

		[gint,parsed_options] = NIM.parse_varargin( varargin );
		if nargin < 2
			gint = [];
		end
		
		if ~isempty(gint) % if computing distribution of filtered stim
			[gendist_y,gendist_x] = hist( gint, n_hist_bins );
          
			% Sometimes the gendistribution has a lot of zeros (dont want to screw up plot)
			[a,b] = sort(gendist_y);
			if a(end) > a(end-1)*1.5
				gendist_y(b(end)) = gendist_y(b(end-1))*1.5;         
			end	
		else
			gendist_x = linspace(-3,3,n_hist_bins); % otherwise, just pick an arbitrary x-axis to plot the NL  
		end
		if strcmp(subunit.NLtype,'nonpar')          
			cur_modx = subunit.NLnonpar.TBx; cur_mody = subunit.NLnonpar.TBy;        
		else
			cur_modx = gendist_x; cur_mody = subunit.apply_NL( cur_modx );
		end
		cur_xrange = cur_modx([1 end]);

		% Adjust nonlinearity as desired by fit-options
		if isfield(parsed_options,'sign') && (subunit.weight == -1)
			cur_mody = -cur_mody;
		end
		if isfield(parsed_options,'y_offset')
			cur_mody = cur_mody + parsed_options.y_offset;
		end

		if ~isempty(gint)          
			[ax,h1,~] = plotyy( cur_modx, cur_mody, gendist_x,gendist_y );          
			if strcmp(subunit.NLtype,'nonpar')            
				set(h1,'Marker','o');          
			end
			set(h1,'linewidth',1)         
			xlim(ax(1),cur_xrange)         
			xlim(ax(2),cur_xrange);          
			if all(cur_mody == 0)
				ylim(ax(1),[-1 1]);   
			else
				ylim(ax(1),[min(cur_mody) max(cur_mody)]);   
			end
			set(ax(2),'ytick',[])   
			yl = ylim();     
			line([0 0],yl,'color','k','linestyle','--');    
			ylabel(ax(1),'f_i(g)','fontsize',8);      
		else
			h = plot(cur_modx,cur_mody,'linewidth',1);         
			if strcmp(subunit.NLtype,'nonpar')            
				set(h,'Marker','o');         
			end
			xlim(cur_xrange)
			ylim([min(cur_mody) max(cur_mody)]);
			ylabel('f_i(g)','fontsize',8);  
		end
		box off
		xlabel('g')
		title('subunit NL')
	end
	
end

%% ********************** Getting Methods *********************************
methods
	
	function filtK = get_filtK( subunit )
	% Usage: filtK = get_filtK( subunit )
	%
	% Gets vector of filter coefs from the subunit
	
		filtK = subunit.filtK;      
	end
	
	function rsub = reinitialize_subunit( sub0, weight, NLtype, Xtarg, NLoffset )
	% Usage: rsub = reinitialize_subunit( sub0, weight, NLtype, Xtarg, NLoffset )
	%
	% Initializes subunit to have reset parameters (including random filter
  	        
		rsub = sub0;
		Npar = length(sub0.filtK);
		rsub.filtK = randn(Npar,1)/Npar; %initialize fitler coefs with gaussian noise	
		if (nargin < 5) || isempty(NLoffset)
			rsub.NLoffset = 0; %default NLoffset is 0	
		else
			rsub.NLoffset = NLoffset;
		end
		if (nargin > 3) && ~isempty(Xtarg)
			rsub.Xtarg = Xtarg;	
		end
		if (nargin > 2) && ~isempty(NLtype)
			rsub.NLtype = NLtype;	
		end
		if (nargin > 1) && ~isempty(weight)
			rsub.weight = weight;			
		end
	end
				
	function nrm = filter_norm( sub0, Xstims )
	% Usage: rsub = subunit.filter_norm( <Xstims> )
	% Returns magnitude of filter if Xstims is blank, otherwise magnitude of convolution

		if (nargin < 2) || isempty(Xstims)
			nrm = sqrt(sum(sub0.filtK.^2));
		else
			if ~iscell(Xstims)
				nrm = abs(sub0.weight) * std( Xstims * sub0.filtK );
			else
				nrm = abs(sub0.weight) * std( Xstims{sub0.Xtarg} * sub0.filtK );
			end
		end
	end

end

%% HIDDEN METHODS
methods (Hidden)      
        
	function fprime = get_TB_derivative( subunit )
	% Usage: fprime = subunit.get_TB_derivative()
	% Calculate the derivative of the piecewise linear function wrt x
	
		if ~strcmp(subunit.NLtype,'nonpar')
			fprime = [];
			return		
		end
		
		fprime = zeros(1,length(subunit.NLnonpar.TBx)-1);
		for n = 1:length(fprime)
			fprime(n) = (subunit.NLnonpar.TBy(n+1)-subunit.NLnonpar.TBy(n))/(subunit.NLnonpar.TBx(n+1)-subunit.NLnonpar.TBx(n));    
		end	
	end
	
	function gout = tb_rep( subunit, gin )
	% Usage: gout = tb_rep( subunit, gin )
	% Projects the input signal gin onto the tent-basis functions associated with this subunit
            
		n_tbs =length(subunit.NLnonpar.TBx); %number of tent-basis functions          
		gout = zeros(length(gin),n_tbs);
		gout(:,1) = SUBUNIT.get_tentbasis_output(gin,subunit.NLnonpar.TBx(1),[-Inf subunit.NLnonpar.TBx(2)]);       
		gout(:,end) = SUBUNIT.get_tentbasis_output(gin,subunit.NLnonpar.TBx(end),[subunit.NLnonpar.TBx(end-1) Inf]);      
		for n = 2:n_tbs-1     
			gout(:,n) = SUBUNIT.get_tentbasis_output(gin, subunit.NLnonpar.TBx(n), [subunit.NLnonpar.TBx(n-1) subunit.NLnonpar.TBx(n+1)] );       
		end
	end
	
end

%% STATIC METHODS   
methods (Static)

	function tent_out = get_tentbasis_output( gin, tent_cent, tent_edges )
	% Usage: tent_out = get_tentbasis_output( gin, tent_cent, tent_edges )          
	%          
	% Takes an input vector and passes it through the tent basis function
	% specified by center location tent_cent and the 2-element vector tent_edges = [left_edge right_edge]
	% specifying the tent bases 'edges'
            
		tent_out = zeros(size(gin)); %initialize NL processed stimulus
            
		% for left side 
		if ~isinf(tent_edges(1)) % if there is a left boundary          
			cur_set = (gin > tent_edges(1)) & (gin <= tent_cent); % find all points left of center and right of boundary     
			tent_out(cur_set) = 1-(tent_cent-gin(cur_set))/(tent_cent - tent_edges(1)); % contribution of this basis function to the processed stimulus
		else
			cur_set = gin <= tent_cent;  
			tent_out(cur_set) = 1;
		end
		
		% for right side      
		if ~isinf(tent_edges(2)) % if there is a left boundary          
			cur_set = (gin >= tent_cent) & (gin < tent_edges(2)); % find all points left of center and right of boundary       
			tent_out(cur_set) = 1-(gin(cur_set)-tent_cent)/(tent_edges(2)-tent_cent); % contribution of this basis function to the processed stimulus
		else
			cur_set = gin >= tent_cent;      
			tent_out(cur_set) = 1;
		end	
	end
	
	function reg_lambdas = init_reg_lambdas()
  % Usage: reg_lambdas = init_reg_lambdas()
	% Creates reg_lambdas struct and sets default values to 0
     
		reg_lambdas.nld2 = 0; %second derivative of tent basis coefs
		reg_lambdas.d2xt = 0; %spatiotemporal laplacian
		reg_lambdas.d2x = 0; %2nd spatial deriv
		reg_lambdas.d2t = 0; %2nd temporal deriv
		reg_lambdas.l2 = 0; %L2 on filter coefs
		reg_lambdas.l1 = 0; %L1 on filter coefs
	end
	
end

end  % (classdef)

