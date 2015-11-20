function nim = reg_path( nim, Robs, Xs, Uindx, XVindx, varargin )
% Usage: nim = reg_path( nim, Robs, Xs, Uindx, XVindx, varargin )
%
% Old function
% [bestfit,L2best] = NMM_RegPath( fit0, Robs, Xs, Uindx, XVindx, targets, L2s, lambdaID, fitparams )
%
% Rudimentary function to fit filters or non-parametric nonlinearities for range of reg values. This function
% is not necessarily robust or tested outside of particular situations.
%
% lambda-ID refers to which lamdba to do regularization path: default = 'd2t'
% vargin of 'silent' = 1 will suppress reg_path output, and 2 will suppress except for reporting maximum

% Set default options
L2s = [];
lambdaID = 'd2t';
Nsubs = length(nim.subunits);
targets = 1:Nsubs;
silent = 0;

% Parse reg_path-specific input arguments
if (length(varargin) == 1) && iscell(varargin)
	varargin = varargin{1};
end

modvarargin{1} = Uindx;
modvarargin{2} = 'silent'; % to pass into subfunctions
modvarargin{3} = 1;

modcounter = 4; j = 1;
while j <= length(varargin)
    flag_name = varargin{j};
    switch lower(flag_name)
			case 'fit_subs'
				targets = varargin{j+1};
				assert(all(ismember(targets,1:Nsubs)),'invalid target subunits specified');
			case 'l2s'    
				L2s = varargin{j+1};  
			case 'silent'
				silent = varargin{j+1};
			case 'lambdaid'
				lambdaID = varargin{j+1}; 
			otherwise
				modvarargin{modcounter} = varargin{j};
				modvarargin{modcounter+1} = varargin{j+1};
				modcounter = modcounter + 2;
		end
		j = j + 2;
end

Nreg = length(L2s);

% Add targets to modvarargin
modvarargin{modcounter} = 'fit_subs';
modcounter = modcounter + 1;

for tar = targets
	if isempty(L2s)  
		%% Do order-of-mag reg first
		L2s = [0 0.1 1.0 10 100 1000 10000 1e5];
		if ~silent 
			fprintf( 'Order-of-magnitude L2 reg path: target = %d\n', tar )
		end		
		LLregs = zeros(length(L2s),1);
		for nn = 1:length(L2s)
			regfit = nim.set_reg_params( 'sub_inds', tar, lambdaID, L2s(nn) );
			
			% Only refit specific target
			modvarargin{modcounter} = tar;
			
			if strcmp( lambdaID, 'nld2' )
				regfit = regfit.fit_upstreamNLs( Robs, Xs, modvarargin );
			else
				regfit = regfit.fit_filters( Robs, Xs, modvarargin );
			end
			
			fitsaveM{nn} = regfit;
			[LL,~,~,LLdata] = regfit.eval_model( Robs, Xs, XVindx );
			LLregs(nn) = LL-LLdata.nullLL;
			if ~silent 
	      fprintf( '  %8.2f: %f\n', L2s(nn), LLregs(nn) )
			end
			
			if nn > 2
				if (LLregs(nn) < LLregs(nn-1)) && (LLregs(nn) < LLregs(nn-2))
					nn = length(L2s)+1;
				end
			end
		end
	
		[~,bestord] = sort(LLregs);
	  % Check to see if contiguous
		if abs(bestord(end)-bestord(end-1)) == 1
			loweredge = min(bestord(end-1:end));
		else  % otherwise go the right direction on the best
			if bestord(end) > bestord(end-1)
				loweredge = bestord(end)-1;
			else
				loweredge = bestord(end);
			end
		end
		mag = L2s(loweredge);
		LLbounds = LLregs(loweredge+[0 1]); 

		% Zoom in on best regularization
		if mag == 0
		 L2s = [0 0.01 0.02 0.04 0.1];
		else
			L2s = mag*[1 2 4 6 8 10];
		end
		Nreg = length(L2s);
		LLregs = zeros(Nreg,1);
		LLregs(1) = LLbounds(1);    LLregs(end) = LLbounds(2);
		fitsave{1} = fitsaveM{loweredge};
		fitsave{Nreg} = fitsaveM{loweredge+1};

		if ~silent
			fprintf( 'Zooming in on L2 reg path (%0.1f-%0.1f):\n', mag, mag*10 )
		end

		for nn = 2:(Nreg-1)
			regfit = nim.set_reg_params( 'sub_inds', tar, lambdaID, L2s(nn) );
			
			if strcmp( lambdaID, 'nld2' )
				regfit = regfit.fit_upstreamNLs( Robs, Xs, modvarargin );
			else
				regfit = regfit.fit_filters( Robs, Xs, modvarargin );
			end
			fitsave{nn} = regfit;

			[LL,~,~,LLdata] = regfit.eval_model( Robs, Xs, XVindx );
			LLregs(nn) = LL-LLdata.nullLL;
			if ~silent
				fprintf( '  %8.2f: %f\n', L2s(nn), LLregs(nn) )
			end
	    if nn > 2
		    if (LLregs(nn) < LLregs(nn-1)) && (LLregs(nn) < LLregs(nn-2))
			    nn = length(L2s)+1;
				end
			end
		end
  
	else
		
		%% Use L2 list estalished in function call
		if ~silent
			fprintf( 'L2 reg path (%d): target = %d', Nreg, tar )
		end
		LLregs = zeros(Nreg,1);

		for nn = 1:length(L2s)
			regfit = nim.set_reg_params( 'sub_inds', tar, lambdaID, L2s(nn) );
			
			if strcmp( lambdaID, 'nld2' )
				regfit = regfit.fit_upstreamNLs( Robs, Xs, modvarargin );
			else
				regfit = regfit.fit_filters( Robs, Xs, modvarargin );
			end
			fitsave{nn} = regfit;
			[LL,~,~,LLdata] = regfit.eval_model( Robs, Xs, XVindx );
			LLregs(nn) = LL-LLdata.nullLL;
			if ~silent
				fprintf( '  %8.2f: %f\n', L2s(nn), LLregs(nn) )
			end
		end
	end
	
	[~,bestnn] = max(LLregs);
	%L2best = L2s(bestnn);
	nim = fitsave{bestnn};
	if silent ~= 1
		fprintf( '    Subunit %d: Best reg = %0.2f\n', tar, L2s(bestnn) )
	end
	L2s = [];
end
end
