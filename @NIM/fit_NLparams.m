function nim = fit_NLparams(nim, Robs, Xstims, varargin)
%         nim = nim.fit_NLparams(Robs, Xstims, <train_inds>, varargin)
%         Optimizes any parameters of specified (parametric) upstream NL functions (including offsets)
%         INPUTS:
%            Robs: vector of response observations (e.g. spike counts)
%            Xstims: cell array of stimuli
%            <train_inds>: index values of data on which to fit the model [default to all indices in provided data]
%            optional flags:
%                ('subs',fit_subs): set of subunits whos filters we want to optimize [default is all]
%                ('gain_funs',gain_funs): matrix of multiplicative factors, one column for each subunit
%                ('optim_params',optim_params): struct of desired optimization parameters
%                ('silent',silent): boolean variable indicating whether to suppress the iterative optimization display
%                ('fit_spk_hist',fit_spk_hist): boolean indicating whether to fit the spk NL filter constant
%        OUTPUTS:
%            nim: output model struct

Nsubs = length(nim.subunits); %number of subunits

%set defaults for optional inputs
fit_subs = 1:Nsubs; %defualt to fitting all subunits that we can
gain_funs = []; %default has no gain_funs
train_inds = nan; %default nan means train on all data
silent = false; %default is show the optimization output
fit_spk_hist = nim.spk_hist.spkhstlen > 0; %default is fit the spkNL filter if it exists
option_list = {'subs','gain_funs','silent','fit_spk_hist'}; %list of possible option strings

% To unwrap varargin if passed as a cell-array
if ~isempty(varargin)
	if iscell(varargin)
		varargin = varargin{1};
	end
end

%over-ride any defaults with user-specified values
OP_loc = find(strcmp(varargin,'optim_params')); %find if optim_params is provided as input
if ~isempty(OP_loc)
    optim_params = varargin{OP_loc+1};
    varargin(OP_loc:(OP_loc+1)) = [];
    OP_fields = lower(fieldnames(optim_params));
    for ii = 1:length(OP_fields) %loop over fields of optim_params
        if ismember(OP_fields{ii},option_list); %if the field is a valid input option, over-ride the default
            eval(sprintf('%s = optim_params.(''%s'');',OP_fields{ii},OP_fields{ii}));
            optim_params = rmfield(optim_params,OP_fields{ii}); %and remove the field from optim_params, so that the remaining fields are options for the optimizer
        end
    end
else
    optim_params = [];
end

%now parse explicit optional input args
j = 1;
while j <= length(varargin)
    flag_name = varargin{j}; %if not a flag, it must be train_inds
    if ~ischar(flag_name)
        train_inds = flag_name;
        j = j + 1; %there's just one arg here
    else
        switch lower(flag_name)
            case 'subs'
                fit_subs = varargin{j+1};
                assert(all(ismember(fit_subs,1:Nsubs)),'specified target doesnt exist');
            case 'gain_funs'
                gain_funs = varargin{j+1};
            case 'optim_params'
                optim_params = varargin{j+1};
                assert(isstruct(optim_params),'optim_params must be a struct');
            case 'silent'
                silent = varargin{j+1};
                assert(ismember(silent,[0 1]),'silent must be 0 or 1');
            case 'fit_spk_hist'
                fit_spk_hist = varargin{j+1};
                assert(ismember(fit_spk_hist,[0 1]),'fit_spk_hist must be 0 or 1');
            otherwise
                error('Invalid input flag');
        end
        j = j + 2;
    end
end

if size(Robs,2) > size(Robs,1); Robs = Robs'; end; %make Robs a column vector
nim.check_inputs(Robs,Xstims,train_inds,gain_funs); %make sure input format is correct

spkhstlen = nim.spk_hist.spkhstlen; %length of spike history filter
if fit_spk_hist; assert(spkhstlen > 0,'no spike history term initialized!'); end;
if fit_spk_hist
    Xspkhst = create_spkhist_Xmat( Robs, nim.spk_hist.bin_edges);
else
    Xspkhst = [];
end
if ~isnan(train_inds) %if specifying a subset of indices to train model params
    for nn = 1:length(Xstims)
        Xstims{nn} = Xstims{nn}(train_inds,:); %grab the subset of indices for each stimulus element
    end
    Robs = Robs(train_inds);
    if ~isempty(Xspkhst); Xspkhst = Xspkhst(train_inds,:); end;
    if ~isempty(gain_funs); gain_funs = gain_funs(train_inds,:); end;
end

upstream_NL_types = nim.get_NLtypes(fit_subs);
excluded_subunits = ismember(upstream_NL_types,{'lin','nonpar'}); %target subunits that have NL types with no parameters
fit_subs(excluded_subunits) = []; %treat these as fixed subuntis
Nfit_subs = length(fit_subs); %number of targeted subunits
if Nfit_subs == 0
    warning('No subunits to fit!');
    return
end
non_fit_subs = setdiff(1:Nsubs,fit_subs); %elements of the model held constant

nontarg_g = nim.process_stimulus(Xstims,non_fit_subs,gain_funs); %get output of nontarget subunits
if ~fit_spk_hist && spkhstlen > 0 %add in spike history filter output, if we're not fitting it
    nontarg_g = nontarg_g + Xspkhst*nim.spk_hist.coefs(:);
end

[~,~,par_gint] = nim.process_stimulus(Xstims,fit_subs,gain_funs); %get the filter outputs of all target subunits with parametric NLs
NLoffsets = [nim.subunits(fit_subs).NLoffset];
par_gint = bsxfun(@minus,par_gint,NLoffsets); %subtract out offsets from filter outputs as we're going to fit these

% CREATE INITIAL PARAMETER VECTOR
% Compute initial fit parameters
init_params = [];
par_cnt = 0; %counts number of parametric NL parameters were estimating
for imod = fit_subs
    init_params = [init_params; nim.subunits(imod).NLparams; nim.subunits(imod).NLoffset]; %compile parametric NL params
    par_cnt = par_cnt + length(nim.subunits(imod).NLparams)+1;
end

% Add in spike history coefs
if fit_spk_hist
    init_params = [init_params; nim.spk_hist.coefs];
end

init_params = [init_params; nim.spkNL.theta]; %add constant offset

% PROCESS CONSTRAINTS
use_con = 0;
[LB,UB] = deal(nan(size(init_params)));
Aeq = []; beq = [];% initialize constraint parameters
% Check for spike history coef constraints
if fit_spk_hist
    % negative constraint on spk history coefs
    if nim.spk_hist.negCon
        spkhist_inds = par_cnt + (1:spkhstlen);
        LB(spkhist_inds) = -Inf;
        UB(spkhist_inds) = 0;
        use_con = 1;
    end
end
par_cnt = 0;
param_inds = cell(Nfit_subs,1);
for ii = 1:Nfit_subs
    cur_npars = length(nim.subunits(fit_subs(ii)).NLparams)+1;
    param_inds{ii} = par_cnt + (1:cur_npars);
    par_cnt = par_cnt + cur_npars;
end
if any(~isnan(LB) | ~isnan(UB)) %check if there are any bound constraints
   LB(isnan(LB)) = -Inf; UB(isnan(UB)) = Inf; %set all unconstrained parameters to +/- Inf
else
    LB = []; UB = []; %if no bound constraints, set these to empty
end

if ~use_con %if there are no constraints
    if exist('minFunc','file') == 2
        optimizer = 'minFunc';
    else
        optimizer = 'fminunc';
    end
else
    optimizer = 'fmincon';
end
optim_params = nim.set_optim_params(optimizer,optim_params,silent);
if ~silent; fprintf('Running optimization using %s\n\n',optimizer); end;

fit_opts = struct('fit_spk_hist', fit_spk_hist, 'fit_subs',fit_subs); %put any additional fitting options into this struct
fit_opts.param_inds = param_inds; %have to add this field separately

opt_fun = @(K) internal_LL_NLs(nim,K, Robs, par_gint, Xspkhst,nontarg_g,gain_funs,fit_opts);

switch optimizer %run optimization
    case 'L1General2_PSSas'
        [params] = L1General2_PSSas(opt_fun,init_params,lambda_L1,optim_params);
    case 'minFunc'
        [params] = minFunc(opt_fun, init_params, optim_params);
    case 'fminunc'
        [params] = fminunc(opt_fun, init_params, optim_params);
    case 'minConf_TMP'
        [params] = minConf_TMP(opt_fun, init_params, LB, UB, optim_params);
    case 'fmincon'
        [params] = fmincon(opt_fun, init_params, A, b, Aeq, beq, LB, UB, [], optim_params);
end
[~,penGrad] = opt_fun(params);
first_order_optim = max(abs(penGrad));
if first_order_optim > nim.opt_check_FO
    warning(sprintf('First-order optimality %.3f, fit might not be converged!',first_order_optim));
end
nim.spkNL.theta = params(end);
for ii = 1:Nfit_subs
    cur_params = params(param_inds{ii});
    nim.subunits(fit_subs(ii)).NLparams = cur_params(1:end-1)';
    nim.subunits(fit_subs(ii)).NLoffset = cur_params(end);
end
if fit_spk_hist
    nim.spk_hist.coefs = params( par_cnt + (1:spkhstlen));
end

[LL,~,mod_internals,LL_data] = nim.eval_model(Robs,Xstims,'gain_funs',gain_funs);
nim = nim.set_subunit_scales(mod_internals.fgint); %update filter scales
cur_fit_details = struct('fit_type','upstream_NLs','LL',LL,'filt_pen',LL_data.filt_pen,...
    'NL_pen',LL_data.NL_pen,'FO_optim',first_order_optim);
nim.fit_props = cur_fit_details;
nim.fit_history = cat(1,nim.fit_history,cur_fit_details);
end

%%
function [penLL, penLLgrad] = internal_LL_NLs(nim,params, Robs, par_gint, Xspkhst, nontarg_g, gain_funs,fit_opts)
%computes the LL and its gradient for a given set of upstream NL parameters

% Useful params
fit_subs = fit_opts.fit_subs; 
param_inds = fit_opts.param_inds;
n_par_pars = sum(cellfun(@(x) length(x),param_inds)); %total number of parametric NL params
Nfit_subs = length(fit_subs);
spkhstlen = nim.spk_hist.spkhstlen;
mod_weights = [nim.subunits(fit_subs).weight]';

% ESTIMATE GENERATING FUNCTIONS (OVERALL AND INTERNAL)
theta = params(end); %offset
G = theta + nontarg_g;

%Add contributions from target subunits with parametric NLs
fgint = zeros(size(par_gint));
for ii = 1:Nfit_subs
    cur_params = params(param_inds{ii});
    par_gint(:,ii) = par_gint(:,ii) + cur_params(end); %add current estimate of offset to filter output
    nim.subunits(fit_subs(ii)).NLparams = cur_params(1:end-1); %set NL parameters to new values
    fgint(:,ii) = nim.subunits(fit_subs(ii)).apply_NL(par_gint(:,ii));
end
% Multiply by weight (and multiplier, if appl) and add to generating function
if isempty(gain_funs)
    G = G + fgint*mod_weights;
else
    G = G + (fgint.*gain_funs(:,fit_subs))*mod_weights;
end

%add contribution from spike history filter
if fit_opts.fit_spk_hist
    G = G + Xspkhst*params(n_par_pars + (1:spkhstlen));
end

pred_rate = nim.apply_spkNL(G);
[penLL,norm_fact] = nim.internal_LL(pred_rate,Robs); %compute LL
%residual = LL'[r].*F'[g]
residual = nim.internal_LL_deriv(pred_rate,Robs) .* nim.apply_spkNL_deriv(G, pred_rate < nim.min_pred_rate);

penLLgrad = zeros(length(params),1); %initialize LL gradient
for ii = 1:Nfit_subs
    param_grad = nim.subunits(fit_subs(ii)).NL_grad_param(par_gint(:,ii));
    penLLgrad(param_inds{ii}) = residual'*param_grad * mod_weights(ii);
end
penLLgrad(end) = sum(residual);% Calculate derivatives with respect to constant term (theta)

% Calculate derivative with respect to spk history filter
if fit_opts.fit_spk_hist
    penLLgrad(n_par_pars+(1:spkhstlen)) = residual'*Xspkhst;
end

% CONVERT TO NEGATIVE LLS AND NORMALIZE BY NSPKS
penLL = -penLL/norm_fact;
penLLgrad = -penLLgrad/norm_fact;
end

