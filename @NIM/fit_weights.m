function nim = fit_weights(nim, Robs, Xstims, varargin)
%         nim = nim.fit_weights(Robs, Xstims, <train_inds>, varargin)
%         estimate linear weights on each subunit
%         INPUTS:
%            Robs: vector of response observations (e.g. spike counts)
%            Xstims: cell array of stimuli
%            <train_inds>: index values of data on which to fit the model [default to all indices in provided data]
%            optional flags:
%                ('fit_subs',fit_subs): set of subunits whos filters we want to optimize [default is all]
%                ('gain_funs',gain_funs): matrix of multiplicative factors, one column for each subunit
%                ('optim_params',optim_params): struct of desired optimization parameters, can also
%                   be used to override any of the default values for other optional inputs
%                ('silent',silent): boolean variable indicating whether to suppress the iterative optimization display
%                ('lambda_L1',lambda_L1): L1 penalty on model weights
%                ('fit_spk_hist',fit_spk_hist): boolean indicating whether to hold the spk NL filter constant
%         OUTPUTS:
%            new nim object with optimized subunit filters
%
Nsubs = length(nim.subunits); %number of subunits

%set defaults for optional inputs
fit_subs = 1:Nsubs; %defualt to fitting all subunits (plus -1 for spkHist filter)
gain_funs = []; %default has no gain_funs
train_inds = nan; %default nan means train on all data
fit_spk_hist = nim.spk_hist.spkhstlen > 0; %default is to fit the spkNL filter if it exists
fit_offsets = false(1,Nsubs); %default is NOT to fit the offset terms
lambda_L1 = 0; %default is no L1 penalty
silent = false; %default is to display optimization output
option_list = {'subs','gain_funs','silent','fit_spk_hist','lambda_l1'}; %list of possible option strings

% To unwrap varargin if passed as a cell-array
if ~isempty(varargin)
	if iscell(varargin) && iscell(varargin{1})
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
    flag_name = varargin{j};
    if ~ischar(flag_name)%if not a flag, it must be train_inds
        train_inds = flag_name;
        j = j + 1; %only one argument here
    else
        switch lower(flag_name)
            case 'subs'
                fit_subs = varargin{j+1};
                assert(all(ismember(fit_subs,1:Nsubs)),'invalid target subunits specified');
            case 'gain_funs'
                gain_funs = varargin{j+1};
            case 'silent'
                silent = varargin{j+1};
                assert(ismember(silent,[0 1]),'silent must be 0 or 1');
            case 'fit_spk_hist'
                fit_spk_hist = varargin{j+1};
                assert(ismember(fit_spk_hist,[0 1]),'fit_spk_hist must be 0 or 1');
            case 'lambda_l1'
                lambda_L1 = varargin{j+1};
                assert(lambda_L1 >= 0,'weight_L1 must be non_negative');
						case 'fit_offsets'
								% do nothing
            otherwise
                error('Invalid input flag');
        end
        j = j + 2;
    end
end
if length(fit_subs) < Nsubs && length(fit_offsets) == Nsubs
    fit_offsets = fit_offsets(fit_subs); %if only fitting subset of filters, set fit_offsets accordingly
end
mod_NL_types = {nim.subunits(fit_subs).NLtype}; %NL types for each targeted subunit
if any(strcmp(mod_NL_types(fit_offsets),'lin'))
    fprintf('Cant fit thresholds for linear subunits, ignoring these\n');
    fit_offsets(strcmp(mod_NL_types(fit_offsets),'lin')) = false;
end
if size(Robs,2) > size(Robs,1); Robs = Robs'; end; %make Robs a column vector
nim.check_inputs(Robs,Xstims,train_inds,gain_funs); %make sure input format is correct

Nfit_subs = length(fit_subs); %number of targeted subunits
non_fit_subs = setdiff([1:Nsubs],fit_subs); %elements of the model held constant
spkhstlen = nim.spk_hist.spkhstlen; %length of spike history filter
if fit_spk_hist; assert(spkhstlen > 0,'no spike history term initialized!'); end;
if spkhstlen > 0 % create spike history Xmat IF NEEDED
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

% PARSE INITIAL PARAMETERS
init_params = [nim.subunits(fit_subs).weight]';
lambda_L1 = ones(size(init_params))*lambda_L1;
lambda_L1 = lambda_L1/sum(Robs); % since we are dealing with LL/spk
Nfit_filt_params = length(init_params); %number of filter coefficients in param vector
% Add in spike history coefs
if fit_spk_hist
    init_params = [init_params; nim.spk_hist.coefs];
    lambda_L1 = [lambda_L1; zeros(size(nim.spk_hist.coefs))];
end

init_params = [init_params; nim.spkNL.theta]; % add constant offset
lambda_L1 = [lambda_L1; 0];
[nontarg_g] = nim.process_stimulus(Xstims,non_fit_subs,gain_funs);
if ~fit_spk_hist && spkhstlen > 0 %add in spike history filter output, if we're not fitting it
    nontarg_g = nontarg_g + Xspkhst*nim.spk_hist.coefs(:);
end
[~, targ_outs] = nim.process_stimulus(Xstims,fit_subs,gain_funs);

% IDENTIFY ANY CONSTRAINTS 
use_con = 0;
LB = -Inf*ones(size(init_params));
UB = Inf*ones(size(init_params));
if fit_spk_hist %if optimizing spk history term
    %negative constraint on spk history coefs
    if nim.spk_hist.negCon
        spkhist_inds = Nfit_filt_params + (1:spkhstlen);
        UB(spkhist_inds) = 0;
        use_con = 1;
    end
end

fit_opts = struct('fit_spk_hist', fit_spk_hist, 'fit_subs', fit_subs); %put any additional fitting options into this struct
%the function we want to optimize
opt_fun = @(K) internal_LL_weights(nim,K,Robs,targ_outs,Xspkhst,nontarg_g,gain_funs,fit_opts);

%determine which optimizer were going to use
if max(lambda_L1) > 0
    assert(~use_con,'Can use L1 penalty with constraints');
    assert(exist('L1General2_PSSas','file') == 2,'Need Mark Schmidts optimization tools installed to use L1');
    optimizer = 'L1General_PSSas';
else
    if ~use_con %if there are no constraints
        if exist('minFunc','file') == 2
            optimizer = 'minFunc';
        else
            optimizer = 'fminunc';
        end
    else
        if exist('minConf_TMP','file')==2
            optimizer = 'minConf_TMP';
        else
            optimizer = 'fmincon';
        end
    end
end
optim_params = nim.set_optim_params(optimizer,optim_params,silent);
if ~silent; fprintf('Running optimization using %s\n\n',optimizer); end;

switch optimizer %run optimization
    case 'L1General_PSSas'
        [params] = L1General2_PSSas(opt_fun,init_params,lambda_L1,optim_params);
    case 'minFunc'
        [params] = minFunc(opt_fun, init_params, optim_params);
    case 'fminunc'
        [params] = fminunc(opt_fun, init_params, optim_params);
    case 'minConf_TMP'
        [params] = minConf_TMP(opt_fun, init_params, LB, UB, optim_params);
    case 'fmincon'
        [params] = fmincon(opt_fun, init_params, [], [], [], [], LB, UB, [], optim_params);
end
[~,penGrad] = opt_fun(params);
first_order_optim = max(abs(penGrad));
if first_order_optim > nim.opt_check_FO
    warning(sprintf('First-order optimality: %.3f, fit might not be converged!',first_order_optim));
end

% PARSE MODEL FIT
nim.spkNL.theta = params(end); %set new offset parameter
if fit_spk_hist
    nim.spk_hist.coefs = params(Nfit_filt_params + (1:spkhstlen));
end
for ii = 1:Nfit_subs
    nim.subunits(fit_subs(ii)).weight = params(ii); %assign new filter values
end
[LL,~,mod_internals,LL_data] = nim.eval_model(Robs,Xstims,'gain_funs',gain_funs);
nim = nim.set_subunit_scales(mod_internals.fgint); %update filter scales
cur_fit_details = struct('fit_type','filter','LL',LL,'filt_pen',LL_data.filt_pen,...
    'NL_pen',LL_data.NL_pen,'FO_optim',first_order_optim);
nim.fit_props = cur_fit_details; %store details of this fit
nim.fit_history = cat(1,nim.fit_history,cur_fit_details);
end

%%
function [penLL, penLLgrad] = internal_LL_weights(nim,params,Robs,targ_outs,Xspkhst,nontarg_g,gain_funs,fit_opts)
%computes the penalized LL and its gradient wrt the linear weights on a set of target subunits

fit_subs = fit_opts.fit_subs;
Nfit_subs = length(fit_subs); %number of targeted subs

% USEFUL VALUES
theta = params(end); % overall model offset
G = theta + nontarg_g; % initialize overall generating function G with the offset term and the contribution from nontarget subs
G = G + targ_outs*params(1:Nfit_subs);

% Add contribution from spike history filter
if fit_opts.fit_spk_hist
    G = G + Xspkhst*params(Nfit_subs + (1:nim.spk_hist.spkhstlen));
end

pred_rate = nim.apply_spkNL(G);
[penLL,norm_fact] = nim.internal_LL(pred_rate,Robs); %compute LL

%residual = LL'[r].*F'[g]
residual = nim.internal_LL_deriv(pred_rate,Robs) .* nim.apply_spkNL_deriv(G,pred_rate <= nim.min_pred_rate);

penLLgrad = zeros(length(params),1); %initialize LL gradient
penLLgrad(end) = sum(residual);      %Calculate derivatives with respect to constant term (theta)

% Calculate derivative with respect to spk history filter
if fit_opts.fit_spk_hist
    penLLgrad(NKtot + length(offset_inds) + (1:nim.spk_hist.spkhstlen)) = residual'*Xspkhst;
end
if isempty(gain_funs)
    penLLgrad(1:Nfit_subs) = residual'* targ_outs;
else
    penLLgrad(1:Nfit_subs) = (gain_funs.*residual)'* targ_outs;
end

% CONVERT TO NEGATIVE LLS AND NORMALIZE BY NSPKS
penLL = -penLL/norm_fact;
penLLgrad = -penLLgrad/norm_fact;

end

