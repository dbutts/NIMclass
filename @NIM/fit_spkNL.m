function nim = fit_spkNL(nim, Robs, Xstims, varargin)
%         nim = nim.fit_spkNL(Robs, Xstims, <train_inds>, varargin)
%         Optimizes the parameters of the spkNL
%         INPUTS:
%            Robs: vector of response observations (e.g. spike counts)
%            Xstims: cell array of stimuli
%            <train_inds>: index values of data on which to fit the model [default to all indices in provided data]
%            optional flags:
%                ('gain_funs',gain_funs): matrix of multiplicative factors, one column for each subunit
%                ('optim_params',optim_params): struct of desired optimization parameters
%                ('silent',silent): boolean variable indicating whether to suppress the iterative optimization display
%                ('hold_const',hold_const): vector of parameter indices to hold constant
%        OUTPUTS:
%            nim: output model struct

%set defaults for optional inputs
gain_funs = []; %default has no gain_funs
train_inds = nan; %default nan means train on all data
silent = false; %default is show the optimization output
hold_const = []; %default is fit all spk NL params

option_list = {'gain_funs','silent','hold_const'}; %list of possible option strings

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
        j = j + 1;
    else
        switch lower(flag_name)
            case 'gain_funs'
                gain_funs = varargin{j+1};
            case 'optim_params'
                optim_params = varargin{j+1};
                assert(isstruct(optim_params),'optim_params must be a struct');
            case 'silent'
                silent = varargin{j+1};
                assert(ismember(silent,[0 1]),'silent must be 0 or 1');
            case 'hold_const'
                hold_const = varargin{j+1};
            otherwise
                error('Invalid input flag');
        end
        j = j + 2;
    end
end

if size(Robs,2) > size(Robs,1); Robs = Robs'; end; %make Robs a column vector
nim.check_inputs(Robs,Xstims,train_inds,gain_funs); %make sure input format is correct

if ~isnan(train_inds) %if specifying a subset of indices to train model params
    Robs = Robs(train_inds);
    for nn = 1:length(Xstims)
        Xstims{nn} = Xstims{nn}(train_inds,:); %grab the subset of indices for each stimulus element
    end
    if ~isempty(gain_funs); gain_funs = gain_funs(train_inds,:); end;
end
[~, ~, mod_internals] = nim.eval_model(Robs, Xstims,'gain_funs',gain_funs);
G = mod_internals.G;

init_params = [nim.spkNL.params nim.spkNL.theta]; %initialize parameters to fit (including the offset term theta)

%BOUND CONSTRAINTS
LB = -Inf*ones(size(init_params));
UB = Inf*ones(size(init_params));
if ismember(nim.spkNL.type,{'lin','rectpow','exp','softplus','logistic'})
    LB(1) = 0; %beta is non-negative
end
if ismember(nim.spkNL.type,{'softplus'})
    LB(2) = 0; %alpha is non-negative
end
if ismember(nim.spkNL.type,{'rectpow'})
    LB(2) = 1; %gamma >= 1 for convexity
end
%equality constraints
Aeq = []; Beq = [];
for i = 1:length(hold_const)
    Aeq = [Aeq; zeros(1,length(init_params))];
    Aeq(end,hold_const(i)) = 1;
    Beq = [Beq; init_params(hold_const(i))];
end

optimizer = 'fmincon';
optim_params = nim.set_optim_params(optimizer,optim_params,silent);
optim_params.GradObj = 'on';
opt_fun = @(K) internal_LL_spkNL(nim,K, Robs, G);
params = fmincon(opt_fun, init_params, [], [], Aeq, Beq, LB, UB, [], optim_params);
[~,penGrad] = opt_fun(params);
first_order_optim = max(abs(penGrad));
if first_order_optim > nim.opt_check_FO
    warning(sprintf('First-order optimality: %.3f, fit might not be converged!',first_order_optim));
end

nim.spkNL.params = params(1:end-1);
nim.spkNL.theta = params(end);

[LL,~,mod_internals,LL_data] = nim.eval_model(Robs,Xstims,'gain_funs',gain_funs);
nim = nim.set_subunit_scales(mod_internals.fgint); %update filter scales
cur_fit_details = struct('fit_type','spkNL','LL',LL,'filt_pen',LL_data.filt_pen,...
    'NL_pen',LL_data.NL_pen,'FO_optim',first_order_optim);
nim.fit_props = cur_fit_details;
nim.fit_hist = cat(1,nim.fit_hist,cur_fit_details);
end

%%
function [LL, LLgrad] = internal_LL_spkNL(nim,params, Robs, G)
%computes the LL and its gradient for given set of spkNL parameters

% ESTIMATE GENERATING FUNCTIONS (OVERALL AND INTERNAL)
nim.spkNL.params = params(1:end-1);
pred_rate = nim.apply_spkNL(G + params(end));

LL = nim.internal_LL(pred_rate,Robs); %compute LL
LL_deriv = nim.internal_LL_deriv(pred_rate,Robs);
spkNL_grad = nim.spkNL_param_grad(params,G);
LLgrad = sum(bsxfun(@times,spkNL_grad,LL_deriv));

% CONVERT TO NEGATIVE LLS AND NORMALIZE BY NSPKS
Nspks = sum(Robs);
LL = -LL/Nspks;
LLgrad = -LLgrad/Nspks;
end

