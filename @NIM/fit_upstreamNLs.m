function nim = fit_upstreamNLs(nim, Robs, Xstims, varargin)
%         nim = nim.fit_upstreamNLs(Robs, Xstims, <train_inds>, varargin)
%         Optimizes the upstream NLs (in terms of tent-basis functions)
%         INPUTS:
%            Robs: vector of response observations (e.g. spike counts)
%            Xstims: cell array of stimuli
%            <train_inds>: index values of data on which to fit the model [default to all indices in provided data]
%            optional flags:
%                ('sub_inds',sub_inds): set of subunits whos filters we want to optimize [default is all]
%                ('gain_funs',gain_funs): matrix of multiplicative factors, one column for each subunit
%                ('optim_params',optim_params): struct of desired optimization parameters
%                ('silent',silent): boolean variable indicating whether to suppress the iterative optimization display
%                ('hold_spkhist',hold_spkhist): boolean indicating whether to hold the spk NL filter constant
%                ('no_rescaling',no_rescaling): boolean indicating if you dont want to rescale the NLs after fitting
%        OUTPUTS:
%            nim: output model struct

Nsubs = length(nim.subunits); %number of subunits
NT = length(Robs); %number of time points

% PROCESS INPUTS
% poss_targets = find(strcmp(nim.get_NLtypes,'nonpar'))'; %set of subunits with nonpar NLs
fit_subs = 1:Nsubs; %defualt to fitting all subunits that we can
gain_funs = []; %default has no gain_funs
train_inds = nan; %default nan means train on all data
optim_params = []; %default has no user-specified optimization parameters
silent = false; %default is show the optimization output
fit_spk_hist = nim.spk_hist.spkhstlen > 0; %default is fit the spkNL filter if it exists
rescale_NLs = true; %default is to rescale the y-axis of NLs after estimation

j = 1;
while j <= length(varargin)
    flag_name = varargin{j}; %if not a flag, it must be train_inds
    if ~ischar(flag_name)
        train_inds = flag_name;
        j = j + 1; %there's just one arg here
    else
        switch lower(flag_name)
            case 'sub_inds'
                fit_subs = varargin{j+1};
                %                 assert(all(ismember(fit_subs,[poss_targets])),'specified target doesnt have non-parametric NL, or doesnt exist');
                assert(all(ismember(fit_subs,1:Nsubs)),'specified target doesnt exist');
            case 'gain_funs'
                gain_funs = varargin{j+1};
            case 'optim_params'
                optim_params = varargin{j+1};
                assert(isstruct(optim_params),'optim_params must be a struct');
            case 'silent'
                silent = varargin{j+1};
                assert(ismember(silent,[0 1]),'silent must be 0 or 1');
            case 'no_rescaling'
                rescale_NLs = varargin{j+1};
                assert(ismember(rescale_NLs,[0 1]),'rescale_NLs must be 0 or 1');
            case 'hold_spkhist'
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
no_params = ismember(upstream_NL_types,{'lin','quad'}); %target subunits that have NL types with no parameters
fit_subs(no_params) = []; %treat these as fixed subuntis
Nfit_subs = length(fit_subs); %number of targeted subunits
if Nfit_subs == 0
    warning('No subunits to fit!');
    return
end
non_fit_subs = setdiff(1:Nsubs,fit_subs); %elements of the model held constant

nonpar_subs = find(strcmp(nim.get_NLtypes(fit_subs),'nonpar')); %subunits with nonparametric upstream NLs
par_subs = find(~strcmp(nim.get_NLtypes(fit_subs),'nonpar')); %subunits with parametric upstream NLs

n_TBs = arrayfun(@(x) length(x.TBx),nim.subunits(fit_subs(nonpar_subs)));  %get the number of TBs for each subunit
if ~isempty(nonpar_subs); assert(length(unique(n_TBs)) == 1,'Have to have same number of tent-bases for each subunit'); end;
n_TBs = unique(n_TBs);
if isempty(n_TBs); n_TBs = 0; end;

nontarg_g = nim.process_stimulus(Xstims,non_fit_subs,gain_funs); %get output of nontarget subunits
if ~fit_spk_hist && spkhstlen > 0 %add in spike history filter output, if we're not fitting it
    nontarg_g = nontarg_g + Xspkhst*nim.spk_hist.coefs(:);
end

% COMPUTE NEW X-MATRIX OUT OF TENT-BASIS OUTPUTS
XNL = zeros(NT,length(nonpar_subs)*n_TBs); %initialize X matrix which is for the NL BFs of each module
for ii = 1:length(nonpar_subs) %for each module
    cur_sub = fit_subs(nonpar_subs(ii));
    gint = Xstims{nim.subunits(cur_sub).Xtarg}*nim.subunits(cur_sub).filtK;
    % The output of the current model's internal filter projected onto the tent basis representation
    if isempty(gain_funs)
        tbf_out = nim.subunits(cur_sub).weight * nim.subunits(cur_sub).tb_rep(gint);
    else
        tbf_out = nim.subunits(cur_sub).weight * bsxfun(@times,nim.subunits(cur_sub).tb_rep(gint),gain_funs(:,cur_sub));
    end
    XNL(:,((ii-1)*n_TBs + 1):(ii*n_TBs)) = tbf_out; % assemble filtered NLBF outputs into X matrix
end

[~,~,par_gint] = nim.process_stimulus(Xstims,fit_subs(par_subs),gain_funs); %get the filter outputs of all target subunits with parametric NLs

% CREATE INITIAL PARAMETER VECTOR
% Compute initial fit parameters
init_params = [];
for imod = fit_subs(nonpar_subs)
    init_params = [init_params; nim.subunits(imod).TBy']; %compile TB parameters 
end
par_cnt = 0; %counts number of parametric NL parameters were estimating
for imod = fit_subs(par_subs)
    init_params = [init_params; nim.subunits(imod).NLparams']; %compile parametric NL params
    par_cnt = par_cnt + length(nim.subunits(imod).NLparams);
end

% Add in spike history coefs
if fit_spk_hist
    init_params = [init_params; nim.spk_hist.coefs];
end

init_params = [init_params; nim.spkNL.theta]; %add constant offset

lambda_nl = nim.get_reg_lambdas('sub_inds',fit_subs(nonpar_subs),'nld2');
if any(lambda_nl > 0)
    Tmat = nim.make_NL_Tmat;
else
    Tmat = [];
end

% PROCESS CONSTRAINTS
use_con = 0;
[LB,UB] = deal(nan(size(init_params)));
Aeq = []; beq = [];% initialize constraint parameters
% Check for spike history coef constraints
if fit_spk_hist
    % negative constraint on spk history coefs
    if nim.spk_hist.negCon
        spkhist_inds = length(nonpar_subs)*n_tbfs + par_cnt + (1:spkhstlen);
        LB(spkhist_inds) = -Inf;
        UB(spkhist_inds) = 0;
        use_con = 1;
    end
end

% Process NL monotonicity constraints, and constraints that the tent basis
% centered at 0 should have coefficient of 0 (eliminate y-shift degeneracy)
A = []; b = [];
if any(arrayfun(@(x) x.TBparams.NLmon,nim.subunits(fit_subs(nonpar_subs))) ~= 0) %if any of our target nonpar subunits have a monotonicity constraint
    zvec = zeros(1,length(init_params)); % indices of tent-bases centered at 0
    for ii = 1:length(nonpar_subs)
        cur_range = (ii-1)*n_TBs + (1:n_TBs);
        % For monotonicity constraint
        if nim.subunits(fit_subs(nonpar_subs(ii))).TBparams.NLmon ~= 0
            for jj = 1:length(cur_range)-1 %create constraint matrix
                cur_vec = zvec;
                cur_vec(cur_range([jj jj + 1])) = nim.subunits(fit_subs(nonpar_subs(ii))).TBparams.NLmon*[1 -1];
                A = cat(1,A,cur_vec);
            end
        end
        % Constrain the 0-coefficient to be 0
        [~,zp] = find(nim.subunits(fit_subs(nonpar_subs(ii))).TBx == 0);
        assert(~isempty(zp),'Need one TB to be centered at 0');
        cur_vec = zvec;
        cur_vec(cur_range(zp)) = 1;
        Aeq = cat(1,Aeq,cur_vec);
    end
    b = zeros(size(A,1),1);
    beq = zeros(size(Aeq,1),1);
    use_con = 1;
end
par_cnt = 0;
param_inds = cell(length(par_subs),1);
for ii = 1:length(par_subs)
    cur_npars = length(nim.subunits(fit_subs(par_subs(ii))).NLparams);
    param_inds{ii} = length(nonpar_subs)*n_TBs + par_cnt + (1:cur_npars);
    par_cnt = par_cnt + cur_npars;
    cur_NL_cons = nim.subunits(fit_subs(par_subs(ii))).NLparam_con;
    LB(param_inds{ii}(cur_NL_cons == 1)) = 0; %positivity constraints
    UB(param_inds{ii}(cur_NL_cons == -1)) = 0; %negativity constraints
    if any(isinf(cur_NL_cons))
       cur_con_ind = false(1,length(init_params));
       cur_con_ind(param_inds{ii}(isinf(cur_NL_cons))) = true;
       Aeq = cat(1,Aeq,cur_con_ind);
       cur_con_to = init_params(cur_con_ind);
       beq = cat(1,beq,cur_con_to);
    end
    if any(cur_NL_cons ~= 0)
        use_con = 1;
    end
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

fit_opts = struct('fit_spk_hist', fit_spk_hist, 'fit_subs',fit_subs,'nonpar_subs',nonpar_subs,...
    'par_subs',par_subs); %put any additional fitting options into this struct
fit_opts.param_inds = param_inds; %have to add this field separately

opt_fun = @(K) internal_LL_NLs(nim,K, Robs, XNL, par_gint, Xspkhst,nontarg_g,gain_funs,Tmat,fit_opts);

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

nlmat = reshape(params(1:length(nonpar_subs)*n_TBs),n_TBs,length(nonpar_subs)); %take output K vector and restructure into a matrix of NLBF coefs, one for each module
nlmat_resc = nlmat;
for ii = 1:length(nonpar_subs);
    cur_pset = ((ii-1)*n_TBs+1) : (ii*n_TBs);
    thisnl = nlmat(:,ii); %NL coefs for current subunit
    cur_std = std(XNL(:,cur_pset)*thisnl);
    if rescale_NLs %rescale so that the std dev of the subunit output is conserved
        thisnl = thisnl*nim.subunits(fit_subs(ii)).scale/cur_std;
    else
        nim.subunits(fit_subs(ii)).scale = cur_std; %otherwise adjust the model output std dev
    end
    nim.subunits(fit_subs(ii)).TBy = thisnl';
    nlmat_resc(:,ii) = thisnl';
end
for ii = 1:length(par_subs)
    cur_params = params(param_inds{ii});
    nim.subunits(fit_subs(par_subs(ii))).NLparams = cur_params';
end
if fit_spk_hist
    nim.spk_hist.coefs = params(length(nonpar_subs)*n_TBs + par_cnt + (1:spkhstlen));
end

% If rescaling the Nls, we need to resestimate the offset theta after scaling
if rescale_NLs && ~isempty(nonpar_subs)
    resc_nlvec = nlmat_resc(:);
    new_g_out = XNL*resc_nlvec;
    G = nontarg_g + new_g_out;
    if ~isempty(par_subs)
        [~, fgint, ~] = process_stimulus(nim,Xstims,fit_subs(par_subs),gain_funs);
        mod_weights = [nim.subunits(fit_subs(par_subs)).weight];
        G = G + fgint*mod_weights';
    end
    if spkhstlen > 0
        G = G + Xspkhst*nim.spk_hist.coefs;
    end
    init_theta = params(end);
    opts.Display = 'off';opts.GradObj = 'on'; opts.LargeScale = 'off';
    new_theta = fminunc( @(K) internal_theta_opt(nim,K,G,Robs), init_theta, opts);
    nim.spkNL.theta = new_theta;
else
    nim.spkNL.theta = params(end);
end

[LL,~,mod_internals,LL_data] = nim.eval_model(Robs,Xstims,'gain_funs',gain_funs);
nim = nim.set_subunit_scales(mod_internals.fgint); %update filter scales
cur_fit_details = struct('fit_type','upstream_NLs','LL',LL,'filt_pen',LL_data.filt_pen,...
    'NL_pen',LL_data.NL_pen,'FO_optim',first_order_optim);
nim.fit_props = cur_fit_details;
nim.fit_hist = cat(1,nim.fit_hist,cur_fit_details);
end

%%
function [penLL, penLLgrad] = internal_LL_NLs(nim,params, Robs, XNL, par_gint, Xspkhst, nontarg_g, gain_funs, Tmat,fit_opts)
%computes the LL and its gradient for a given set of upstream NL parameters

% Useful params
fit_subs = fit_opts.fit_subs; nonpar_subs = fit_opts.nonpar_subs; par_subs = fit_opts.par_subs;
param_inds = fit_opts.param_inds;
n_par_pars = sum(cellfun(@(x) length(x),param_inds)); %total number of parametric NL params
Nfit_subs = length(fit_subs);
if ~isempty(nonpar_subs); n_TBs = length(nim.subunits(fit_subs(nonpar_subs(1))).TBx); end;
spkhstlen = nim.spk_hist.spkhstlen;
mod_weights = [nim.subunits(fit_subs).weight]';

% ESTIMATE GENERATING FUNCTIONS (OVERALL AND INTERNAL)
theta = params(end); %offset
G = theta + nontarg_g;
if ~isempty(nonpar_subs)
    all_TBy = params(1:length(nonpar_subs)*n_TBs);
    G = G + XNL*all_TBy;
end

%Add contributions from target subunits with parametric NLs
if ~isempty(par_subs)
    fgint = zeros(size(par_gint));
    for ii = 1:length(par_subs)
        cur_params = params(param_inds{ii});
        nim.subunits(fit_subs(par_subs(ii))).NLparams = cur_params; %set NL parameters to new values
        fgint(:,ii) = nim.subunits(fit_subs(par_subs(ii))).apply_NL(par_gint(:,ii));
    end
    % Multiply by weight (and multiplier, if appl) and add to generating function
    if isempty(gain_funs)
        G = G + fgint*mod_weights(par_subs);
    else
        G = G + (fgint.*gain_funs(:,fit_subs(par_subs)))*mod_weights(par_subs);
    end
end

%add contribution from spike history filter
if fit_opts.fit_spk_hist
    G = G + Xspkhst*params(length(nonpar_subs)*n_TBs + n_par_pars + (1:spkhstlen));
end

pred_rate = nim.apply_spkNL(G);
penLL = nim.internal_LL(pred_rate,Robs); %compute LL
%residual = LL'[r].*F'[g]
residual = nim.internal_LL_deriv(pred_rate,Robs) .* nim.apply_spkNL_deriv(G, pred_rate < nim.min_pred_rate);

penLLgrad = zeros(length(params),1); %initialize LL gradient
if ~isempty(nonpar_subs)
    penLLgrad(1:length(nonpar_subs)*n_TBs) = residual'*XNL; %gradient for tent-basis coefs
end
for ii = 1:length(par_subs)
    param_grad = nim.subunits(fit_subs(par_subs(ii))).NL_grad_param(par_gint(:,ii));
    penLLgrad(param_inds{ii}) = residual'*param_grad * mod_weights(par_subs(ii));
end
penLLgrad(end) = sum(residual);% Calculate derivatives with respect to constant term (theta)

% Calculate derivative with respect to spk history filter
if fit_opts.fit_spk_hist
    penLLgrad(Nfit_subs*n_TBs+(1:spkhstlen)) = residual'*Xspkhst;
end

% COMPUTE L2 PENALTIES AND GRADIENTS
lambdas = nim.get_reg_lambdas('sub_inds',fit_subs(nonpar_subs),'nld2');
if any(lambdas > 0)
    TBymat = reshape(all_TBy,n_TBs,[]);
    reg_penalties = lambdas.* sum((Tmat * TBymat).^2);
    pen_grads = 2*(Tmat' * Tmat * TBymat);
    pen_grads = reshape(bsxfun(@times,pen_grads,lambdas),[],1);
    penLL = penLL - sum(reg_penalties);
    penLLgrad(1:length(nonpar_subs)*n_TBs) = penLLgrad(1:length(nonpar_subs)*n_TBs) - pen_grads;
end
% CONVERT TO NEGATIVE LLS AND NORMALIZE BY NSPKS
Nspks = sum(Robs);
penLL = -penLL/Nspks;
penLLgrad = -penLLgrad/Nspks;
end

%%
function [LL,grad] = internal_theta_opt(nim,theta,G,Robs)
%computes LL and its gradient for given additive offset term theta
G = G + theta;
pred_rate = nim.apply_spkNL(G);
LL = nim.internal_LL(pred_rate,Robs);
%residual = LL'[r].*F'[g]
residual = nim.internal_LL_deriv(pred_rate,Robs) .* nim.apply_spkNL_deriv(G,pred_rate < nim.min_pred_rate);
grad = sum(residual);
Nspks = sum(Robs);
LL=-LL/Nspks;
grad=-grad'/Nspks;
end


