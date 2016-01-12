function nim = fit_upstreamNLs(nim, Robs, Xstims, varargin)
%         nim = nim.fit_upstreamNLs(Robs, Xstims, <train_inds>, varargin)
%         Optimizes the upstream NLs (in terms of tent-basis functions)
%         INPUTS:
%            Robs: vector of response observations (e.g. spike counts)
%            Xstims: cell array of stimuli
%            <train_inds>: index values of data on which to fit the model [default to all indices in provided data]
%            optional flags:
%                ('fit_subs',fit_subs): set of subunits whos filters we want to optimize [default is all nonpar subunits]
%                ('gain_funs',gain_funs): matrix of multiplicative factors, one column for each subunit
%                ('optim_params',optim_params): struct of desired optimization parameters
%                ('silent',silent): boolean variable indicating whether to suppress the iterative optimization display
%                ('fit_spk_hist',fit_spk_hist): boolean indicating whether to hold the spk NL filter constant
%                ('rescale_nls',rescale_nls): boolean indicating if you want to rescale the NLs after fitting
%        OUTPUTS:
%            nim: output model struct

Nsubs = length(nim.subunits); %number of subunits

%set defaults for optional inputs
poss_fitsubs = find(strcmp(nim.get_NLtypes,'nonpar'))'; %can only fit subunits with nonpar NLs
fit_subs = find(strcmp(nim.get_NLtypes,'nonpar'))'; %default is to fit all subunits with nonpar NLs
gain_funs = []; %default has no gain_funs
train_inds = nan; %default nan means train on all data
silent = false; %default is show the optimization output
fit_spk_hist = nim.spk_hist.spkhstlen > 0; %default is fit the spkNL filter if it exists
rescale_nls = true; %default is to rescale the y-axis of NLs after estimation

option_list = {'subs','gain_funs','silent','rescale_nls','fit_spk_hist'}; %list of possible option strings

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

% Now parse explicit optional input args
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
                assert(all(ismember(fit_subs,poss_targets)),'specified target doesnt have non-parametric NL, or doesnt exist');
            case 'gain_funs'
                gain_funs = varargin{j+1};
            case 'optim_params'
                optim_params = varargin{j+1};
                assert(isstruct(optim_params),'optim_params must be a struct');
            case 'silent'
                silent = varargin{j+1};
                assert(ismember(silent,[0 1]),'silent must be 0 or 1');
            case 'rescale_nls'
                rescale_nls = varargin{j+1};
                assert(ismember(rescale_nls,[0 1]),'rescale_nls must be 0 or 1');
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
NT = length(Robs); %number of time points

Nfit_subs = length(fit_subs); %number of targeted subunits
if Nfit_subs == 0
	warning('No subunits to fit!');
	return
end
non_fit_subs = setdiff(1:Nsubs,fit_subs); %elements of the model held constant

n_TBs = arrayfun(@(x) length(x.TBx),nim.subunits(fit_subs));  %get the number of TBs for each subunit
assert(length(unique(n_TBs)) == 1,'Have to have same number of tent-bases for each subunit'); 
n_TBs = unique(n_TBs);

nontarg_g = nim.process_stimulus(Xstims,non_fit_subs,gain_funs); %get output of nontarget subunits
if ~fit_spk_hist && spkhstlen > 0 %add in spike history filter output, if we're not fitting it
    nontarg_g = nontarg_g + Xspkhst*nim.spk_hist.coefs(:);
end

% COMPUTE NEW X-MATRIX OUT OF TENT-BASIS OUTPUTS
XNL = zeros(NT,Nfit_subs*n_TBs); %initialize X matrix which is for the NL BFs of each module
for ii = 1:Nfit_subs %for each module
    cur_sub = fit_subs(ii);
    gint = Xstims{nim.subunits(cur_sub).Xtarg}*nim.subunits(cur_sub).filtK;
    % The output of the current model's internal filter projected onto the tent basis representation
    if isempty(gain_funs)
        tbf_out = nim.subunits(cur_sub).weight * nim.subunits(cur_sub).tb_rep(gint);
    else
        tbf_out = nim.subunits(cur_sub).weight * bsxfun(@times,nim.subunits(cur_sub).tb_rep(gint),gain_funs(:,cur_sub));
    end
    XNL(:,((ii-1)*n_TBs + 1):(ii*n_TBs)) = tbf_out; % assemble filtered NLBF outputs into X matrix
end

% CREATE INITIAL PARAMETER VECTOR
% Compute initial fit parameters
init_params = [];
for imod = fit_subs
    init_params = [init_params; nim.subunits(imod).TBy']; %compile TB parameters 
end

% Add in spike history coefs
if fit_spk_hist
    init_params = [init_params; nim.spk_hist.coefs];
end
init_params = [init_params; nim.spkNL.theta]; %add constant offset

lambda_nl = nim.get_reg_lambdas('subs',fit_subs,'nld2');
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
        spkhist_inds = Nfit_subs*n_tbfs + (1:spkhstlen);
        LB(spkhist_inds) = -Inf;
        UB(spkhist_inds) = 0;
        use_con = 1;
    end
end

% Process NL monotonicity constraints, and constraints that the tent basis
% centered at 0 should have coefficient of 0 (eliminate y-shift degeneracy)
A = []; b = [];
if any(arrayfun(@(x) x.TBparams.NLmon,nim.subunits(fit_subs)) ~= 0) %if any of our target nonpar subunits have a monotonicity constraint
    zvec = zeros(1,length(init_params)); % indices of tent-bases centered at 0
    for ii = 1:Nfit_subs
        cur_range = (ii-1)*n_TBs + (1:n_TBs);
        % For monotonicity constraint
        if nim.subunits(fit_subs(ii)).TBparams.NLmon ~= 0
            for jj = 1:length(cur_range)-1 %create constraint matrix
                cur_vec = zvec;
                cur_vec(cur_range([jj jj + 1])) = nim.subunits(fit_subs(ii)).TBparams.NLmon*[1 -1];
                A = cat(1,A,cur_vec);
            end
        end
        % Constrain the 0-coefficient to be 0
        [~,zp] = find(nim.subunits(fit_subs(ii)).TBx == 0);
        assert(~isempty(zp),'Need one TB to be centered at 0');
        cur_vec = zvec;
        cur_vec(cur_range(zp)) = 1;
        Aeq = cat(1,Aeq,cur_vec);
    end
    b = zeros(size(A,1),1);
    beq = zeros(size(Aeq,1),1);
    use_con = 1;
end
if any(~isnan(LB) | ~isnan(UB)) %check if there are any bound constraints
   LB(isnan(LB)) = -Inf; UB(isnan(UB)) = Inf; %set all unconstrained parameters to +/- Inf
else
    LB = []; UB = []; %if no bound constraints, set these to empty
end

if isfield(optim_params,'optimizer')
    optimizer = optim_params.('optimizer'); %if user-specified optimizer
else
    if ~use_con %if there are no constraints
        if exist('minFunc','file') == 2
            optimizer = 'minFunc';
        else
            optimizer = 'fminunc';
        end
    else
        optimizer = 'fmincon';
    end
end
optim_params = nim.set_optim_params(optimizer,optim_params,silent);
if ~silent; fprintf('Running optimization using %s\n\n',optimizer); end;

fit_opts = struct('fit_spk_hist', fit_spk_hist, 'fit_subs',fit_subs); %put any additional fitting options into this struct

opt_fun = @(K) internal_LL_NLs(nim,K, Robs, XNL, Xspkhst, nontarg_g, Tmat, fit_opts);

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
    case 'fminsearch'
        params = fminsearch(opt_fun,init_params,optim_params);
end
[~,penGrad] = opt_fun(params);
first_order_optim = max(abs(penGrad));
if first_order_optim > nim.opt_check_FO
    warning(sprintf('First-order optimality %.3f, fit might not be converged!',first_order_optim));
end

nlmat = reshape(params(1:Nfit_subs*n_TBs),n_TBs,Nfit_subs); %take output K vector and restructure into a matrix of NLBF coefs, one for each module
nlmat_resc = nlmat;
for ii = 1:Nfit_subs
    cur_pset = ((ii-1)*n_TBs+1) : (ii*n_TBs);
    thisnl = nlmat(:,ii); %NL coefs for current subunit
    cur_std = std(XNL(:,cur_pset)*thisnl);
    if rescale_nls %rescale so that the std dev of the subunit output is conserved
        thisnl = thisnl*nim.subunits(fit_subs(ii)).scale/cur_std;
    else
        nim.subunits(fit_subs(ii)).scale = cur_std; %otherwise adjust the model output std dev
    end
    nim.subunits(fit_subs(ii)).TBy = thisnl';
    nlmat_resc(:,ii) = thisnl';
end
if fit_spk_hist
    nim.spk_hist.coefs = params(Nfit_subs*n_TBs + (1:spkhstlen));
end

% If rescaling the Nls, we need to resestimate the offset theta after scaling
if rescale_nls
    resc_nlvec = nlmat_resc(:);
    new_g_out = XNL*resc_nlvec;
    G = nontarg_g + new_g_out;
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
nim.fit_history = cat(1,nim.fit_history,cur_fit_details);
end

%%
function [penLL, penLLgrad] = internal_LL_NLs(nim,params, Robs, XNL, Xspkhst, nontarg_g, Tmat,fit_opts)
%computes the LL and its gradient for a given set of upstream NL parameters

% Useful params
fit_subs = fit_opts.fit_subs; 
Nfit_subs = length(fit_subs);
spkhstlen = nim.spk_hist.spkhstlen;
n_TBs = length(nim.subunits(fit_subs(1)).TBx); 

% ESTIMATE GENERATING FUNCTIONS (OVERALL AND INTERNAL)
theta = params(end); %offset
G = theta + nontarg_g;
all_TBy = params(1:Nfit_subs*n_TBs);
G = G + XNL*all_TBy;

%add contribution from spike history filter
if fit_opts.fit_spk_hist
    G = G + Xspkhst*params(Nfit_subs*n_TBs + (1:spkhstlen));
end

pred_rate = nim.apply_spkNL(G);
[penLL,norm_fact] = nim.internal_LL(pred_rate,Robs); %compute LL

%residual = LL'[r].*F'[g]
residual = nim.internal_LL_deriv(pred_rate,Robs) .* nim.apply_spkNL_deriv(G, pred_rate < nim.min_pred_rate);

penLLgrad = zeros(length(params),1); %initialize LL gradient
penLLgrad(1:Nfit_subs*n_TBs) = residual'*XNL; %gradient for tent-basis coefs
penLLgrad(end) = sum(residual);% Calculate derivatives with respect to constant term (theta)

% Calculate derivative with respect to spk history filter
if fit_opts.fit_spk_hist
    penLLgrad(Nfit_subs*n_TBs+(1:spkhstlen)) = residual'*Xspkhst;
end

% COMPUTE L2 PENALTIES AND GRADIENTS
lambdas = nim.get_reg_lambdas('sub_inds',fit_subs,'nld2');
if any(lambdas > 0)
    TBymat = reshape(all_TBy,n_TBs,[]);
    reg_penalties = lambdas.* sum((Tmat * TBymat).^2);
    pen_grads = 2*(Tmat' * Tmat * TBymat);
    pen_grads = reshape(bsxfun(@times,pen_grads,lambdas),[],1);
    penLL = penLL - sum(reg_penalties);
    penLLgrad(1:Nfit_subs*n_TBs) = penLLgrad(1:Nfit_subs*n_TBs) - pen_grads;
end
% CONVERT TO NEGATIVE LLS AND NORMALIZE BY NSPKS
penLL = -penLL/norm_fact;
penLLgrad = -penLLgrad/norm_fact;
end
%%
function [LL,grad] = internal_theta_opt(nim,theta,G,Robs)
%computes LL and its gradient for given additive offset term theta
G = G + theta;
pred_rate = nim.apply_spkNL(G);
[LL,norm_fact] = nim.internal_LL(pred_rate,Robs);
%residual = LL'[r].*F'[g]
residual = nim.internal_LL_deriv(pred_rate,Robs) .* nim.apply_spkNL_deriv(G,pred_rate < nim.min_pred_rate);
grad = sum(residual);
LL=-LL/norm_fact;
grad=-grad'/norm_fact;
end


