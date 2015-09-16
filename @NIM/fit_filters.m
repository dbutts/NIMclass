function nim = fit_filters(nim, Robs, Xstims, varargin)
%         nim = nim.fit_filters(Robs, Xstims, <train_inds>, varargin)
%         estimate filters of NIM model.
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
%         OUTPUTS:
%            new nim object with optimized subunit filters
%
Nsubs = length(nim.subunits); %number of subunits

% PROCESS INPUTS
fit_subs = 1:Nsubs; %defualt to fitting all subunits (plus -1 for spkHist filter)
gain_funs = []; %default has no gain_funs
train_inds = nan; %default nan means train on all data
optim_params = []; %default has no user-specified optimization parameters
fit_spk_hist = nim.spk_hist.spkhstlen > 0; %default is to fit the spkNL filter if it exists
silent = false; %default is to display optimization output
j = 1;
while j <= length(varargin)
    flag_name = varargin{j};
    if ~ischar(flag_name)%if not a flag, it must be train_inds
        train_inds = flag_name;
        j = j + 1; %only one argument here
    else
        switch lower(flag_name)
            case 'sub_inds'
                fit_subs = varargin{j+1};
                assert(all(ismember(fit_subs,1:Nsubs)),'invalid target subunits specified');
            case 'gain_funs'
                gain_funs = varargin{j+1};
            case 'optim_params'
                optim_params = varargin{j+1};
                assert(isstruct(optim_params),'optim_params must be a struct');
            case 'silent'
                silent = varargin{j+1};
                assert(ismember(silent,[0 1]),'silent must be 0 or 1');
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
init_params = [];
lambda_L1 = zeros(size(init_params));
sign_con = zeros(size(init_params));
for imod = fit_subs
    cur_kern = nim.subunits(imod).filtK;
    if (nim.subunits(imod).Ksign_con ~= 0) %add sign constraints on the filters of this subunit if needed
        sign_con(length(init_params)+(1:length(cur_kern))) = nim.subunits(imod).Ksign_con;
    end
    lambda_L1(length(init_params) + (1:length(cur_kern))) = nim.subunits(imod).reg_lambdas.l1;
    init_params = [init_params; cur_kern]; % add coefs to initial param vector
end
lambda_L1 = lambda_L1'/sum(Robs); % since we are dealing with LL/spk
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

% IDENTIFY ANY CONSTRAINTS
use_con = 0;
LB = -Inf*ones(size(init_params));
UB = Inf*ones(size(init_params));
% Constrain any of the filters to be positive or negative
if any(sign_con ~= 0)
    LB(sign_con == 1) = 0;
    UB(sign_con == -1) = 0;
    use_con = 1;
end
if fit_spk_hist %if optimizing spk history term
    %negative constraint on spk history coefs
    if nim.spk_hist.negCon
        spkhist_inds = Nfit_filt_params + (1:spkhstlen);
        UB(spkhist_inds) = 0;
        use_con = 1;
    end
end

% GENERATE REGULARIZATION MATRICES
Tmats = nim.make_Tikhonov_matrices();

fit_opts = struct('fit_spk_hist', fit_spk_hist, 'fit_subs',fit_subs); %put any additional fitting options into this struct
%the function we want to optimize
opt_fun = @(K) internal_LL_filters(nim,K,Robs,Xstims,Xspkhst,nontarg_g,gain_funs,Tmats,fit_opts);

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
kOffset = 0; %position counter for indexing param vector
for ii = 1:Nfit_subs
    filtLen = length(nim.subunits(fit_subs(ii)).filtK);
    cur_kern = params((1:filtLen) + kOffset); %grab parameters corresponding to this subunit's filters
    nim.subunits(fit_subs(ii)).filtK = cur_kern(:); %assign new filter values
    kOffset = kOffset + filtLen;
end

[LL,~,mod_internals,LL_data] = nim.eval_model(Robs,Xstims,'gain_funs',gain_funs);
nim = nim.set_subunit_scales(mod_internals.fgint); %update filter scales
cur_fit_details = struct('fit_type','filter','LL',LL,'filt_pen',LL_data.filt_pen,...
    'NL_pen',LL_data.NL_pen,'FO_optim',first_order_optim);
nim.fit_props = cur_fit_details; %store details of this fit
nim.fit_hist = cat(1,nim.fit_hist,cur_fit_details);
end

%%
function [penLL, penLLgrad] = internal_LL_filters(nim,params,Robs,Xstims,Xspkhst,nontarg_g,gain_funs,Tmats,fit_opts)
%computes the penalized LL and its gradient wrt the filters for the given nim
%with parameter vector params

fit_subs = fit_opts.fit_subs;
Nfit_subs = length(fit_subs); %number of targeted subs

% USEFUL VALUES
theta = params(end); % offset
gint = nan(length(Robs),Nfit_subs); %initialize matrix for storing filter outputs
filtLen = zeros(Nfit_subs,1); %store the length of each (target) sub's filter
filtKs = cell(Nfit_subs,1); %store the filter coefs for all (target) subs)
param_inds = cell(Nfit_subs,1); %this will store the index values of each subunit's filter coefs within the parameter vector
Xtarg_set = [nim.subunits(fit_subs).Xtarg]; %vector of Xfit_subs for set of subunits being optimized
un_Xtargs = unique(Xtarg_set); %set of unique Xfit_subs
mod_NL_types = {nim.subunits(fit_subs).NLtype}; %NL types for each targeted subunit
unique_NL_types = unique(mod_NL_types); %unique set of NL types being used
mod_weights = [nim.subunits(fit_subs).weight]'; %signs of targeted subunits

G = theta + nontarg_g; % initialize overall generating function G with the offset term and the contribution from nontarget subs

NKtot = 0;  %init filter coef counter
for ii = 1:Nfit_subs %loop over subunits, get filter coefs and their indices within the parameter vector
    filtLen(ii) = length(nim.subunits(fit_subs(ii)).filtK); % length of filter
    param_inds{ii} = NKtot + (1:filtLen(ii)); %set of param indices associated with this subunit's filters
    filtKs{ii} = params(param_inds{ii}); %store filter coefs
    NKtot = NKtot + filtLen(ii); %inc counter
end
for ii = 1:length(un_Xtargs) %loop over the unique Xtargs and compute the generating signals for all relevant filters
    cur_subs = find(Xtarg_set == un_Xtargs(ii)); %set of targeted subunits that act on this Xtarg
    gint(:,cur_subs) = Xstims{un_Xtargs(ii)} * cat(2,filtKs{cur_subs}); %apply filters to stimulus
end

fgint = gint; %init subunit outputs by filter outputs
for ii = 1:length(unique_NL_types) %loop over unique subunit NL types and apply NLs to gint in batch
    cur_subs = find(strcmp(mod_NL_types,unique_NL_types{ii})); %set of subs with this NL type
    if ~strcmp(unique_NL_types{ii},'lin') %if its a linear subunit we dont have to do anything
        NLparam_mat = cat(1,nim.subunits(fit_subs(cur_subs)).NLparams); %matrix of upstream NL parameters
        if isempty(NLparam_mat) || (length(cur_subs) > 1 && max(max(abs(diff(NLparam_mat)))) > 0) 
            use_batch_calc = true; %if there are no NLparams, or if all subunits have the same NLparams, use batch calc
        else
            use_batch_calc = false;
        end
        if strcmp(unique_NL_types{ii},'nonpar') || ~use_batch_calc %if were using nonpar NLs or parametric NLs with unique parameters, need to apply NLs individually
            for jj = 1:length(cur_subs) %for TB NLs need to apply each subunit's NL individually
                fgint(:,cur_subs(jj)) = nim.subunits(fit_subs(cur_subs(jj))).apply_NL(gint(:,cur_subs(jj)));
            end
        else %apply upstream NL in batch to all subunits in current set
            fgint(:,cur_subs) = nim.subunits(fit_subs(cur_subs(1))).apply_NL(gint(:,cur_subs)); %apply upstream NL to all subunits of this type
        end
    end
end

% Multiply by weight (and multiplier, if appl) and add to generating function
if ~isempty(fit_subs)
if isempty(gain_funs)
    G = G + fgint*mod_weights;
else
    G = G + (fgint.*gain_funs(:,fit_subs))*mod_weights;
end
end

% Add contribution from spike history filter
if fit_opts.fit_spk_hist
    G = G + Xspkhst*params(NKtot + (1:nim.spk_hist.spkhstlen));
end

pred_rate = nim.apply_spkNL(G);
penLL = nim.internal_LL(pred_rate,Robs); %compute LL

%residual = LL'[r].*F'[g]
residual = nim.internal_LL_deriv(pred_rate,Robs) .* nim.apply_spkNL_deriv(G,pred_rate <= nim.min_pred_rate);

penLLgrad = zeros(length(params),1); %initialize LL gradient
penLLgrad(end) = sum(residual);      %Calculate derivatives with respect to constant term (theta)

% Calculate derivative with respect to spk history filter
if fit_opts.fit_spk_hist
    penLLgrad(NKtot+(1:nim.spk_hist.spkhstlen)) = residual'*Xspkhst;
end

for ii = 1:length(un_Xtargs) %loop over unique Xfit_subs and compute LL grad wrt stim filters
    cur_sub_inds = find(Xtarg_set == un_Xtargs(ii)); %set of subunits with this Xtarget
    cur_NL_types = mod_NL_types(cur_sub_inds); %NL types of current subs
    cur_unique_NL_types = unique(cur_NL_types); %set of unique NL types
    
    if length(cur_sub_inds) == 1 && strcmp(cur_unique_NL_types,'lin') %if there's only a single linear subunit, this is a faster calc
        if isempty(gain_funs)
            penLLgrad(param_inds{cur_sub_inds}) = residual'*Xstims{un_Xtargs(ii)} * nim.subunits(cur_sub_inds).weight;
        else
            penLLgrad(param_inds{cur_sub_inds}) = (gain_funs.*residual)'*Xstims{un_Xtargs(ii)} * nim.subunits(cur_sub_inds).weight;
        end
    else %otherwise, compute a matrix of upstream NL derivatives fpg
        fpg = ones(length(residual),length(cur_sub_inds)); %initialize to linear NL derivative (all ones)
        for jj = 1:length(cur_unique_NL_types) %loop over unique NL types
            cur_sub_subinds = find(strcmp(cur_NL_types,cur_unique_NL_types{jj})); %indices of current subset of subunits
            if strcmp(cur_unique_NL_types{jj},'nonpar')
                for kk = 1:length(cur_sub_subinds) %if nonpar, need to apply each NL derivative individually
                    fpg(:,cur_sub_subinds(kk)) = nim.subunits(fit_subs(cur_sub_inds(cur_sub_subinds(kk)))).apply_NL_deriv(gint(:,cur_sub_inds(cur_sub_subinds(kk))));
                end
            else %otherwise we can apply the NL to all subunits at once
                fpg(:,cur_sub_subinds) = nim.subunits(fit_subs(cur_sub_inds(cur_sub_subinds(1)))).apply_NL_deriv(gint(:,cur_sub_inds(cur_sub_subinds)));
            end
        end
        target_params = cat(2,param_inds{cur_sub_inds}); %indices of filter coefs for current set of targeted subunits
        %LL grad is residual * f'(.) *X *w, computed in parallel for all subunits targeting this Xtarg
        if isempty(gain_funs)
            penLLgrad(target_params) = bsxfun(@times,(bsxfun(@times,fpg,residual)'*Xstims{un_Xtargs(ii)}),mod_weights(cur_sub_inds))';
        else
            penLLgrad(target_params) = bsxfun(@times,(bsxfun(@times,fpg.*gain_funs(:,sub_inds),residual)'*Xstims{un_Xtargs(ii)}),mod_weights(cur_sub_inds))';
        end
    end
end

net_penalties = zeros(size(fit_subs));
net_pen_grads = zeros(length(params),1);
for ii = 1:length(Tmats) %loop over the derivative regularization matrices
    cur_subs = find([nim.subunits(fit_subs).Xtarg] == Tmats(ii).Xtarg); %set of subunits acting on the stimulus given by this Tmat
    if ~isempty(cur_subs)
    penalties = sum((Tmats(ii).Tmat * cat(2,filtKs{cur_subs})).^2);
    pen_grads = 2*(Tmats(ii).Tmat' * Tmats(ii).Tmat * cat(2,filtKs{cur_subs}));
    cur_lambdas = nim.get_reg_lambdas(Tmats(ii).type,'sub_inds',fit_subs(cur_subs)); %current lambdas
    net_penalties(cur_subs) = net_penalties(cur_subs) + penalties.*cur_lambdas;
    net_pen_grads(cat(2,param_inds{cur_subs})) = net_pen_grads(cat(2,param_inds{cur_subs})) + reshape(bsxfun(@times,pen_grads,cur_lambdas),[],1);
    end
end
l2_lambdas = nim.get_reg_lambdas('sub_inds',fit_subs,'l2');
if any(l2_lambdas > 0)
    net_penalties = net_penalties + l2_lambdas.*cellfun(@(x) sum(x.^2),filtKs)';
    for ii = 1:length(un_Xtargs)
        cur_subs = find(Xtarg_set == un_Xtargs(ii)); %set of targeted subunits that act on this Xtarg
        net_pen_grads(cat(2,param_inds{cur_subs})) = net_pen_grads(cat(2,param_inds{cur_subs})) + reshape(2*bsxfun(@times,l2_lambdas(cur_subs),cat(2,filtKs{cur_subs})),[],1);
    end
end

penLL = penLL - sum(net_penalties);
penLLgrad = penLLgrad - net_pen_grads;

% CONVERT TO NEGATIVE LLS AND NORMALIZE BY NSPKS
Nspks = sum(Robs);
penLL = -penLL/Nspks;
penLLgrad = -penLLgrad/Nspks;

end

