function [] = display_model(nim,Robs,Xstims,varargin)
%         [] = nim.display_model(<Robs>,<Xstims>,varargin)
%         Creates a display of the elements of a given NIM
%              INPUTS:
%                   <Robs>: observed spiking data. Needed if you want to utilize a spike-history
%                       filter. Otherwise set as empty
%                   <Xstims>: Stimulus cell array. Needed if you want to display the distributions of generating signals
%                   optional_flags:
%                         ('xtargs',xtargs): indices of stimuli for which we want to plot the filters
%                         'no_spknl': include this flag to suppress plotting of the spkNL
%                         'no_spk_hist': include this flag to suppress plotting of spike history filter
%                         ('gain_funs',gain_funs): if you want the computed generating signals to account for specified gain_funs

if nargin < 2; Robs = []; end;
if nargin < 3; Xstims = []; end;

Xtargs = [1:length(nim.stim_params)]; %default plot filters for all stimuli
plot_spkNL = true;
plot_spk_hist = true;
gain_funs = [];
j = 1; %initialize counter after required input args
while j <= length(varargin)
    switch lower(varargin{j})
        case 'xtargs'
            Xtargs = varargin{j+1};
            assert(all(ismember(Xtargs,1:length(nim.stim_params))),'invalid Xtargets specified');
            j = j + 2;
        case 'no_spknl'
            plot_spkNL = false;
            j = j + 1;
        case 'gain_funs'
            gain_funs = varargin{j+1};
            j = j + 2;
        case 'no_spk_hist'
            plot_spk_hist = false;
            j = j + 1;
        otherwise
            error('Invalid input flag');
    end
end

Nsubs = length(nim.subunits);
spkhstlen = nim.spk_hist.spkhstlen;
if spkhstlen > 0 && (plot_spk_hist || plot_spkNL)
    Xspkhst = create_spkhist_Xmat(Robs,nim.spk_hist.bin_edges);
end
n_hist_bins = 500; %internal parameter determining histogram resolution
if ~isempty(Xstims)
    [G, ~, gint] = nim.process_stimulus(Xstims,1:Nsubs,gain_funs);
    G = G + nim.spkNL.theta; %add in constant term
    if spkhstlen > 0 %add in spike history filter output
        G = G + Xspkhst*nim.spk_hist.coefs(:);
    end
else
    G = []; gint = [];
end

% PLOT SPIKING NL FUNCTION
if ~isempty(G) && plot_spkNL
    fig_handles.spk_nl = figure();
    n_bins = 1000; %bin resolution for G distribution
    [Gdist_y,Gdist_x] = hist(G,n_hist_bins); %histogram the generating signal
    
    %this is a hack to deal with cases where the threshold linear terms
    %create a min value of G
    if Gdist_y(1) > 2*Gdist_y(2)
        Gdist_y(1) = 1.5*Gdist_y(2);
    end
    
    cur_xrange = Gdist_x([1 end]);
    if strcmp(nim.spkNL.type,'logistic')
        NLx = linspace(cur_xrange(1),cur_xrange(2) + diff(cur_xrange)/2,500);
        cur_xrange = NLx([1 end]);
    else
        NLx = Gdist_x;
    end
    NLy = nim.apply_spkNL(NLx);
    NLy = NLy/nim.stim_params(1).dt; %convert to correct firing rate units
    
    [ax,h1,h2] = plotyy(NLx,NLy,Gdist_x,Gdist_y);
    set(h1,'linewidth',1)
    yr = [min(NLy) max(NLy)];
    xlim(ax(1),cur_xrange)
    xlim(ax(2),cur_xrange);
    ylim(ax(1),yr);
    
    xlabel('Generating function')
    ylabel(ax(1),'Predicted firing rate','fontsize',14);
    ylabel(ax(2),'Probability','fontsize',14)
    set(ax(2),'ytick',[]);
    title('Spiking NL','fontsize',14)
end

if nim.spk_hist.spkhstlen > 0 && plot_spk_hist
    fig_handles.spk_hist = figure();
    subplot(2,1,1)
    stairs(nim.spk_hist.bin_edges(1:end-1)*nim.stim_params(1).dt,nim.spk_hist.coefs);
    xlim(nim.spk_hist.bin_edges([1 end])*nim.stim_params(1).dt)
    xl = xlim();
    line(xl,[0 0],'color','k','linestyle','--');
    xlabel('Time lag');
    ylabel('Spike history filter')
    title('Spike history term','fontsize',14)
    
    subplot(2,1,2)
    stairs(nim.spk_hist.bin_edges(1:end-1)*nim.stim_params(1).dt,nim.spk_hist.coefs);
    xlim(nim.spk_hist.bin_edges([1 end-1])*nim.stim_params(1).dt)
    set(gca,'xscale','log')
    xl = xlim();
    line(xl,[0 0],'color','k','linestyle','--');
    xlabel('Time lag');
    ylabel('Spike history filter')
    title('Spk Hist Log-axis','fontsize',14)
end

% CREATE FIGURE SHOWING INDIVIDUAL SUBUNITS
for tt = Xtargs(Xtargs > 0) %loop over stimuli
    cur_subs = find([nim.subunits(:).Xtarg] == tt); %set of subunits acting on this stim
    
    if ~isempty(cur_subs)
        fig_handles.stim_filts = figure();
        if nim.stim_params(tt).dims(3) > 1 %if 2-spatial-dimensional stim
            n_columns = nim.stim_params(tt).dims(1) + 1;
            n_rows = length(cur_subs);
        else
            n_columns = max(round(sqrt(length(cur_subs)/2)),1);
            n_rows = ceil(length(cur_subs)/n_columns);
        end
        nLags = nim.stim_params(tt).dims(1); %time lags
        dt = nim.stim_params(tt).dt; %time res
        nPix = squeeze(nim.stim_params(tt).dims(2:end)); %spatial dimensions
        %create filter time lag axis
        if isempty(nim.stim_params(tt).tent_spacing)
            tax = (0:(nLags-1))*dt;
        else
            tax = (0:nim.stim_params(tt).tent_spacing:(nLags-1)*nim.stim_params(tt).tent_spacing)*dt;
        end
        tax = tax * 1000; % put in units of ms
        
        for imod = 1:length(cur_subs)
            cur_sub = nim.subunits(cur_subs(imod));
            
            if nim.stim_params(tt).dims(3) == 1 %if < 2 spatial dimensions
                %PLOT FILTER
                subplot(n_rows,2*n_columns,(imod-1)*2+1);
                if nPix == 1 %if temporal-only stim
                    %                             if isfield(thismod, 'keat_basis')
                    %                                 kblen = size(thismod.keat_basis,2);
                    %                                 tax = (0:kblen-1)*dt*1000;
                    %                                 plot(tax,thismod.filtK(:)'*thismod.keat_basis,'.-');
                    %                             else
                    plot(tax,cur_sub.filtK,'.-');
                    %                             end
                    xr = tax([1 end]);
                    line(xr,[0 0],'color','k','linestyle','--');
                    xlim(xr);
                    xlabel('Time lag')
                    ylabel('Filter coef');
                elseif nPix(2) == 1
                    imagesc(1:nPix(1),tax,reshape(cur_sub.filtK,nLags,nPix(1)));
                    cl = max(abs(cur_sub.filtK));
                    caxis([-cl cl]);
                    %colormap(jet);
                    colormap(gray);
                    set(gca,'ydir','normal');
                    xlabel('Pixels')
                    ylabel('Time lags');
                end
                if strcmp(cur_sub.NLtype,'lin')
                    title('Linear stimulus filter','fontsize',14)
                elseif cur_sub.weight > 0
                    title('Excitatory stimulus filter','fontsize',14);
                elseif cur_sub.weight < 0
                    title('Suppressive stimulus filter','fontsize',14);
                end
            else %if 2-spatial dimensional stim
                maxval = max(abs(cur_sub.filtK));
                for jj = 1:nim.stim_params(tt).dims(1) %loop over time slices
                    subplot(n_rows,n_columns,(imod-1)*n_columns + jj);
                    cur_fdims = jj - 1 + (1:nim.stim_params(tt).dims(1):prod(nim.stim_params(tt).dims));
                    imagesc(1:nPix(1),1:nPix(2),reshape(cur_sub.filtK(cur_fdims),nim.stim_params(tt).dims(2:end)));
                    colormap(gray)
                    if strcmp(cur_sub.NLtype,'lin')
                        title(sprintf('Lin-input Lag %d',jj-1),'fontsize',10);
                    elseif cur_sub.weight > 0
                        title(sprintf('E-Input Lag %d',jj-1),'fontsize',10);
                    elseif cur_sub.weight < 0
                        title(sprintf('S-Input Lag %d',jj-1),'fontsize',10);
                    end
                    caxis([-maxval maxval]*0.85);
                end
            end
            
            %PLOT UPSTREAM NL
            if nim.stim_params(tt).dims(3) == 1
                subplot(n_rows,2*n_columns,(imod-1)*2+2);
            else
                subplot(n_rows,n_columns,(imod)*n_columns);
            end
            if ~isempty(gint) %if computing distribution of filtered stim
                [gendist_y,gendist_x] = hist(gint(:,cur_subs(imod)),n_hist_bins);
                
                % Sometimes the gendistribution has a lot of zeros (dont want to screw up plot)
                [a b] = sort(gendist_y);
                if a(end) > a(end-1)*1.5
                    gendist_y(b(end)) = gendist_y(b(end-1))*1.5;
                end
            else
                gendist_x = linspace(-3,3,n_hist_bins); %otherwise, just pick an arbitrary x-axis to plot the NL
            end
            if strcmp(cur_sub.NLtype,'nonpar')
                cur_modx = cur_sub.TBx; cur_mody = cur_sub.TBy;
            else
                cur_modx = gendist_x; cur_mody = cur_sub.apply_NL(cur_modx);
            end
            cur_xrange = cur_modx([1 end]);
            
            if ~isempty(gint)
                [ax,h1,h2] = plotyy(cur_modx,cur_mody,gendist_x,gendist_y);
                if strcmp(cur_sub.NLtype,'nonpar')
                    set(h1,'Marker','o');
                end
                set(h1,'linewidth',1)
                xlim(ax(1),cur_xrange)
                xlim(ax(2),cur_xrange);
                ylim(ax(1),[min(cur_mody) max(cur_mody)]);
                set(ax(2),'ytick',[])
                yl = ylim();
                line([0 0],yl,'color','k','linestyle','--');
                ylabel(ax(1),'Subunit output','fontsize',12);
                ylabel(ax(2),'Probability','fontsize',12)
            else
                h = plot(cur_modx,cur_mody,'linewidth',1);
                if strcmp(cur_sub.NLtype,'nonpar')
                    set(h,'Marker','o');
                end
                xlim(cur_xrange)
                ylim([min(cur_mody) max(cur_mody)]);
                ylabel('Subunit output','fontsize',12);
            end
            box off
            xlabel('Internal generating function')
            title('Upstream NL','fontsize',14)
        end
    end
end
end

