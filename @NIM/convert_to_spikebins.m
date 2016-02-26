function spikebins = convert_to_spikebins( binned_spikes )
% Usage: spikebins = convert_to_spikebins( binned_spikes )
% Takes a binned spike count vector and converts it into a vector of length Nspks containing spike indices 

binned_spikes = binned_spikes(:);
unique_spk_cnts = unique(binned_spikes);
spikebins = [];
for i = 2:length(unique_spk_cnts)
	cur_set = find(binned_spikes == unique_spk_cnts(i));
	spikebins = [spikebins; repmat(cur_set,unique_spk_cnts(i),1)];
end
spikebins = sort(spikebins);
