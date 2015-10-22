# NIMclass
class-based implementation of nonlinear input model
Builds on the original "NIMtoolbox" which can be downloaded here: http://neurotheory.umd.edu/nimcode
See also: McFarland JM, Cui Y, Butts DA (2013) Inferring nonlinear neuronal computation based on physiologically plausible inputs. PLoS Computational Biology 9(7): e1003142.

In addition to being object-oriented, there are several additions/changes.
1) The stimulus Xmat now must be a cell array, and each subunit has a field which specifies which element of the stimulus cell array it 'acts on'. 
2) There is now an option to fit an 'offset' term along with the filters.
3) Optional inputs are now provided through 'option-flag/value' pairs
4) The likelihood function is now specified separately from the noise distribution model.
5) There are several new options for the upstream NLs, and spkNL, including 'rectified power-law functions'

and many more minor changes... See the help docs for each function.
The fit_filters function has also had some significant revamping that greatly speeds up estimation of high-dimensional filters, particularly for models with lots of subunits
