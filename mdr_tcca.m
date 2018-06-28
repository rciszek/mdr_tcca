% mdr_tcca - A MATLAB implementation of multiview dimensionality reduction using 
% Tensor Canonical Correlation Analysis, presented by Luo et al. in 
% https://arxiv.org/pdf/1502.02330.pdf
%
% Syntax:  [reduction] = mdr_tcca(views,d,epsilon)
%
% Inputs:
%    views - A cell array of NxM matrices
%    d - Final dimensionality
%    epsilon - Regularization trade-off factor, non-negative.
%    maxiters - Maximum number of ALS iterations in CP decomposition
%    verbosity - Fit error verbosity. If zero no info is printed, 
%                otherwise info is printed every n iterations.
%
% Outputs:
%    reduction - Multiview dimensionality reduction
%    f - Factorization as ktensor
%
% Example: 
%    mvdr_tcca({first_view, second_view})
%
% Requires: Tensor Toolbox for MATLAB by Sandia National Labs.
%           https://gitlab.com/tensors/tensor_toolbox
%
% Author: Robert Ciszek 
% July 2016; Last revision: 24-June-2017

function [Z,f] = mdr_tcca(views,varargin)
    %Parse inputs
    params = inputParser;
    params.addRequired('views',@iscell);
    params.addParameter('d',2,@(x) isscalar(x) & x > 0);
    params.addParameter('epsilon',1,@(x) isscalar(x) & x > 0);
    params.addParameter('maxiters', 50,@(x) isscalar(x) & x > 0);
    params.addParameter('verbosity', 1,@(x) isscalar(x));
    
    params.parse(views,varargin{:});

    views = params.Results.views;
    d = params.Results.d;
    epsilon = params.Results.epsilon;
    maxiters = params.Results.maxiters;
    verbosity = params.Results.verbosity;
    
    
    %All views are assumed to contain equal number of samples
    n_samples = size(views{1},1);
    n_views = length(views);
    
    %Center each view
    for i=1:n_views
        views{i} = tensor(views{i} - repmat(mean(views{i} ), n_samples,1));
    end
    
    %Calculate variances
    variances = cell(size(views));
    for i=1:n_views
        variances{i} =  (double(views{i})'*double(views{i}))/n_samples;
        variances{i} = variances{i} +  epsilon*ones(size(variances{i}));
    end
    %Calculate covariances
    covariances = [];
    for i=1:n_samples
       outer_product = views{1}(i,:);
       for j=2:length(views)
           outer_product = ttt(outer_product,views{j}(i,:));
       end
       if isempty(covariances)
           covariances = outer_product;
       else
           covariances = covariances+outer_product;
       end
    end   
    covariances = covariances / n_samples;
    
    M = covariances;
    for i=1:length(variances)
       M = ttm(M,pinv(variances{i})^1/2,i);
    end
    
    f = cp_als(M,d,'maxiters',maxiters,'printitn',verbosity);
    Z = zeros(n_samples,d*2);
    for i=1:n_views
        Z(:,(1+(i-1)*d):(i*d)) = double(views{i})*(pinv(variances{i})^1/2)*f.U{i};
    end
    
end
