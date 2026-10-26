function [x,x_debias,objective,times,debias_start,mses,max_svd] = ...
         TwIST_GPU(y,A,tau,varargin)
%
% TwIST_GPU - GPU加速版本的TwIST算法
% 
% 此版本专为GPU计算优化，所有中间变量保持在GPU内存中
% 输入y可以是gpuArray或普通数组，输出x为gpuArray
%
% 使用方法与TwIST相同，但会利用GPU进行加速
%
% 作者: 基于Jose Bioucas-Dias和Mario Figueiredo的TwIST修改

%--------------------------------------------------------------
% test for number of required parametres
%--------------------------------------------------------------
if (nargin-length(varargin)) ~= 3
     error('Wrong number of required parameters');
end
%--------------------------------------------------------------
% Set the defaults for the optional parameters
%--------------------------------------------------------------
stopCriterion = 1;
tolA = 0.01;
debias = 0;
maxiter = 1000;
maxiter_debias = 200;
miniter = 5;
miniter_debias = 5;
init = 0;
enforceMonotone = 1;
compute_mse = 0;
plot_ISNR = 0;
AT = 0;
verbose = 1;
alpha = 0;
beta  = 0;
sparse = 1;
tolD = 0.001;
phi_l1 = 0;
psi_ok = 0;
% default eigenvalues 
lam1=1e-4;   lamN=1;
% 

% constants ans internal variables
for_ever = 1;
% maj_max_sv: majorizer for the maximum singular value of operator A
max_svd = 6*1e6; % 初始值
 
% Set the defaults for outputs that may not be computed
debias_start = 0;
x_debias = [];
mses = [];

%--------------------------------------------------------------
% Read the optional parameters
%--------------------------------------------------------------
if (rem(length(varargin),2)==1)
  error('Optional parameters should always go by pairs');
else
  for i=1:2:(length(varargin)-1)
    switch upper(varargin{i})
     case 'LAMBDA'
       lam1 = varargin{i+1};
     case 'ALPHA'
       alpha = varargin{i+1};
     case 'BETA'
       beta = varargin{i+1};
     case 'PSI'
       psi_function = varargin{i+1};
     case 'PHI'
       phi_function = varargin{i+1};   
     case 'STOPCRITERION'
       stopCriterion = varargin{i+1};
     case 'TOLERANCEA'       
       tolA = varargin{i+1};
     case 'TOLERANCED'
       tolD = varargin{i+1};
     case 'DEBIAS'
       debias = varargin{i+1};
     case 'MAXITERA'
       maxiter = varargin{i+1};
     case 'MAXIRERD'
       maxiter_debias = varargin{i+1};
     case 'MINITERA'
       miniter = varargin{i+1};
     case 'MINITERD'
       miniter_debias = varargin{i+1};
     case 'INITIALIZATION'
       if prod(size(varargin{i+1})) > 1   % we have an initial x
	 init = 33333;    % some flag to be used below
	 x = varargin{i+1};
       else 
	 init = varargin{i+1};
       end
     case 'MONOTONE'
       enforceMonotone = varargin{i+1};
     case 'SPARSE'
       sparse = varargin{i+1};
     case 'TRUE_X'
       compute_mse = 1;
       true = varargin{i+1};
       if prod(double((size(true) == size(y))))
           plot_ISNR = 1;
       end
     case 'AT'
       AT = varargin{i+1};
     case 'VERBOSE'
       verbose = varargin{i+1};
     otherwise
      % Hmmm, something wrong with the parameter string
      error(['Unrecognized option: ''' varargin{i} '''']);
    end;
  end;
end
%%%%%%%%%%%%%%


% twist parameters 
rho0 = (1-lam1/lamN)/(1+lam1/lamN);
if alpha == 0 
    alpha = 2/(1+sqrt(1-rho0^2));
end
if  beta == 0 
    beta  = alpha*2/(lam1+lamN);
end


if (sum(stopCriterion == [0 1 2 3])==0)
   error(['Unknwon stopping criterion']);
end

% if A is a function handle, we have to check presence of AT,
if isa(A, 'function_handle') & ~isa(AT,'function_handle')
   error(['The function handle for transpose of A is missing']);
end 


% if A is a matrix, we find out dimensions of y and x,
% and create function handles for multiplication by A and A',
% so that the code below doesn't have to distinguish between
% the handle/not-handle cases
if ~isa(A, 'function_handle')
   AT = @(x) A'*x;
   A = @(x) A*x;
end
% from this point down, A and AT are always function handles.

% 确保y在GPU上
if ~isa(y, 'gpuArray')
    y = gpuArray(single(y));
end

% Precompute A'*y since it'll be used a lot
Aty = AT(y);

% if phi was given, check to see if it is a handle and that it 
% accepts two arguments
if exist('psi_function','var')
   if isa(psi_function,'function_handle')
      try  % check if phi can be used, using Aty, which we know has 
           % same size as x
           dummy = psi_function(Aty,tau); 
           psi_ok = 1;
      catch
         error(['Something is wrong with function handle for psi'])
      end
   else
      error(['Psi does not seem to be a valid function handle']);
   end
else %if nothing was given, use soft thresholding
   psi_function = @(x,tau) soft_gpu(x,tau);
end

% if psi exists, phi must also exist
if (psi_ok == 1)
   if exist('phi_function','var')
      if isa(phi_function,'function_handle')
         try  % check if phi can be used, using Aty, which we know has 
              % same size as x
              dummy = phi_function(Aty); 
         catch
           error(['Something is wrong with function handle for phi'])
         end
      else
        error(['Phi does not seem to be a valid function handle']);
      end
   else
      error(['If you give Psi you must also give Phi']); 
   end
else  % if no psi and phi were given, simply use the l1 norm.
   phi_function = @(x) sum(abs(x(:))); 
   phi_l1 = 1;
end
    

%--------------------------------------------------------------
% Initialization
%--------------------------------------------------------------
switch init
    case 0   % initialize at zero, using AT to find the size of x
       x = AT(gpuArray.zeros(size(y), classUnderlying(y)));
    case 1   % initialize randomly, using AT to find the size of x
       x = gpuArray.randn(size(AT(gpuArray.zeros(size(y), classUnderlying(y)))), classUnderlying(y));
    case 2   % initialize x0 = A'*y
       x = Aty; 
    case 33333
       % initial x was given as a function argument; 确保在GPU上
       if ~isa(x, 'gpuArray')
           x = gpuArray(single(x));
       end
       if size(A(x)) ~= size(y)
          error(['Size of initial x is not compatible with A']); 
       end
    otherwise
       error(['Unknown ''Initialization'' option']);
end

% now check if tau is an array; if it is, it has to 
% have the same size as x
if prod(size(tau)) > 1
   try,
      dummy = x.*tau;
   catch,
      error(['Parameter tau has wrong dimensions; it should be scalar or size(x)']),
   end
end
      
% if the true x was given, check its size
if compute_mse & (size(true) ~= size(x))  
   error(['Initial x has incompatible size']); 
end


% if tau is large enough, in the case of phi = l1, thus psi = soft,
% the optimal solution is the zero vector
if phi_l1
   max_tau = max(abs(Aty(:)));
   if (tau >= max_tau)&(psi_ok==0)
      x = gpuArray.zeros(size(Aty), classUnderlying(Aty));
      objective(1) = 0.5*(y(:)'*y(:));
      times(1) = 0;
      if compute_mse
        mses(1) = sum(true(:).^2);
      end
      return
   end
end


% define the indicator vector or matrix of nonzeros in x
nz_x = (x ~= 0.0);
num_nz_x = gather(sum(nz_x(:)));  % gather用于显示

% Compute and store initial value of the objective function
resid =  y-A(x);
prev_f = gather(0.5*(resid(:)'*resid(:)) + tau*phi_function(x));


% start the clock
t0 = tic;

times(1) = toc(t0);
objective(1) = prev_f;

if compute_mse
   % mses(1) = sum(sum((x-true).^2));
   mses(1) = gather(sum((x-true).^2,"all"));
end

cont_outer = 1;
iter = 1;

if verbose
    fprintf(1,'\nInitial objective = %10.6e,  nonzeros=%7d\n',...
        prev_f,num_nz_x);
end

% variables controling first and second order iterations
IST_iters = 0;
TwIST_iters = 0;

% initialize
xm2=x;
xm1=x;

%--------------------------------------------------------------
% TwIST iterations
%--------------------------------------------------------------
while cont_outer
    % gradient
    grad = AT(resid);
    while for_ever
        % IST estimate
        x = psi_function(xm1 + grad/max_svd,tau/max_svd);

        if (IST_iters >= 2) | ( TwIST_iters ~= 0)
            % set to zero the past when the present is zero
            % suitable for sparse inducing priors
            if sparse
                mask = (x ~= 0);
                xm1 = xm1.* mask;
                xm2 = xm2.* mask;
            end
            % two-step iteration
            xm2 = (alpha-beta)*xm1 + (1-alpha)*xm2 + beta*x;
            % compute residual
            resid = y-A(xm2);
            f = gather(0.5*(resid(:)'*resid(:)) + tau*phi_function(xm2));
            if (f > prev_f) & (enforceMonotone)
                TwIST_iters = 0;  % do a IST iteration if monotonocity fails
            else
                TwIST_iters = TwIST_iters+1; % TwIST iterations
                IST_iters = 0;
                x = xm2;
                if mod(TwIST_iters,10000) == 0
                    max_svd = 0.9*max_svd;
                end
                break;  % break loop while
            end
        else
            resid = y-A(x);
            f = gather(0.5*(resid(:)'*resid(:)) + tau*phi_function(x));
            if f > prev_f
                % if monotonicity  fails here  is  because
                % max eig (A'A) > 1. Thus, we increase our guess
                % of max_svs
                max_svd = 2*max_svd;
                if verbose
                    fprintf('Incrementing S=%2.2e\n',max_svd)
                end
                IST_iters = 0;
                TwIST_iters = 0;
            else
                TwIST_iters = TwIST_iters + 1;
                break;  % break loop while
            end
        end
    end

    xm2 = xm1;
    xm1 = x;



    %update the number of nonzero components and its variation
    nz_x_prev = nz_x;
    nz_x = (x~=0.0);
    num_nz_x = gather(sum(nz_x(:)));
    num_changes_active = gather(sum(nz_x(:)~=nz_x_prev(:)));

    % take no less than miniter and no more than maxiter iterations
    switch stopCriterion
        case 0,
            % compute the stopping criterion based on the change
            % of the number of non-zero components of the estimate
            criterion =  num_changes_active;
        case 1,
            % compute the stopping criterion based on the relative
            % variation of the objective function.
            criterion = abs(f-prev_f)/prev_f;
        case 2,
            % compute the stopping criterion based on the relative
            % variation of the estimate.
            criterion = gather(norm(x(:)-xm1(:))/norm(x(:)));
        case 3,
            % continue if not yet reached target value tolA
            criterion = f;
        otherwise,
            error(['Unknwon stopping criterion']);
    end
    cont_outer = ((iter <= maxiter) & (criterion > tolA));
    if iter <= miniter
        cont_outer = 1;
    end



    iter = iter + 1;
    prev_f = f;
    objective(iter) = f;
    times(iter) = toc(t0);

    if compute_mse
        err = true - x;
        mses(iter) = gather(err(:)'*err(:));
    end

    % print out the various stopping criteria
    if verbose
        if plot_ISNR
            fprintf(1,'Iteration=%4d, ISNR=%4.5e  objective=%9.5e, nz=%7d, criterion=%7.3e\n',...
                iter, 10*log10(sum((y(:)-true(:)).^2)/sum((x(:)-true(:)).^2) ), ...
                f, num_nz_x, criterion/tolA);
        else
            fprintf(1,'Iteration=%4d, objective=%9.5e, nz=%7d,  criterion=%7.3e\n',...
                iter, f, num_nz_x, criterion/tolA);
        end
    end

end
%--------------------------------------------------------------
% end of the main loop
%--------------------------------------------------------------

% Printout results
if verbose
    fprintf(1,'\nFinished the main algorithm!\nResults:\n')
    fprintf(1,'||A x - y ||_2 = %10.3e\n',gather(resid(:)'*resid(:)))
    fprintf(1,'||x||_1 = %10.3e\n',gather(sum(abs(x(:)))))
    fprintf(1,'Objective function = %10.3e\n',f);
    fprintf(1,'Number of non-zero components = %d\n',num_nz_x);
    fprintf(1,'CPU time so far = %10.3e\n', times(iter));
    fprintf(1,'\n');
end


%--------------------------------------------------------------
% If the 'Debias' option is set to 1, we try to
% remove the bias from the l1 penalty, by applying CG to the
% least-squares problem obtained by omitting the l1 term
% and fixing the zero coefficients at zero.
%--------------------------------------------------------------
if debias
    if verbose
        fprintf(1,'\n')
        fprintf(1,'Starting the debiasing phase...\n\n')
    end

    x_debias = x;
    zeroind = (x_debias~=0);
    cont_debias_cg = 1;
    debias_start = iter;

    % calculate initial residual
    resid = A(x_debias);
    resid = resid-y;
    resid_prev = eps*ones(size(resid), 'like', resid);

    rvec = AT(resid);

    % mask out the zeros
    rvec = rvec .* zeroind;
    rTr_cg = gather(rvec(:)'*rvec(:));

    % set convergence threshold for the residual || RW x_debias - y ||_2
    tol_debias = tolD * (rvec(:)'*rvec(:));

    % initialize pvec
    pvec = -rvec;

    % main loop
    while cont_debias_cg

        % calculate A*p = Wt * Rt * R * W * pvec
        RWpvec = A(pvec);
        Apvec = AT(RWpvec);

        % mask out the zero terms
        Apvec = Apvec .* zeroind;

        % calculate alpha for CG
        alpha_cg = rTr_cg / gather(pvec(:)'* Apvec(:));

        % take the step
        x_debias = x_debias + alpha_cg * pvec;
        resid = resid + alpha_cg * RWpvec;
        rvec  = rvec  + alpha_cg * Apvec;

        rTr_cg_plus = gather(rvec(:)'*rvec(:));
        beta_cg = rTr_cg_plus / rTr_cg;
        pvec = -rvec + beta_cg * pvec;

        rTr_cg = rTr_cg_plus;

        iter = iter+1;

        objective(iter) = gather(0.5*(resid(:)'*resid(:)) + ...
            tau*phi_function(x_debias(:)));
        times(iter) = toc(t0);

        if compute_mse
            err = true - x_debias;
            mses(iter) = gather(err(:)'*err(:));
        end

        % in the debiasing CG phase, always use convergence criterion
        % based on the residual (this is standard for CG)
        if verbose
            fprintf(1,' Iter = %5d, debias resid = %13.8e, convergence = %8.3e\n', ...
                iter, gather(resid(:)'*resid(:)), gather(rTr_cg / tol_debias));
        end
        cont_debias_cg = ...
            (iter-debias_start <= miniter_debias )| ...
            ((gather(rTr_cg) > gather(tol_debias)) & ...
            (iter-debias_start <= maxiter_debias));

    end
    if verbose
        fprintf(1,'\nFinished the debiasing phase!\nResults:\n')
        fprintf(1,'||A x - y ||_2 = %10.3e\n',gather(resid(:)'*resid(:)))
        fprintf(1,'||x||_1 = %10.3e\n',gather(sum(abs(x(:)))))
        fprintf(1,'Objective function = %10.3e\n',f);
        nz = (x_debias~=0.0);
        fprintf(1,'Number of non-zero components = %d\n',gather(sum(nz(:))));
        fprintf(1,'CPU time so far = %10.3e\n', times(iter));
        fprintf(1,'\n');
    end
end

if compute_mse
    mses = mses/length(true(:));
end

end

%--------------------------------------------------------------
% soft thresholding for GPU arrays
%--------------------------------------------------------------
function y = soft_gpu(x,T)
    y = max(abs(x) - T, 0);
    y = y./(y+T) .* x;
end
