%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% ECON 634 Macro II
%%% Problem Set 4
%%% the Aiyagari Model
%%% Yaobin Wang
%%% 11/08/2017
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Question 1 
%%% FOCs for the Firm
%%% r_t = alpha*[K_t^(alpha-1)]*[N_t^(1-alpha)]+(1-delta);
%%% w_t = (1-alpha)*(K_t^alpha)*[N_t^(-alpha)];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Question 2 
%%% Houldholds' Recursive Problem
%%% v(z,a) = max_{a'? Gamma(z,a)} u(z*w+r*a-a')+beta*E_{z'|z}[v(z',a')],
%%% where u(c) = c^(1-sigma)/(1-sigma), return function;
%%%       ln(z_t+1) = rho*ln(z_t)+epsilon_t;
%%%       epsilon ~ N(0,(sigma_epsilon)^2);
%%%
%%% State Variables:
%%% exogenous employment status z, AR-1 in logs
%%% current period asset a
%%%
%%% Control Variable: next period assst a'
%%%
%%% State Space: Z*A
%%%
%%% Constraint Correspondence: Gamma(z,a)= [a_lo, z*w+r*a]

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Question 3
%%% Discretize the exogeneous state variable z into a grid w/ m points
clear all; close all; clc;
% parameters
alpha = 1/3;
beta = 0.99;
sigma = 2;
delta = 0.025;
a_lo = 0;
rho = 0.5;
sigma_e = 0.2;
% discretize z using Tauchen's Method
m = 5; % number of grid points
[ln_z, PI] = TAUCHEN(m,rho,sigma_e,3); % ln(z)-grid and transition matrix
PI_inv = PI^100; % the invariant distribution
prob = PI_inv(1,:);
z = exp(ln_z); % z-grid
N_s = prob*z; % aggregate effective labor supply
N_d = N_s; %labor market clearing

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Question 4
%%% Discretize the endogenous state variable a into a grid w/ n points
n = 500; % number of grid points
a_min = a_lo;
a_max = 100; % guess and verify later so that a_max is not binding
a = linspace(a_min, a_max, n); % asset (row) vector

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Question 5
%%% Solving the model numerically
K_min = 0;
K_max = 100;
dis1 = 1;
tol1 = 0.01;
tic;
% iterate over aggregate capital stock
while abs(dis1)>= tol1
    % initial guess for K
    K_guess = (K_max + K_min)/2;
    % factor prices
    r = alpha*K_guess^(alpha-1)*N_d^(1-alpha)+(1-delta);
    w = (1-alpha)*(K_guess^alpha)*N_d^(-alpha);
    % current return (utility) function
    cons = bsxfun(@plus,bsxfun(@minus,r*a',a),permute(w*z', [1 3 2]));
    ret = (cons .^ (1-sigma)) ./ (1 - sigma); % current period utility
    ret (cons < 0) = -Inf; % negative consumption is impossible
    % initial value function guess
    v_guess = zeros(m,n);
    % value function iteration
    dis2 = 1;
    tol2 = 1e-06;
    while dis2 > tol2
        % CONSTRUCT RETURN + EXPECTED CONTINUATION VALUE (n*n*m)
        v = ret + beta * repmat(permute((PI*v_guess),[3 2 1]),[n 1 1]);        
        % CHOOSE HIGHEST VALUE (ASSOCIATED WITH a' CHOICE)
        [vfn, p_indx] = max(v, [], 2); % vfn n*1*m
        % Distance between current guess and value function
        dis2 = max(max(abs(permute(vfn, [3 1 2])-v_guess)));        
        % if dis1 > tol, update guess. O.w. exit.
        v_guess = permute(vfn, [3 1 2]);
    end
    % keep decision rule
    pol_indx = permute(p_indx,[3,1,2]);
    pol_fn = a(pol_indx); % pol_fn m*n;
    % set up initial distribution (uniform)
    Mu = ones(m,n)/(m*n);
    % iterate over distribution
    dis3 = 1;
    tol3 = 1e-06;
    while dis3 >= tol3
        [emp_ind, a_ind, mass] = find(Mu); % find non-zero indices
        MuNew = zeros(size(Mu));
        for ii = 1:length(emp_ind)
            % which a prime does the policy fn prescribe?
            apr_ind = pol_indx(emp_ind(ii), a_ind(ii)); 
            % which mass of households goes to which exogenous state?
            MuNew(:, apr_ind) = MuNew(:, apr_ind) + ...
                (PI(emp_ind(ii), :) * mass(ii))';
        end
        dis3 = max(max(abs(MuNew-Mu)));
        Mu = MuNew;
    end
    %check for market clearing
    aggsav = sum(sum(Mu.*pol_fn));
    dis1 = aggsav - K_guess;
    % adjust K_guess
    if dis1 > 0
        K_min = K_guess;
    else
        K_max = K_guess;
    end
    display(['K_guess = ', num2str(K_guess)]);
    display(['dis1 = ', num2str(dis1)]);
    display(['new K_guess is ', num2str((K_max+K_min)/2)]);
end
time = toc;
K = K_guess;
display(['The steady state capital stock is ',num2str(K)]);
display(['Total time of solving the model with VFI is ', num2str(time)]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Question 6
%%% Analyze the results

% interest rates
display(['The steady state interest rate is ', num2str(r)]);
r_CM = 1/beta;
display(['The interest rate in complete market is ', num2str(r_CM)]);
%%% The steady state interest rate in the Aiyagari model is slightly lower
%%% than the interest rate in the compelet market case.

% Policy Functions
figure
plot(a,pol_fn)
legend('z = 0.5002','z = 0.7072','z = 1','z = 1.4140', 'z = 1.9993',...
    'Location','NorthWest')
xlabel('a')
ylabel('policy function (a'')')
title('The Policy Functions for the Five Productivity States')

% Lorenz Curve and Gini Coeff. for Wealth
d = reshape(Mu',[m*n 1]);
wealth = reshape(repmat(a, [m 1])',[m*n 1]);
d_wealth = cumsum(sortrows([d,d.*wealth,wealth],3));
G_wealth = bsxfun(@rdivide,d_wealth,d_wealth(end,:));
L_wealth = G_wealth*100;
Gini_wealth = 1-(sum((G_wealth(1:end-1,2)+G_wealth(2:end,2)).*...
    diff(G_wealth(:,1))));
display(['The Gini coefficient for wealth is ', num2str(Gini_wealth)]);
figure
plot(L_wealth(:,1),L_wealth(:,2),L_wealth(:,1),L_wealth(:,1),'--k')
legend('Lorenz Curve','45 degree line','Location','NorthWest')
xlabel('Cumulative Share of Population')
ylabel('Cumulative Share of Wealth')
title('Lorenz Curve for Wealt   h')

% Wealth/Asset Distribution
figure
plot(a,Mu(1,:),a,Mu(2,:),a,Mu(3,:),a,Mu(4,:),a,Mu(5,:))
legend('z = 0.5002','z = 0.7072','z = 1','z = 1.4140', 'z = 1.9993')
xlabel('Wealth')
ylabel('share of the population')
title('Wealth Distribution')