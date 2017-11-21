%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Question 7
%%% Policy Function Iteration
clear all; close all; clc;
alpha = 1/3;
beta = 0.99;
sigma = 2;
delta = 0.025;
a_lo = 0;
rho = 0.5;
sigma_e = 0.2;
k = 30; % number of PFI

m = 5;
[ln_z, PI] = TAUCHEN(m,rho,sigma_e,3);
PI_inv = PI^100;
prob = PI_inv(1,:);
z = exp(ln_z);
N_s = prob*z;
N_d = N_s;

n = 500;
a_min = a_lo;
a_max = 100;
a = linspace(a_min, a_max, n);

K_min = 0;
K_max = 100;
dis1 = 1;
tol1 = 0.01;
tic;
while abs(dis1)>= tol1
    K_guess = (K_max + K_min)/2;
    r = alpha*K_guess^(alpha-1)*N_d^(1-alpha)+(1-delta);
    w = (1-alpha)*(K_guess^alpha)*N_d^(-alpha);
    cons = bsxfun(@plus,bsxfun(@minus,r*a',a),permute(w*z', [1 3 2]));
    ret = (cons .^ (1-sigma)) ./ (1 - sigma);
    ret (cons < 0) = -Inf;
    v_guess = zeros(m,n);
    
    dis2 = 1;
    tol2 = 1e-06;
    while dis2 > tol2
        v = ret + beta * repmat(permute((PI*v_guess),[3 2 1]),[n 1 1]);        
        [vfn, p_indx] = max(v, [], 2);
        pol_indx = permute(p_indx,[3,1,2]);
        dis2 = max(max(abs(permute(vfn, [3 1 2])-v_guess)));        
        v_guess = permute(vfn, [3 1 2]);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PFI
        % construct Q matrix from policy index and PI
        Q_mat = makeQmatrix(pol_indx,PI);
        
        % construct return vector and value vector
        pol_fn = a(pol_indx); % pol_fn m*n;
        u_mat = bsxfun(@minus, r*a, pol_fn);
        u_mat = bsxfun(@plus, u_mat, z*w);
        u_mat = (u_mat.^(1-sigma))./(1-sigma);
        u_vec = u_mat(:);
        
        w_vec = v_guess(:);
        
        %PFI
        for ii = 1:k
            w_vec_new = u_vec+beta*Q_mat*w_vec;
            w_vec = w_vec_new;
        end
        
        v_guess = reshape(w_vec, m, n);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
    end
    
    Mu = ones(m,n)/(m*n);
    dis3 = 1;
    tol3 = 1e-06;
    while dis3 >= tol3
        [emp_ind, a_ind, mass] = find(Mu);
        MuNew = zeros(size(Mu));
        for ii = 1:length(emp_ind)
            apr_ind = pol_indx(emp_ind(ii), a_ind(ii)); 
            MuNew(:, apr_ind) = MuNew(:, apr_ind) + ...
                (PI(emp_ind(ii), :) * mass(ii))';
        end
        dis3 = max(max(abs(MuNew-Mu)));
        Mu = MuNew;
    end

    aggsav = sum(sum(Mu.*pol_fn));
    dis1 = aggsav - K_guess;

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
display(['Total time of solving the model with PFI is ', num2str(time)]);
% PFI helps to boost up the program

% interest rates
display(['The steady state interest rate is ', num2str(r)]);
r_CM = 1/beta;
display(['The interest rate in complete market is ', num2str(r_CM)]);

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
title('Lorenz Curve for Wealth')

% Wealth/Asset Distribution
figure
plot(a,Mu(1,:),a,Mu(2,:),a,Mu(3,:),a,Mu(4,:),a,Mu(5,:))
legend('z = 0.5002','z = 0.7072','z = 1','z = 1.4140', 'z = 1.9993')
xlabel('Wealth')
ylabel('share of the population')
title('Wealth Distribution')