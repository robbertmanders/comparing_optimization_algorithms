%% OLS
clear
clc

iris = csvread('set3.csv',1,1);
%iris = csvread('C:\Users\Gebruiker\Dropbox\BAOR\Optimization\Project\simdata-2.csv',1,1);

rng(26)
m = size(iris,2) - 1;       % number of features
n = size(iris,1);           % total number of obs

iris = iris(randperm(size(iris,1)),:);

iris(:,1:m) = normalize(iris(:,1:m));
%iris(:,1:m) = (iris(:,1:m) - repmat(mean(iris(:,1:m)),[n 1]))./repmat(std(iris(:,1:m)),[n 1]);

iris(:,m+1) = iris(:,m+1) - mean(iris(:,m+1)) + randn(n,1)*5;

beta = sdpvar(m,1);

% 80% train, 10% val, 10% test
Xtrain = iris(1:round(0.8*n), 1:m);
ytrain = iris(1:round(0.8*n), m+1);

Xval = iris((round(0.8*n)+1):round(0.9*n), 1:m);
Yval = iris((round(0.8*n)+1):round(0.9*n), m+1);

Xtest = iris((round(0.9*n)+1):n, 1:m);
Ytest = iris((round(0.9*n)+1):n, m+1);

n = length(Xtrain);         %number of obs in training

%%
residuals = ytrain-Xtrain*beta;

bound = sdpvar(length(residuals),1);

Constraints = [-bound <= residuals <= bound];

opti = optimize(Constraints,sum(bound));

OLSbeta = value(beta);

%% Binary programming formulation

ops = sdpsettings('solver', 'gurobi');

tic
for o = 1:1
    M = 1000;
    K = 6 %o * (m/10);
    z = binvar(m,1);
    beta = sdpvar(m,1);
    
    objective = norm(ytrain - Xtrain*beta)^2;
    
    constraints = [sum(z) <= K];
    constraints = [constraints, -M * z <= beta <= M * z];
    
    %ops = sdpsettings('solver', 'gplx');
    optibin = optimize(constraints, objective, ops);
    
    Binarybeta = value(beta);
    Binaryscoreval(o) = mean((Yval - Xval * Binarybeta).^2);
    Binaryscore(o) = mean((Ytest - Xtest * Binarybeta).^2);
end

tijdtrainbin = toc;
%% Lambdas

lambda1 = [0, 0.001, 0.01 , 0.1, 1, 10];
lambda2 = [0, 0.001, 0.01 , 0.1, 1, 10];
func_tol = 0.001;
fun = @(beta) 1/(2*n) * norm(ytrain - Xtrain*beta)^2 + lambda2(w)/2 * norm(beta)^2 + lambda1(q) * norm(beta, 1);

%% Model Elastic

% ops = sdpsettings('solver', 'gurobi');

tic
for q = 1 : length(lambda1)
    for w = 1 : length(lambda2)
        beta = sdpvar(m,1);
        
        objective = 1/(2*n) * norm(ytrain - Xtrain*beta)^2 + lambda2(w)/2 * norm(beta)^2 + lambda1(q) * norm(beta, 1);
        
        optiel = optimize([], objective);
        
        Elasticbeta = value(beta);
        Elasticscoreval(q,w) = mean((Yval - Xval * Elasticbeta).^2);
        Elasticscore(q,w) = mean((Ytest - Xtest * Elasticbeta).^2);
 
    end
end

tijdtrainela = toc;
%% Model Pathwise
tic
for q = 1:length(lambda1)
    for w = 1:length(lambda2)
        fun = @(beta) 1/(2*n) * norm(ytrain - Xtrain*beta)^2 + lambda2(w)/2 * norm(beta)^2 + lambda1(q) * norm(beta, 1);
        beta = rand(m,1)*60;
        func_tol2 = 0.0000001;
        iter = 1;
        maxiter = 10000;
        
        while iter < maxiter
            improved = fun(beta);
            for j = 1:length(beta)
                ymachtding = Xtrain*beta - Xtrain(:,j)*beta(j);
                somding = sum(Xtrain(:,j) .* (ytrain - ymachtding))/n;
                if (somding > 0 && lambda1(q) < abs(somding))
                    S = somding - lambda1(q);
                elseif (somding < 0 && lambda1(q) < abs(somding))
                    S = somding + lambda1(q);
                else
                    S = 0;
                end
                beta(j) = max(S, 0) / (1 + lambda2(w));
            end
            iter = iter + 1;
            if abs(fun(beta) - improved) < func_tol2
                break
            end
        end
        Pathwisebeta = beta;
        Pathwisescoreval(q,w) = mean((Yval - Xval * Pathwisebeta).^2);
        Pathwisescore(q,w) = mean((Ytest - Xtest * Pathwisebeta).^2);
    end
end
tijdtrainpath = toc;

%% Model Gradient
tic
for q = 1 :length(lambda1)
    for w = 1 :length(lambda2)        
        % Gradient descent
        fun = @(beta) 1/(2*n) * norm(ytrain - Xtrain*beta)^2 + lambda2(w)/2 * norm(beta)^2 + lambda1(q) * norm(beta, 1);
        beta = rand(m,1)*60;
        grad = []; 
        for i = 1:length(beta)
            grad(i) = -Xtrain(:,i)'*(ytrain-Xtrain*beta)/n + lambda2(w) * beta(i) + sign(beta(i)) * lambda1(q);
        end
        d = -grad;
        
        %parameters
        iter = 1;
        gamma = 0.2;
        maxIter = 10000;
        
        fvals = [];
        improved = fun(beta);
        fvals(iter) = fun(beta);
        while iter < maxIter
            iter = iter + 1;
            beta1 = beta + gamma * d';           % gradient descent
            fvals(iter) = fun(beta1);   % evaluate objective function
            for i = 1:length(beta)
                grad(i) = -Xtrain(:,i)'*(ytrain-Xtrain*beta)/n + lambda2(w) * beta(i) + sign(beta(i)) * lambda1(q);
            end
            d = -grad;
            d = d/norm(d);
            beta = beta1;
%             fprintf('iter = %3d, val = %f \n', iter, fvals(end))
            if abs(fvals(end - 1) - fvals(end)) < func_tol
                break
            end
        end
        Gradientbeta = beta;
        Gradientscoreval(q,w) = mean((Yval - Xval * Gradientbeta).^2);
        Gradientscore(q,w) = mean((Ytest - Xtest * Gradientbeta).^2);
    end
end
tijdtraingrad = toc;
%% Optimal solutions
[xbin, ybin] = find(round(Binaryscoreval*100)/100==(min(min(round(Binaryscoreval*100)/100))));
[xgrad, ygrad] = find(Gradientscoreval==(min(min(Gradientscoreval))));
[xpath, ypath] = find(Pathwisescoreval==(min(min(Pathwisescoreval))));
[xelas, yelas] = find(Elasticscoreval==(min(min(Elasticscoreval))));

%% binary optimal solution
tic
M = 1000;
z = binvar(m,1);
Kbin = 6 %ybin(1) * (m/10)
beta = sdpvar(m,1);


objective = norm(ytrain - Xtrain*beta)^2;

constraints = [sum(z) <= Kbin];
constraints = [constraints, -M * z <= beta <= M * z];

optibin = optimize(constraints, objective);

Binarybetalatex = value(beta);
Binaryscorevallatex = mean((Yval - Xval * Binarybetalatex).^2);
Binaryscorelatex = mean((Ytest - Xtest * Binarybetalatex).^2);

bintijd = toc;
%% elastic
tic
lambda1_optel = lambda1(xelas);
lambda2_optel = lambda2(yelas);
fun = @(beta) 1/(2*n) * norm(ytrain - Xtrain*beta)^2 + lambda2_optel/2 * norm(beta)^2 + lambda1_optel * norm(beta, 1);

objective = 1/(2*n) * norm(ytrain - Xtrain*beta)^2 + lambda2_optel/2 * norm(beta)^2 + lambda1_optel * norm(beta, 1);

optiel = optimize([], objective);

Elasticbetalatex = value(beta);
Elasticscorevallatex = mean((Yval - Xval * Elasticbetalatex).^2);
Elasticscorelatex = mean((Ytest - Xtest * Elasticbetalatex).^2);

elasttijd = toc;
%% Pathwise
tic
lambda1_optpat = lambda1(xpath);
lambda2_optpat = lambda2(ypath);
beta = rand(m,1);
fun = @(beta) 1/(2*n) * norm(ytrain - Xtrain*beta)^2 + lambda2_optpat/2 * norm(beta)^2 + lambda1_optpat * norm(beta, 1);

iter = 1;
maxiter = 10000;

% sum(X.*(repmat(beta,[1,n])'),2)
while iter < maxiter
    improved = fun(beta);
    for j = 1:length(beta)
        ymachtding = Xtrain*beta - Xtrain(:,j)*beta(j);
        somding = sum(Xtrain(:,j) .* (ytrain - ymachtding))/n;
        if (somding > 0 && lambda1_optpat < abs(somding))
            S = somding - lambda1_optpat;
        elseif (somding < 0 && lambda1_optpat < abs(somding))
            S = somding + lambda1_optpat;
        else
            S = 0;
        end
        beta(j) = max(S, 0) / (1 + lambda2_optpat);
    end
    iter = iter + 1;
    if abs(fun(beta) - improved) < func_tol2
        break
    end
end

Pathwisebetalatex = beta;
Pathwisescorevallatex = mean((Yval - Xval * Pathwisebetalatex).^2);
Pathwisescorelatex = mean((Ytest - Xtest * Pathwisebetalatex).^2);

pathtijd = toc;
%% Gradient descent
tic 
lambda1_optgrad = lambda1(xgrad);
lambda2_optgrad = lambda2(ygrad);
fun = @(beta) 1/(2*n) * norm(ytrain - Xtrain*beta)^2 + lambda2_optgrad/2 * norm(beta)^2 + lambda1_optgrad * norm(beta, 1);
beta = rand(m,1);

grad = [];

for i = 1:length(beta)
    grad(i) = -Xtrain(:,i)'*(ytrain-Xtrain*beta)/n + lambda2_optgrad * beta(i) + sign(beta(i)) * lambda1_optgrad;
end

d = -grad;

%parameters
iter = 1;
gamma = 0.2;
maxIter = 10000;

fvals = [];
improved = fun(beta);
fvals(iter) = fun(beta);
while iter < maxIter
    iter = iter + 1;
    beta1 = beta + gamma * d';          % gradient descent
    fvals(iter) = fun(beta1);           % evaluate objective function
    for i = 1:length(beta)
        grad(i) = -Xtrain(:,i)'*(ytrain-Xtrain*beta)/n + lambda2_optgrad * beta(i) + sign(beta(i)) * lambda1_optgrad;
    end
    d = -grad;
    d = d/norm(d);
    beta = beta1;
%     fprintf('iter = %3d, val = %f \n', iter, fvals(end))
    if abs(fvals(end - 1) - fvals(end)) < func_tol
        break
    end
end
Gradientbetalatex = beta;
Gradientscorevallatex = mean((Yval - Xval * Gradientbetalatex).^2);
Gradientscorelatex = mean((Ytest - Xtest * Gradientbetalatex).^2);

gradtijd = toc;
%% latex tables data
%opt value betas and MSE
BetaLatex = [Elasticbetalatex, Gradientbetalatex, Pathwisebetalatex, Binarybetalatex];
MetricesLatex = [Elasticscorevallatex, Gradientscorevallatex,Pathwisescorevallatex, Binaryscorevallatex; Elasticscorelatex, Gradientscorelatex,Pathwisescorelatex, Binaryscorelatex; tijdtrainela, tijdtraingrad, tijdtrainpath, tijdtrainbin]; 

databin = [Binaryscorevallatex, Binaryscorelatex, tijdtrainbin, 0, 0, Kbin, sum((round(Binarybetalatex*10)/10) == 0)];
datael = [Elasticscorevallatex, Elasticscorelatex, tijdtrainela, lambda1_optel, lambda2_optel, 0, sum((round(Elasticbetalatex*1)/1) == 0)];
datapat = [Pathwisescorevallatex, Pathwisescorelatex, tijdtrainpath, lambda1_optpat, lambda2_optpat, 0, sum((round(Pathwisebetalatex*1)/1) == 0)];
datagrad = [Gradientscorevallatex, Gradientscorelatex, tijdtraingrad, lambda1_optgrad, lambda2_optgrad, 0, sum((round(Gradientbetalatex*1)/1) == 0)];

output = [datael; datagrad; datapat; databin];
