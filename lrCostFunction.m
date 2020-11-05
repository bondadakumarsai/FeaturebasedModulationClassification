function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

J = 0;
grad = zeros(size(theta));

    z = X * theta;
    
    g = sigmoid(z);
    
    J = (-1/m)* (sum(y.*log(g) + (1-y).*log(1-g)) - (lambda*(sum(theta.^2)-theta(1)^2))/2);
    
    grad = (1/m)* ((X'*(g-y))+lambda*theta);
    grad(1) = grad(1) - (lambda*theta(1)/m);

% =============================================================

grad = grad(:);

end
