function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
featureSize = length(theta);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
temp_sigmoid = sigmoid(X*theta);
J =  sum((0-y).*log(temp_sigmoid) - (1-y).*log(1-temp_sigmoid))/m + lambda*sum(theta(2:featureSize).^2)/(2*m);
for i = 2:featureSize
	grad(i) = ((temp_sigmoid-y)'*X(:,i))/m + lambda*theta(i)/m;
grad(1) = ((temp_sigmoid-y)'*X(:,1))/m;




% =============================================================

end
