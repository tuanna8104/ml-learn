function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

for i = 1:m;
	input_data = [1, X(i,:)];
	input_data = input_data';
	z_first_layer = Theta1*input_data;
	output_first_layer = sigmoid(z_first_layer);
	input_second_layer = [1; output_first_layer];
	z_second_layer = Theta2*input_second_layer;
	output_second_layer = sigmoid(z_second_layer);
	origin_labels = zeros(num_labels, 1);
	output = y(i);
	origin_labels(output) = 1;
	labels = origin_labels';
	deltaJ = (0-labels)*log(output_second_layer) - (1 - labels)*(log(1 - output_second_layer));
	J += deltaJ;
	theta = output_second_layer - origin_labels;
	Theta2_grad = Theta2_grad + theta*input_second_layer';
	theta_hidden = (Theta2'*theta);
	theta_hidden = theta_hidden(2:end);
	theta_hidden = theta_hidden.*sigmoidGradient(z_first_layer);	
	Theta1_grad = Theta1_grad + theta_hidden*input_data';
endfor

neutral_theta1 = Theta1(:, 2:end);
neutral_theta2 = Theta2(:, 2:end);
J = (J + (sum(sum(neutral_theta1.^2)) + sum(sum(neutral_theta2.^2)))*lambda/2)/m;
Theta2_grad = Theta2_grad./m + (lambda/m)*[zeros(rows(neutral_theta2), 1), neutral_theta2];
Theta1_grad = Theta1_grad./m + (lambda/m)*[zeros(rows(neutral_theta1), 1), neutral_theta1];

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%



% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
