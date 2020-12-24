% Quadratic Programming
% This script gives an example of using the quadratic programming
% function in MATLAB

%% Example 1
% min   (0.4*x1^2) - (5*x1) + (x2^2) - (6*x2) + 50
% s.t. 
%       x2 - x1 >= 2
%       0.3*x1 + x2 >= 8
%       0 <= x1 <= 10
%       0 <= x2 <= 10

% Lower and upper bounds for the decision variables
LB = [0 0]';
UB = [10 10]';

%% Plot Objective Function, Constraints, and Solution
% Create the meshgrid and cost function plot
[x1, x2] = meshgrid(LB(1):0.1:UB(1), LB(2):0.1:UB(2));
costPlot = (0.4*x1.^2) - (5*x1) + (x2.^2) - (6*x2) + 50;

figure(1)
contour(x1, x2, costPlot, 20, 'LineWidth', 2)
xlabel('x_1'); ylabel('x_2'); zlabel('cost')
hold on
plot([0;8], [2;10], 'k', 'LineWidth', 2)
plot([0;10], [8;5], 'k', 'LineWidth', 2)

H = 2*[0.4 0; 0 1];
f = [-5 -6]';
A = [1 -1; -0.3 -1];
b = [-2 -8]';

[X, J] = quadprog(H,f,A,b,[],[],LB,UB)

plot(X(1), X(2), 'r.', 'MarkerSize', 30)