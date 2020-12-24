% Nonlinear Programming 
% This script is an example for nonlinear programming in MATLAB

%% Example 1
% min   sin(x1) + (0.1*x2^2) + (0.05*x1^2)
% s.t. 
%       -5 <= x1 <= 1
%       -3 <= x2 <= 3
%       (x1+3)^3 - x2 = 0

% Lower and upper bounds for the decision variables
LB = [-5 -3]';
UB = [1 3]';

% Linear inequality and equality constraints 
% for the solver
A = [];     B = [];
Aeq = [];   Beq = [];

xInitial = [-3 -3]';
X = zeros(2,3);
cost = zeros(1,3);

% Solve the optimization problem for the 3 nonlinear constraints in the 
% example video
options = optimoptions('fmincon', 'Algorithm', 'sqp', 'Display',...
    'iter-detailed', 'MaxFunctionEvaluations', 100000, 'MaxIterations',...
    2000, 'FunctionTolerance', 1e-10);
[X(:,1), cost(1)] = fmincon(@(x) obj_function1(x), xInitial, A, B, Aeq, Beq, LB,...
    UB, @(x) nonlinear1(x), options);
[X(:,2), cost(2)] = fmincon(@(x) obj_function1(x), xInitial, A, B, Aeq, Beq, LB,...
    UB, @(x) nonlinear2(x), options);
[X(:,3), cost(3)] = fmincon(@(x) obj_function1(x), xInitial, A, B, Aeq, Beq, LB,...
    UB, @(x) nonlinear3(x), options);

% Print out the minima found
X
cost

% Results of running the script
% X =
%    -1.9960   -1.9960   -1.4276
%     1.0121    1.0121   -0.0000
% 
% cost =
%    -0.6093   -0.6093   -0.8879


%% Plot Objective Function, Constraints, and Solution
% Create the meshgrid and cost function plot
[x1, x2] = meshgrid(LB(1):0.1:UB(1), LB(2):0.1:UB(2));
costPlot = sin(x1) + (0.1*x2.^2) + (0.05*x1.^2);

% Range and constraint
x = x1(1,:);
y = (x+3).^3;

figure(1)
contour(x1, x2, costPlot, 20, 'LineWidth', 2)
xlabel('x_1'); ylabel('x_2'); zlabel('cost')
hold on
plot(x, y, 'k', 'LineWidth', 2)
xlim([LB(1) UB(1)])
ylim([LB(2) UB(2)])


plot(X(1,1), X(2,1), 'r.', 'MarkerSize', 30)
plot(X(1,2), X(2,2), 'b.', 'MarkerSize', 30)
plot(X(1,3), X(2,3), 'c.', 'MarkerSize', 30)
legend('Cost','(x_1 + 3)^3 - x_2 = 0', 'Location', 'southeast')
hold off


%% Helper functions
% These are all implemented in separate files for easy usage in MATLAB 
% versions older than R2016b

% Objective Function to optimize
function obj = obj_function1(x)
    obj = sin(x(1)) + (0.1*x(2)^2) + (0.05*x(1)^2);
end

% Equality Constraint (x1+3)^3 - x2 = 0
function [C, Ceq] = nonlinear1(x)
    C = [];    
    Ceq = (x(1)+3)^3 - x(2);
end

% Inequality Constraint (x1+3)^3 - x2 <= 0
function [C, Ceq] = nonlinear2(x)
    Ceq = [];    
    C = ((x(1)+3)^3 - x(2));
end

% Inequality Constraint (x1+3)^3 - x2 >= 0
function [C, Ceq] = nonlinear3(x)
    Ceq = [];    
    C = -((x(1)+3)^3 - x(2));
end