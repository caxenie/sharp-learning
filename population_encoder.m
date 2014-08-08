% function to generate the population encoded variable as input for the net
function R = population_encoder(x, maxX, N)
sig = 5.0; % standard deviation 
v = 0; % spontaneous activity of a neuron
C = 1; % presentation constant 
K = 1; % max firing rate (Hz) (ignore - not modeling nurophysiology here :)
% pattern of activity, or output tuning curve
R = zeros(N, 1);     
% calculate output 
for j = 1:N % for each neuron in the population
    % for Poisson generator use this and maxX will be 2*pi
    % temp = cos( x - (2*pi*j/N) ) - 1 ;
    % otherwise
    temp = -(x - maxX*j/N)^2;
    % fj is the lamda value for poisson Neuron j
    fj = C * (K*exp(temp / sig^2) + v); 
    % for Poisson generator use this 
    % R(j) = poissrnd(fj);
    % otherwise output
    R(j) = fj;
end

    
    
    
    