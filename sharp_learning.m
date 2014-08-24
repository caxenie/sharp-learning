%% SIMPLE IMPLEMENTATION OF THE SHARP LEARNING ALGORITHM
% connectivity as shown in the paper example in Cook et. al - Sharp Learning
% CLAMP is selective and can be connected on each map which follow the structure A->B->C->A
% prepare environment
clear all; clc; close all;
%% INIT SIMULATION
% enables dynamic visualization on network runtime
DYN_VISUAL      = 1;
% verbose in standard output
VERBOSE         = 0;
% number of populations in the network
N_POP = 3;
% number of neurons in each population
N_NEURONS       = 200;
% max range value @ init for weights and activities in the population
MAX_INIT_RANGE  = 1;
% WTA circuit settling threshold
EPSILON         = 1e-3;
%% INIT NETWORK DYNAMICS
% epoch iterator in outer loop (HL, HAR)
t       = 1;
% network iterator in inner loop (WTA)
tau     = 1;
% constants for WTA circuit (convolution based WTA), these will provide a
% profile peaked at ~ TARGET_VAL_ACT
DELTA   = -0.005;                   % displacement of the convolutional kernel (neighborhood)
SIGMA   = 5.0;                      % standard deviation in the exponential update rule
SL      = 4.5;                      % scaling factor of neighborhood kernel
GAMMA   = SL/(SIGMA*sqrt(2*pi));    % convolution scaling factor
% constants for Hebbian linkage
ALPHA_L = 0.5*1e-2;                 % Hebbian learning rate
ALPHA_D = 1.0*1e-2;                 % Hebbian decay factor ALPHA_D >> ALPHA_L
% constants for HAR
C       = 0.005;                    % scaling factor in homeostatic activity regulation
TARGET_VAL_ACT  = 0.4;              % amplitude target for HAR
A_TARGET        = TARGET_VAL_ACT*ones(N_NEURONS, 1); % HAR target activity vector
% constants for neural units in neural populations
M       = 1; % slope in logistic function @ neuron level
S       = 5.5; % shift in logistic function @ neuron level
ETA     = 0.25; % weight decay of activity in each populations
%% CREATE NETWORK AND INITIALIZE
% create a network given the simulation constants
populations = create_init_network(N_POP, N_NEURONS, GAMMA, SIGMA, DELTA, MAX_INIT_RANGE, TARGET_VAL_ACT);
% buffers for changes in activity in WTA loop
act = zeros(N_NEURONS, N_POP);
old_act = zeros(N_NEURONS, N_POP);
% buffers for running average of population activities in HAR loop
old_avg = zeros(N_POP, N_NEURONS);
cur_avg = zeros(N_POP, N_NEURONS);
% the new rate values
delta_a1 = zeros(N_NEURONS, 1);
delta_a2 = zeros(N_NEURONS, 1);
delta_a3 = zeros(N_NEURONS, 1);
%% NETWORK SIMULATION LOOP
while(1)
    %% INPUT DATA
    % pick a new sample and feed it to the input (noiseless input)
    % population in the network (in this case X -> A -> B -> C -> A)
    input_data.X = population_encoder(t, N_NEURONS, N_NEURONS);
    % normalize input such that the activity in all units sums to 1.0
    %input_data.X = input_data.X./sum(input_data.X);
    % reinit the other populations with random activity in [0,1] and
    % normalize accross all units in the population
    input_data.Y = rand(N_NEURONS, 1)*MAX_INIT_RANGE;
    input_data.Z = rand(N_NEURONS, 1)*MAX_INIT_RANGE;
    % clamp input to neural population
    populations(1).a = input_data.X;
    populations(2).a = input_data.Y;
    populations(3).a = input_data.Z;
    %% MAIN LOOP WTA, HL and HAR
    % given the input sample wait for WTA circuit to settle and then
    % perform a learning step of Hebbian learning and HAR
    while(1)
        % for each neuron in first population, A
        % update the change in activity
        delta_a1 = compute_s(populations(1).h + ...
            populations(1).Wext*populations(3).a +...
            populations(1).Wint*populations(1).a , M, S);
        % for each neuron in second population, B
        % update the change in activity
        delta_a2 = compute_s(populations(2).h + ...
            populations(2).Wext*populations(1).a +...
            populations(2).Wint*populations(2).a , M, S);
        
        % for each neuron in third population, C
        % update the change in activity
        delta_a3 = compute_s(populations(3).h + ...
            populations(3).Wext*populations(2).a +...
            populations(3).Wint*populations(3).a , M, S);
        % update the activities of each population
        for pidx = 1:N_POP
            populations(pidx).a = (1-ETA)*populations(1).a + ETA*delta_a1;
            act(:, pidx) = populations(pidx).a;
        end
        % check if activity has settled in the WTA loop
        q = (sum(sum(abs(act - old_act)))/(N_POP*N_NEURONS));
        if(q <= EPSILON)
            if VERBOSE==1
                fprintf('WTA converged after %d iterations\n', tau);
            end
            tau = 1;
            break;
        end
        % update history of activities
        old_act = act;
        % increment time step in WTA loop
        tau = tau + 1;
    end  % WTA convergence loop
    % update Hebbian linkage between the populations (decaying Hebbian rule)
    % compute the changes in weights
    populations(1).Wext = (1-ALPHA_D)*populations(1).Wext + ...
        ALPHA_L*populations(1).a*populations(2).a';
    populations(2).Wext = (1-ALPHA_D)*populations(2).Wext + ...
        ALPHA_L*populations(2).a*populations(3).a';
    populations(3).Wext = (1-ALPHA_D)*populations(3).Wext + ...
        ALPHA_L*populations(3).a*populations(1).a';
    
    % compute the inverse time for exponential averaging of HAR activity
    omegat = 0.002 + 0.998/(t+2);
    % for each population in the network
    for pop_idx = 1:N_POP
        % update Homeostatic Activity Regulation terms
        % compute exponential average of each population at current step
        cur_avg(pop_idx, :) = (1-omegat)*old_avg(pop_idx, :) + omegat*populations(pop_idx).a';
        % update homeostatic activity terms given current and target act.
        populations(pop_idx).h = populations(pop_idx).h + C*(TARGET_VAL_ACT - cur_avg(pop_idx, :)');
    end
    % update averging history
    old_avg = cur_avg;
    % increment timestep for HL and HAR loop
    t = t + 1;
    % print epoch counter
    if VERBOSE==1
        fprintf('HL and HAR dynamics at iteration %d \n', t);
    end
    if(t==N_NEURONS)
        t = 1;
    end
    % visualize runtime data
    if(DYN_VISUAL==1)
        visualize_runtime(input_data, populations);
    end
end % end loop for individual sample presentation

