% simple implementation of the sharp learning algorithm
% connectivity as shown in the paper example in Cook et. al - Sharp Learning
% CLAMP is selective and can be connected on each A->B->C->A

% prepare environment
clear all; clc; close all;

%% INIT SIMULATION
% enables dynamic visualization on network runtime
DYN_VISUAL = 1;
% verbose in standard output
VERBOSE = 1;
% number of populations in the network
N_POP = 3;
% number of neurons in each population
N_NEURONS = 50;
% max range value @ init for weights and activities in the population
MAX_INIT_RANGE = 1;
% WTA circuit settling threshold
EPSILON = 1e-4;
% clamp the input to a unit given by CLAMP id
CLAMP =1;
% numbers of epochs to go through each values (add a smarter criteria here)
MAX_EPOCHS = 2000;

%% INIT INPUT DATA
sensory_data = load('artificial_data_set.mat');
DATASET_LEN = length(sensory_data.x);
% epoch iterator (iterator through the input dataset)
t = 1;
% network iterator (iterator for a given input value in the WTA loop)
tau = 1;

%% INIT NETWORK DYNAMICS
% constants for WTA circuit (convolution based WTA)
DELTA = -0.005; % displacement of the convolutional kernel (neighborhood)
SIGMA = 5.0; % standard deviation in the exponential update rule
SL = 4.5; % scaling factor of neighborhood kernel
GAMMA = SL/(SIGMA*sqrt(2*pi)); % convolution scaling factor

% constants for Hebbian linkage
ALPHA_L = 0.05*1e-3; % Hebbian learning rate
ALPHA_D = 1.0*1e-3; % Hebbian decay factor ALPHA_D >> ALPHA_L

% constants for HAR
C = 6.0; % scaling factor in homeostatic activity regulation
TARGET_VAL_ACT = 0.4; % amplitude target for HAR
A_TARGET = TARGET_VAL_ACT*ones(N_NEURONS, 1); % HAR target activity vector
omegat = zeros(MAX_EPOCHS, 1); % inverse time for activity averaging

% constants for neural units in neural populations
M = 1.0; % slope in logistic function @ neuron level
S = 1.55; % shift in logistic function @ neuron level

%% CREATE NETWORK AND INITIALIZE
% create a network given the simulation constants
populations = create_init_network(N_POP, N_NEURONS, GAMMA, SIGMA, DELTA, MAX_INIT_RANGE, TARGET_VAL_ACT);

% buffers for changes in activity in WTA loop
delta_a = zeros(N_POP, N_NEURONS)*MAX_INIT_RANGE;
old_delta_a = zeros(N_POP, N_NEURONS)*MAX_INIT_RANGE;
% buffers for running average of population activities in HAR loop
old_avg = zeros(N_POP, N_NEURONS);
cur_avg = zeros(N_POP, N_NEURONS);

%% NETWORK SIMULATION LOOP
% go through each entry in the dataset and present it for MAX_EPOCHS times
% maybe add a stoping condition based on some criteria
for didx =1:DATASET_LEN
    % run the sharp learning network
    while(1)
        
        % reinit randomly the activities in each population except the clamped
        for pop_idx = 1:N_POP
            if(pop_idx~=CLAMP)
                populations(pop_idx).a = rand(N_NEURONS, 1)*MAX_INIT_RANGE;
            end
        end
        
        % pick a new sample from the dataset and feed it to the input
        % population in the network (in this case in->A->B->C->A)
        populations(CLAMP).a = population_encoder(sensory_data.x(didx), max(sensory_data.x(:)),  N_NEURONS);
        
        % given the input sample wait for WTA circuit to settle
        while(1)
            % neural units activity update for each population given the
            % network's cycllic connectivity A->B->C->A
            populations(1).a = compute_s(populations(1).h + ...
                populations(1).Wint*populations(1).a + ...
                populations(1).Wext*populations(3).a, M, S);
            
            populations(2).a = compute_s(populations(2).h + ...
                populations(2).Wint*populations(2).a + ...
                populations(2).Wext*populations(1).a, M, S);
            
            populations(3).a = compute_s(populations(3).h + ...
                populations(3).Wint*populations(3).a + ...
                populations(3).Wext*populations(2).a, M, S);
            
            % current activation values for threshold check
            for pop_idx = 1:N_POP
                delta_a(pop_idx, :) = populations(pop_idx).a;
            end
            
            % check if activity has settled in the WTA loop
            if((sum(sum(abs(delta_a - old_delta_a)))/(N_POP*N_NEURONS))<EPSILON)
                if VERBOSE==1
                    fprintf('Network converged after %d iterations\n', tau);
                end
                tau = 0;
                break;
            end
            
            % update history of activities
            old_delta_a = delta_a;
            % update WTA loop time step
            tau = tau + 1;
            
            % visualize runtime data for each input sample
            if(DYN_VISUAL==1)
                visualize_runtime(populations, tau, t);
            end
            
            
        end  % WTA convergence loop
        
        % update Hebbian linkage between the populations
        populations(1).Wext = (1-ALPHA_D)*populations(1).Wext + ...
            ALPHA_L*populations(3).a*populations(1).a';
        
        populations(2).Wext = (1-ALPHA_D)*populations(2).Wext + ...
            ALPHA_L*populations(1).a*populations(2).a';
        
        populations(3).Wext = (1-ALPHA_D)*populations(3).Wext + ...
            ALPHA_L*populations(2).a*populations(3).a';
        
        % compute the inverse time for exponential averaging of HAR activity
        omegat(t) = 0.002 + 0.998/(t+2);
        
        % for each population in the network do HL normalization and HAR update
        for pop_idx = 1:N_POP
            % perform Hebbian weight normalization
            populations(pop_idx).Wext = populations(pop_idx).Wext./sum(populations(pop_idx).Wext(:));
            
            % update Homeostatic Activity Regulation terms
            % compute exponential average of each population at current step
            cur_avg(pop_idx, :) = (1-omegat(t))*old_avg(pop_idx, :) + omegat(t)*populations(pop_idx).a';
            % update HAR terms
            populations(pop_idx).h = -C*(cur_avg(pop_idx, :)' - A_TARGET);
        end
        
        % update HAR averging history
        old_avg = cur_avg;
        
        % check net end condition / criteria given the current input sample
        if(t == MAX_EPOCHS)
            if VERBOSE==1
                fprintf('Sharp learning training epochs %d\n', t);
            end
            t = 0;
        end
        
        % increment network time
        t = t + 1;
        
    end % end main relaxation loop for WTA, HL and HAR
    
end % end loop through all dataset items

% visualize runtime data and end of simulation for entire dataset
if(DYN_VISUAL==1)
    visualize_runtime(populations, tau, t);
end