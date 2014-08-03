% simple implementation of the sharp learning algorithm

% prepare environment
clear all; clc; close all;

%% INIT NETWORK
% enables dynamic visualization on network runtime
DYN_VISUAL = 0;
% number of populations in the network
N_POP = 2;
% number of neurons in each population
N_NEURONS = 100;
% max range value @ init for weights and activities in the population
MAX_INIT_RANGE = 0;
% WTA circuit settling threshold
EPSILON = 1e-12;

% constants in network dynamics

% constants for WTA circuit
GAMMA = 0.5; % scaling factor of weight update dynamics
DELTA = 0.1; % decay term of weight update dyanamics
SIGMA = 0.5; % standard deviation in the exponential update rule

% constants for Hebbian linkage
ALPHA_L = 0.3; % regulates the speed of connection learning
ALPHA_D = 0.7; % weight decay factor ALPHA_D > ALPHA_L

% constants for HAR
C = 0.3; % scaling factor in homeostatic activity regulation
A_TARGET = 1; % target activity for HAR
OMEGA = 0.5;  % inverse time constant of averaging

% constants for neural units in populations
M = 0.4; % slope in logistic function @ neuron level
S = 0.6; % bias in logistic function @ neuron level

%% INIT INPUT DATA
sensory_data = sensory_data_setup('robot_data_jras_paper', 'tracker_data_jras_paper');
% size of the input dataset
MAX_EPOCHS = length(sensory_data.timeunits);
% epoch iterator (iterator through the input dataset)
t = 1;
% network iterator (iterator for a given input value)
tau = 1;

%% CREATE NETWORK AND INITIALIZE
populations(1) = struct(  'idx', 1,...
    'lsize', N_NEURONS, ...
    'Wint', rand(N_NEURONS, N_NEURONS)*MAX_INIT_RANGE, ...
    'Wext', rand(N_NEURONS, N_NEURONS)*MAX_INIT_RANGE, ...
    'a', rand(N_NEURONS, 1)*MAX_INIT_RANGE,...
    'h', zeros(N_NEURONS, 1));
populations(2) = struct(  'idx', 2,...
    'lsize', N_NEURONS, ...
    'Wint', rand(N_NEURONS, N_NEURONS)*MAX_INIT_RANGE, ...
    'Wext', rand(N_NEURONS, N_NEURONS)*MAX_INIT_RANGE, ...
    'a', rand(N_NEURONS, 1)*MAX_INIT_RANGE,...
    'h', zeros(N_NEURONS, 1));

% changes in activity
delta_a = zeros(N_POP, N_NEURONS);
old_delta_a = zeros(N_POP, N_NEURONS);
% running average of population activities
old_avg = zeros(N_POP, N_NEURONS);
cur_avg = zeros(N_POP, N_NEURONS);

%% NETWORK SIMULATION LOOP
for t = 1:10
    
    % reinit activity randomly
    populations(1).a = rand(N_NEURONS, 1)*MAX_INIT_RANGE; % receives ext in
    populations(2).a = rand(N_NEURONS, 1)*MAX_INIT_RANGE;
    
    % pick a new sample from the dataset and feed it to the input
    % population in the network (in this case in->A->B)
    populations(1).a = population_encoder(sensory_data.heading.gyro(t)*pi/180, N_NEURONS);
    
    % normalize activity between [0,1]
    populations(1).a = populations(1).a./sum(populations(1).a);
    
    %-----------------------------------------------------------------------------------------------
    % visualize encoding process
    if(DYN_VISUAL==1)
        set(gcf, 'color', 'white');
        % input
        subplot(3,2,1);
        acth1 = plot(populations(1).a, '-r', 'LineWidth', 2); box off;
        xlabel('neuron index'); ylabel('activation in layer 1');
        subplot(3,2,2);
        acth2 = plot(populations(2).a, '-b','LineWidth', 2); box off;
        xlabel('neuron index'); ylabel('activation in layer 2');
        
        % weights - internal (within population)
        subplot(3,2,3);
        vis_data1 = populations(1).Wint(1:N_NEURONS, 1:N_NEURONS)';
        acth4 = pcolor(vis_data1); %colorbar;
        box off; grid off; axis xy; xlabel('layer 1 - neuron index'); ylabel('layer 2 - neuron index');
        subplot(3,2,4);
        vis_data2 = populations(2).Wint(1:N_NEURONS, 1:N_NEURONS)';
        acth5 = pcolor(vis_data2); %colorbar;
        box off; grid off; axis xy; xlabel('layer 2 - neuron index'); ylabel('layer 1 - neuron index');
        
        % weights - between (between population)
        subplot(3,2,5);
        vis_data3 = populations(1).Wext(1:N_NEURONS, 1:N_NEURONS)';
        acth6 = pcolor(vis_data1); %colorbar;
        box off; grid off; axis xy; xlabel('layer 1 - neuron index'); ylabel('layer 1 - neuron index');
        subplot(3,2,6);
        vis_data4 = populations(2).Wext(1:N_NEURONS, 1:N_NEURONS)';
        acth7 = pcolor(vis_data2); %colorbar;
        box off; grid off; axis xy; xlabel('layer 2 - neuron index'); ylabel('layer 2 - neuron index');
        
       
        % refresh visualization
        set(acth1, 'YDataSource', 'populations(1).a');
        set(acth2, 'YDataSource', 'populations(2).a');
        set(acth4, 'CData', vis_data1);
        set(acth5, 'CData', vis_data2);
        set(acth6, 'CData', vis_data3);
        set(acth7, 'CData', vis_data4);
        refreshdata(acth1, 'caller');
        refreshdata(acth2, 'caller');
        refreshdata(acth4, 'caller');
        refreshdata(acth5, 'caller');
        refreshdata(acth6, 'caller');
        refreshdata(acth7, 'caller');
        drawnow;
    end
    
    %------------------------------------------------------------------------------------------------
       
    % given the input sample wait for WTA circuit to settle
    while(1)
        % update the weights in the WTA circuits in each population
        populations(1).Wint = GAMMA*compute_d(N_NEURONS, SIGMA) - DELTA;
        populations(2).Wint = GAMMA*compute_d(N_NEURONS, SIGMA) - DELTA;
        
        % neural units update dynamics (activity)
        populations(1).a = compute_s(populations(1).h + ...
            populations(1).Wint*populations(1).a + ...
            populations(1).Wext*populations(1).a, M, S);
        populations(2).a = compute_s(populations(2).h + ...
            populations(2).Wint*populations(2).a + ...
            populations(2).Wext*populations(2).a, M, S);
        
        % compute the change in units activity in each population
        delta_a(1, :) = populations(1).a;
        delta_a(2, :) = populations(2).a;
        
        % check if network has settled
        if(sum((old_delta_a(1, :) - delta_a(1,:)).^2)<=EPSILON && ...
           sum((old_delta_a(2, :) - delta_a(2,:)).^2)<=EPSILON)
            fprintf('network has settled in %d iterations\n', tau);
            tau = 0;
            break;
        end
        % update history
        tau = tau + 1;
        old_delta_a = delta_a;
    end
    
    % update Hebbian linkage
    populations(1).Wext = (1-ALPHA_D)*populations(1).Wext + ...
        ALPHA_L*populations(1).a*populations(2).a';
    
    populations(2).Wext = (1-ALPHA_D)*populations(2).Wext + ...
        ALPHA_L*populations(2).a*populations(1).a';
    
    % update Homeostatic Activity Regulation terms
    % compute running average of each population at current step t
    cur_avg(1, :) = (1-OMEGA)*old_avg(1, :) + OMEGA*populations(1).a';
    cur_avg(2, :) = (1-OMEGA)*old_avg(2, :) + OMEGA*populations(2).a';
    
    % update homeostatic activity terms
    populations(1).h = -C*(cur_avg(1, :)' - A_TARGET*ones(N_NEURONS, 1));
    populations(2).h = -C*(cur_avg(2, :)' - A_TARGET*ones(N_NEURONS, 1));
    
    % update averging history
    old_avg = cur_avg;
    fprintf('training epoch %d\n', t);
end % end simulation for each sample in the input dataset
%% VISUALIZATION
% weights after learning
figure; set(gcf, 'color', 'white');
subplot(1,2,1);
pcolor(populations(1).Wext'); colorbar;
box off; grid off; axis xy; xlabel('input layer'); ylabel('projection layer');
subplot(1,2,2);
pcolor(populations(2).Wext'); colorbar;
box off; grid off; axis xy; xlabel('projection layer'); ylabel('input layer');






