% simple implementation of the sharp learning algorithm
% connectivity as shown in the paper example in Cook et. all - Sharp Learning
% CLAMP is selctive and can be connected on each A->B->C->A

% prepare environment
clear all; clc; close all;

%% INIT NETWORK
% enables dynamic visualization on network runtime
DYN_VISUAL = 1;
% number of populations in the network
N_POP = 3;
% number of neurons in each population
N_NEURONS = 40;
% max range value @ init for weights and activities in the population
MAX_INIT_RANGE = 0.1 ;
% WTA circuit settling threshold
EPSILON = 1e-12;
% type of input data real (sensory data) or generated
REAL_DATA = 0;
% the type of activity update - one shot (as given in the paper) or
% progressive using a gradient descent towards the correct values
PROGRESSIVE_UPDATE = 0;
% clamp the input to a unit given by CLAMP id
CLAMP = 1;

% constants in network dynamics

% learning rate for activation update if PROGRESSIVE_UPDATE = 1
ETA = 0.005;

% constants for WTA circuit
GAMMA = 1; % scaling factor of weight update dynamics
DELTA = 0.0; % decay term of weight update dyanamics
SIGMA = 0.85; % standard deviation in the exponential update rule

% constants for Hebbian linkage
ALPHA_L = 0.1; % regulates the speed of connection learning
ALPHA_D = 0.5; % weight decay factor ALPHA_D > ALPHA_L

% constants for HAR
C = 1; % scaling factor in homeostatic activity regulation
A_TARGET = 1.0; % target activity for HAR
OMEGA = 0.5;  % inverse time constant of averaging

% constants for neural units in populations
M = 1/(N_NEURONS/2); % slope in logistic function @ neuron level
S = 0; % shift in logistic function @ neuron level

%% INIT INPUT DATA
if(REAL_DATA==1)
    sensory_data = sensory_data_setup('robot_data_jras_paper', 'tracker_data_jras_paper');
    % size of the input dataset
    MAX_EPOCHS = length(sensory_data.timeunits);
    STEP_SIZE = 10;
else
    sensory_data = load('artificial_algebraic_data.mat');
    MAX_EPOCHS = length(sensory_data.x);
    STEP_SIZE = 1;
end
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
    'h', rand(N_NEURONS, 1)*MAX_INIT_RANGE);
populations(2) = struct(  'idx', 2,...
    'lsize', N_NEURONS, ...
    'Wint', rand(N_NEURONS, N_NEURONS)*MAX_INIT_RANGE, ...
    'Wext', rand(N_NEURONS, N_NEURONS)*MAX_INIT_RANGE, ...
    'a', rand(N_NEURONS, 1)*MAX_INIT_RANGE,...
    'h', rand(N_NEURONS, 1)*MAX_INIT_RANGE);
populations(3) = struct(  'idx', 3,...
    'lsize', N_NEURONS, ...
    'Wint', rand(N_NEURONS, N_NEURONS)*MAX_INIT_RANGE, ...
    'Wext', rand(N_NEURONS, N_NEURONS)*MAX_INIT_RANGE, ...
    'a', rand(N_NEURONS, 1)*MAX_INIT_RANGE,...
    'h', rand(N_NEURONS, 1)*MAX_INIT_RANGE);

% changes in activity
delta_a = zeros(N_POP, N_NEURONS);
old_delta_a = zeros(N_POP, N_NEURONS);
% running average of population activities
old_avg = zeros(N_POP, N_NEURONS);
cur_avg = zeros(N_POP, N_NEURONS);

%% NETWORK SIMULATION LOOP
for t = 1:STEP_SIZE:MAX_EPOCHS
    
    % pick a new sample from the dataset and feed it to the input
    % population in the network (in this case in->A->B)
    if(REAL_DATA==1)
        % first input is gyroscope data
        populations(CLAMP).a = population_encoder(sensory_data.heading.gyro(t)*pi/180, N_NEURONS);
    else
        populations(CLAMP).a = population_encoder(sensory_data.x(t), N_NEURONS);
    end
       
    % given the input sample wait for WTA circuit to settle
    while(1)
        %-----------------------------------------------------------------------------------------------
        
        % update the weights in the WTA circuits in each population
        populations(1).Wint = GAMMA*compute_d(N_NEURONS, SIGMA) - DELTA;
        populations(2).Wint = GAMMA*compute_d(N_NEURONS, SIGMA) - DELTA;
        populations(3).Wint = GAMMA*compute_d(N_NEURONS, SIGMA) - DELTA;
        
        % neural units update dynamics (activity)
        if(PROGRESSIVE_UPDATE==1)
            delta_a(1, :) = (populations(1).h + ...
                populations(1).Wint*populations(1).a + ...
                populations(1).Wext*populations(3).a);
            
            delta_a(2, :) = (populations(2).h + ...
                populations(2).Wint*populations(2).a + ...
                populations(2).Wext*populations(1).a);
            
            delta_a(3, :) = (populations(3).h + ...
                populations(3).Wint*populations(3).a + ...
                populations(3).Wext*populations(2).a);
            
            populations(1).a = populations(1).a + ETA*delta_a(1, :)';
            populations(2).a = populations(2).a + ETA*delta_a(2, :)';
            populations(3).a = populations(3).a + ETA*delta_a(3, :)';
            
            populations(1).a = compute_s(populations(1).a, M, S);
            populations(2).a = compute_s(populations(2).a, M, S);
            populations(3).a = compute_s(populations(3).a, M, S);
            
        else
            populations(1).a = compute_s(populations(1).h + ...
                populations(1).Wint*populations(1).a + ...
                populations(1).Wext*populations(3).a, M, S);
            
            populations(2).a = compute_s(populations(2).h + ...
                populations(2).Wint*populations(2).a + ...
                populations(2).Wext*populations(1).a, M, S);
            
            populations(3).a = compute_s(populations(3).h + ...
                populations(3).Wint*populations(3).a + ...
                populations(3).Wext*populations(2).a, M, S);
            
            delta_a(1, :) = populations(1).a;
            delta_a(2, :) = populations(2).a;
            delta_a(3, :) = populations(3).a;
        end
        
        % check if network has settled
        if(sum((old_delta_a - delta_a).^2)<=EPSILON)
            fprintf('network has settled in %d iterations\n', tau);
            tau = 0;
            break;
        end
        % update history
        tau = tau + 1;
        old_delta_a = delta_a;
    
        %-----------------------------------------------------------------------------------------------
  
    end  % WTA convergence loop
    
    % update Hebbian linkage

    populations(1).Wext = (1-ALPHA_D)*populations(1).Wext + ...
        ALPHA_L*populations(3).a*populations(1).a';
    
    populations(2).Wext = (1-ALPHA_D)*populations(2).Wext + ...
        ALPHA_L*populations(1).a*populations(2).a';
    
    populations(3).Wext = (1-ALPHA_D)*populations(3).Wext + ...
        ALPHA_L*populations(2).a*populations(3).a';
    
    % update Homeostatic Activity Regulation terms
    % compute running average of each population at current step t
    cur_avg(1, :) = (1-OMEGA)*old_avg(1, :) + OMEGA*populations(1).a';
    cur_avg(2, :) = (1-OMEGA)*old_avg(2, :) + OMEGA*populations(2).a';
    cur_avg(3, :) = (1-OMEGA)*old_avg(3, :) + OMEGA*populations(3).a';
    
    % update homeostatic activity terms
    populations(1).h = -C*(cur_avg(1, :)' - A_TARGET*ones(N_NEURONS, 1));
    populations(2).h = -C*(cur_avg(2, :)' - A_TARGET*ones(N_NEURONS, 1));
    populations(3).h = -C*(cur_avg(3, :)' - A_TARGET*ones(N_NEURONS, 1));
    
    % update averging history
    old_avg = cur_avg;
    fprintf('training epoch %d\n', t);
  
    
      % visualize encoding process
    if(DYN_VISUAL==1)
        set(gcf, 'color', 'white');
        % input
        subplot(3, 3, 1);
        acth0 = plot(population_encoder(sensory_data.x(t), N_NEURONS), '-r', 'LineWidth', 2); box off;
        xlabel('neuron index'); ylabel('input value encoded in population');
        % lateral connectivity within population 
        subplot(3, 3, 2);
        vis_data1 = populations(1).Wint;
        acth1 = pcolor(vis_data1);
        box off; grid off; axis xy;
        xlabel('layer - neuron index'); ylabel('layer - neuron index');
        subplot(3, 3, 3);
        vis_data2 = populations(1).Wint;
        acth2 = surf(vis_data2);
        box off; grid off; axis xy;
        xlabel('layer - neuron index'); ylabel('layer - neuron index');
        zlabel('connection strength');
        % activities for each population (both overall activity and homeostasis)
        subplot(3, 3, 4);
        acth3 = plot(populations(1).a, '-r', 'LineWidth', 2); box off;
        axis([0 N_NEURONS 0 1]);
        xlabel('neuron index'); ylabel('activation in layer 1');
        subplot(3, 3, 5);
        acth4 = plot(populations(2).a, '-b', 'LineWidth', 2); box off;
        axis([0 N_NEURONS 0 1]);
        xlabel('neuron index'); ylabel('activation in layer 2');
        subplot(3, 3, 6);
        acth5 = plot(populations(3).a, '-g', 'LineWidth', 2); box off;
        axis([0 N_NEURONS 0 1]);
        % hebbian links between populations
        subplot(3, 3, 7);
        vis_data3 = populations(1).Wext;
        acth6= pcolor(vis_data3);
        box off; grid off; axis xy;
        xlabel('layer 1 - neuron index'); ylabel('layer 2 - neuron index');
        subplot(3, 3, 8);
        vis_data4 = populations(2).Wext;
        acth7 = pcolor(vis_data4);
        box off; grid off; axis xy;
        xlabel('layer 2 - neuron index'); ylabel('layer 3 - neuron index');
        subplot(3, 3, 9);
        vis_data5 = populations(3).Wext;
        acth8 = pcolor(vis_data5);
        box off; grid off; axis xy;
        xlabel('layer 3 - neuron index'); ylabel('layer 1 - neuron index');
        
        % refresh visualization
        set(acth0, 'YData', population_encoder(sensory_data.x(t), N_NEURONS));
        set(acth1, 'CData', vis_data1);
        set(acth2, 'CData', vis_data2);
        set(acth3, 'YData', populations(1).a);
        set(acth4, 'YData', populations(2).a);
        set(acth5, 'YData', populations(3).a);
        set(acth6, 'CData', vis_data3);
        set(acth7, 'CData', vis_data4);
        set(acth8, 'CData', vis_data5);
        drawnow;
    end
    
    %------------------------------------------------------------------------------------------------
    
end % end main relaxation loop for WTA, HL and HAR

%% VISUALIZATION
% weights after learning
figure; set(gcf, 'color', 'white');
subplot(1,3,1);
pcolor(populations(1).Wext);
box off; grid off; axis xy; xlabel('layer 1'); ylabel('layer 2');
subplot(1,3,2);
pcolor(populations(2).Wext);
box off; grid off; axis xy; xlabel('layer 2'); ylabel('layer 3');
subplot(1,3,3);
pcolor(populations(3).Wext);
box off; grid off; axis xy; xlabel('layer 3'); ylabel('layer 1');






