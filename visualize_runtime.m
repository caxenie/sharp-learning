% function to visualize network data at a given iteration during runtime
function visualize_runtime(populations, tau, t)
set(gcf, 'color', 'w');
suptitle(sprintf('Network dynamics: tau = %d (WTA) | t = %d (HL, HAR)', tau, t));

% activities for each population (both overall activity and homeostasis)
subplot(2, 3, 1);
acth3 = plot(populations(1).a, '-r', 'LineWidth', 2); box off;
xlabel('neuron index'); ylabel('activation in layer 1');
axis([0, populations(1).lsize, 0, 1]);
subplot(2, 3, 2);
acth4 = plot(populations(2).a, '-b', 'LineWidth', 2); box off;
xlabel('neuron index'); ylabel('activation in layer 2');
axis([0, populations(2).lsize, 0, 1]);
subplot(2, 3, 3);
acth5 = plot(populations(3).a, '-g', 'LineWidth', 2); box off;
xlabel('neuron index'); ylabel('activation in layer 3');
axis([0, populations(3).lsize, 0, 1]);

% hebbian links between populations
subplot(2,3,4);
vis_data4 = populations(2).Wext;
acth7 = pcolor(vis_data4);
box off; grid off;
xlabel('layer 1 - neuron index'); ylabel('layer 2 - neuron index');
subplot(2, 3, 5);
vis_data5 = populations(3).Wext;
acth8 = pcolor(vis_data5);
box off; grid off;
xlabel('layer 2 - neuron index'); ylabel('layer 3 - neuron index');
subplot(2, 3, 6);
vis_data3 = populations(1).Wext;
acth6= pcolor(vis_data3);
box off; grid off;
xlabel('layer 3 - neuron index'); ylabel('layer 1 - neuron index');

% refresh visualization
set(acth3, 'YData', populations(1).a);
set(acth4, 'YData', populations(2).a);
set(acth5, 'YData', populations(3).a);
set(acth6, 'CData', vis_data3);
set(acth7, 'CData', vis_data4);
set(acth8, 'CData', vis_data5);
drawnow;
end