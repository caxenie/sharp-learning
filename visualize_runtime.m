% function to visualize network data at a given iteration during runtime
function visualize_runtime(input_data, populations)
set(gcf, 'color', 'w');
% population encoded inputs in each population of the network
subplot(3, 3, 1);
acth03 = plot(input_data.X, '.r', 'LineWidth', 2); box off;
xlabel('neuron index'); ylabel('population coded input in layer 1');
box off;%axis([0,  populations(1).lsize, 0, max(input_data.X)]);
subplot(3, 3, 2);
acth04 = plot(input_data.Y, '.b', 'LineWidth', 2); box off;
xlabel('neuron index'); ylabel('popultion coded input in layer 2');
box off;%axis([0,  populations(2).lsize, 0, max(input_data.X)]);
subplot(3, 3, 3);
acth05 = plot(input_data.Z, '.g', 'LineWidth', 2); box off;
xlabel('neuron index'); ylabel('population coded input in layer 3');
box off;%axis([0,  populations(3).lsize, 0, max(input_data.X)]);

% activities for each population (both overall activity and homeostasis)
max_act = max([max(populations(1).a), max(populations(2).a), max(populations(3).a)]);
subplot(3, 3, 4);
acth3 = plot(populations(1).a, '-r', 'LineWidth', 2); box off;
xlabel('neuron index'); ylabel('activation in layer 1');
box off;%axis([0,  populations(1).lsize, 0, max_act]);
subplot(3, 3, 5);
acth4 = plot(populations(2).a, '-b', 'LineWidth', 2); box off;
xlabel('neuron index'); ylabel('activation in layer 2');
box off;%axis([0,  populations(2).lsize, 0, max_act]);
subplot(3, 3, 6);
acth5 = plot(populations(3).a, '-g', 'LineWidth', 2); box off;
xlabel('neuron index'); ylabel('activation in layer 3');
box off;%axis([0,  populations(3).lsize, 0, max_act]);

% hebbian links between populations
hpc1 = subplot(3 ,3,7);
ax1=get(hpc1,'position'); % Save the position as ax
set(hpc1,'position',ax1); % Manually setting this holds the position with colorbar
vis_data4 = populations(2).Wext;
acth7 = imagesc(vis_data4); caxis([0, max(populations(2).Wext(:))]); colorbar;
box off; grid off; set(gca,'XAxisLocation','top');
xlabel('layer 1 - neuron index'); ylabel('layer 2 - neuron index');
hpc2 = subplot(3, 3, 8);
ax2=get(hpc2,'position'); % Save the position as ax
set(hpc2,'position',ax2); % Manually setting this holds the position with colorbar
vis_data5 = populations(3).Wext;
acth8 = imagesc(vis_data5);caxis([0, max(populations(3).Wext(:))]); colorbar;
box off; grid off; set(gca,'XAxisLocation','top');
xlabel('layer 2 - neuron index'); ylabel('layer 3 - neuron index');
hpc3 = subplot(3, 3, 9);
ax3=get(hpc3,'position'); % Save the position as ax
set(hpc3,'position',ax3); % Manually setting this holds the position with colorbar
vis_data3 = populations(1).Wext;
acth6= imagesc(vis_data3);caxis([0, max(populations(1).Wext(:))]); colorbar;
box off; grid off; set(gca,'XAxisLocation','top');
xlabel('layer 3 - neuron index'); ylabel('layer 1 - neuron index');

% refresh visualization
set(acth03, 'YData', input_data.X);
set(acth04, 'YData', input_data.Y);
set(acth05, 'YData', input_data.Z);
set(acth3, 'YData', populations(1).a);
set(acth4, 'YData', populations(2).a);
set(acth5, 'YData', populations(3).a);
set(acth6, 'CData', vis_data3);
set(acth7, 'CData', vis_data4);
set(acth8, 'CData', vis_data5);
drawnow;
end