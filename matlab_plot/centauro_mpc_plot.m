%% MPC loop
close all
clc

mpc_iter = length(F1_mpc);

time = zeros(1,length(F1_mpc));
time(1) = 0;
for j=1:mpc_iter-1
    time(j+1) = time(j) + Dt_ipopt(1);
end

% forces
fig=figure;
fig.Units='centimeters';
fig.Position=[10 10 40 30];
subplot(2,2,1);
plot(time, F1_mpc(:,:),'LineWidth', 1.5)
ylim([-100,1000]); xlim([time(1) time(end)]);
set(gca,'TickLabelInterpreter','latex');grid on
ylabel('$N$','Interpreter','latex'); xlabel('time [s]','Interpreter','latex');
legend('x','y','z','Interpreter','latex');
title('F1 (front left)','Interpreter','latex');
subplot(2,2,2);
plot(time, F2_mpc(:,:),'LineWidth', 1.5)
ylim([-100,1000]); xlim([time(1) time(end)]);
set(gca,'TickLabelInterpreter','latex');grid on
ylabel('$N$','Interpreter','latex'); xlabel('time [s]','Interpreter','latex');
legend('x','y','z','Interpreter','latex');
title('F2 (front right)','Interpreter','latex');
subplot(2,2,3);
plot(time, F4_mpc(:,:),'LineWidth', 1.5)
ylim([-100,1000]); xlim([time(1) time(end)]);
set(gca,'TickLabelInterpreter','latex');grid on
ylabel('$N$','Interpreter','latex'); xlabel('time [s]','Interpreter','latex');
legend('x','y','z','Interpreter','latex');
title('F4 (hind left)','Interpreter','latex');
subplot(2,2,4);
plot(time, F3_mpc(:,:),'LineWidth', 1.5)
ylim([-100,1000]); xlim([time(1) time(end)]);
set(gca,'TickLabelInterpreter','latex');grid on
ylabel('$N$','Interpreter','latex'); xlabel('time [s]','Interpreter','latex');
legend('x','y','z','Interpreter','latex');
title('F3 (hind right)','Interpreter','latex');

% contact positions
fig=figure;
fig.Units='centimeters';
fig.Position=[10 10 40 30];
subplot(2,2,1);
plot(time, C1_mpc(:,:),'LineWidth', 1.5)
xlim([time(1) time(end)]);
set(gca,'TickLabelInterpreter','latex');grid on
ylabel('$m$','Interpreter','latex'); xlabel('time [s]','Interpreter','latex');
legend('x','y','z','Interpreter','latex');
title('C1 (front left)','Interpreter','latex');
subplot(2,2,2);
plot(time, C2_mpc(:,:),'LineWidth', 1.5)
xlim([time(1) time(end)]);
set(gca,'TickLabelInterpreter','latex');grid on
ylabel('$m$','Interpreter','latex'); xlabel('time [s]','Interpreter','latex');
legend('x','y','z','Interpreter','latex');
title('C2 (front right)','Interpreter','latex');
subplot(2,2,3);
plot(time, C4_mpc(:,:),'LineWidth', 1.5)
xlim([time(1) time(end)]);
set(gca,'TickLabelInterpreter','latex');grid on
ylabel('$m$','Interpreter','latex'); xlabel('time [s]','Interpreter','latex');
legend('x','y','z','Interpreter','latex');
title('C4 (hind left)','Interpreter','latex');
subplot(2,2,4);
plot(time, C3_mpc(:,:),'LineWidth', 1.5)
xlim([time(1) time(end)]);
set(gca,'TickLabelInterpreter','latex');grid on
ylabel('$m$','Interpreter','latex'); xlabel('time [s]','Interpreter','latex');
legend('x','y','z','Interpreter','latex');
title('C3 (hind right)','Interpreter','latex');

% floating base joints
fig=figure;
fig.Units='centimeters';
fig.Position=[10 10 40 30];
subplot(2,2,1);
plot(time, Q_mpc(:,1:7),'LineWidth', 1.5)
xlim([time(1) time(end)]);
set(gca,'TickLabelInterpreter','latex');grid on
xlabel('time [s]','Interpreter','latex');
title('q floating base','Interpreter','latex');
subplot(2,2,2);
plot(time, Qdot_mpc(:,1:6),'LineWidth', 1.5)
xlim([time(1) time(end)]);
set(gca,'TickLabelInterpreter','latex');grid on
xlabel('time [s]','Interpreter','latex');
title('qdot floating base','Interpreter','latex');
subplot(2,2,3);
plot(time, Qddot_mpc(:,1:6),'LineWidth', 1.5)
xlim([time(1) time(end)]);
set(gca,'TickLabelInterpreter','latex');grid on
xlabel('time [s]','Interpreter','latex');
title('qddot floating base','Interpreter','latex');
subplot(2,2,4);
plot(time, Tau_mpc(:,1:6),'LineWidth', 1.5)
ylim([-10, 10]); xlim([time(1) time(end)]);
set(gca,'TickLabelInterpreter','latex');grid on
xlabel('time [s]','Interpreter','latex');
title('tau floating base','Interpreter','latex');

% actuated joints
fig=figure;
fig.Units='centimeters';
fig.Position=[10 10 40 30];
subplot(2,2,1);
plot(time, Q_mpc(:,8:end),'LineWidth', 1.5)
xlim([time(1) time(end)]);
set(gca,'TickLabelInterpreter','latex');grid on
xlabel('time [s]','Interpreter','latex');
title('q joints','Interpreter','latex');
subplot(2,2,2);
plot(time, Qdot_mpc(:,7:end),'LineWidth', 1.5)
xlim([time(1) time(end)]);
set(gca,'TickLabelInterpreter','latex');grid on
xlabel('time [s]','Interpreter','latex');
title('qdot joints','Interpreter','latex');
subplot(2,2,3);
plot(time, Qddot_mpc(:,7:end),'LineWidth', 1.5)
xlim([time(1) time(end)]);
set(gca,'TickLabelInterpreter','latex');grid on
xlabel('time [s]','Interpreter','latex');
title('qddot joints','Interpreter','latex');
subplot(2,2,4);
plot(time, Tau_mpc(:,7:end),'LineWidth', 1.5)
xlim([time(1) time(end)]);
set(gca,'TickLabelInterpreter','latex');grid on
xlabel('time [s]','Interpreter','latex');
title('tau joints','Interpreter','latex');

% ground projection Cartesian quantities
fig=figure;
fig.Units='centimeters';
fig.Position=[10 10 30 30];
h1 = plot(CoM_mpc(:,1), CoM_mpc(:,2),'LineWidth', 1.);
hold on
h2 = plot(Waist_pos_mpc(:,1), Waist_pos_mpc(:,2),'LineWidth', 1.);
hold on
h3 = plot(C1_mpc(:,1), C1_mpc(:,2), 'LineWidth', 1.);
hold on
h4 = plot(C2_mpc(:,1), C2_mpc(:,2), 'o', 'MarkerSize', 15, 'LineWidth', 1.);
set(h4, 'markerfacecolor', get(h4, 'color'));
hold on
h5 = plot(C3_mpc(:,1), C3_mpc(:,2), 'LineWidth', 1.5);
hold on
h6 = plot(C4_mpc(:,1), C4_mpc(:,2), 'o', 'MarkerSize', 15, 'LineWidth', 1.);
set(h6, 'markerfacecolor', get(h6, 'color'));
hold on
h7 = plot(CoM_mpc(1,1), CoM_mpc(1,2), 'o');
set(h7, 'markerfacecolor', get(h1, 'color'), 'MarkerEdgeColor', get(h1, 'color'));
hold on
h8 = plot(Waist_pos_mpc(1,1), Waist_pos_mpc(1,2), 'o');
set(h8, 'markerfacecolor', get(h2, 'color'), 'MarkerEdgeColor', get(h2, 'color'));
hold on
h9 = plot(C1_mpc(1,1), C1_mpc(1,2), 'o');
set(h9, 'markerfacecolor', get(h3, 'color'), 'MarkerEdgeColor', get(h3, 'color'));
hold on
h10 = plot(C3_mpc(1,1), C3_mpc(1,2), 'o');
set(h10, 'markerfacecolor', get(h5, 'color'), 'MarkerEdgeColor', get(h5, 'color'));
xlim([-0.6 0.6]); ylim([-0.6 0.6]);
set(gca,'xtick', [-0.6:0.1:0.6]);
set(gca,'ytick', [-0.6:0.1:0.6]);
set(gca,'TickLabelInterpreter','latex');grid on
legend('CoM','waist','C1','C2','C3', 'C4','Interpreter','latex');
xlabel('x [m]','Interpreter','latex');
ylabel('y [m]','Interpreter','latex');
title('ground projection of Cartesian quantities','Interpreter','latex');

% waist pose history
fig=figure;
fig.Units='centimeters';
fig.Position=[10 10 30 30];
plot(time, Waist_pos_mpc, 'LineWidth', 1.5)
hold on
plot(time, Waist_rot_mpc, 'LineWidth', 1.5)
legend('x','y','z','roll','pitch','yaw','Interpreter','latex');
set(gca,'TickLabelInterpreter','latex');grid on
xlabel('time [s]','Interpreter','latex');
title('waist pose history','Interpreter','latex');

% elapsed time
fig=figure;
fig.Units='centimeters';
fig.Position=[10 10 30 30];
plot(time, elapsed_mpc, 'LineWidth', 1.5)
xlim([time(1) time(end)]);
set(gca,'TickLabelInterpreter','latex');grid on
xlabel('time [s]','Interpreter','latex');
title('elapsed time mpc','Interpreter','latex');



