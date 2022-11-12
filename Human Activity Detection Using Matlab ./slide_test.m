clc;clear all;clear;
format longg;

%Initializing parameters

% Detection window length
windowLength = 5;
% Number of windows between detections
detectionInterval = 1;
% resampling bcz phn data isnt sampled
uniformSampleRate = 50; % Hz. 


%%
%Step 4: load recorded data and uniformly resample it
testWalk = timehandeling('run.mat');
a = testWalk(:,1:3);
t = testWalk(:,4);
% Resampling the raw data to obtain uniformly sampled acceleration data
newTime = 0:1/50:(t(end)-t(1));
x = a(:,1);
y = a(:,2);
z = a(:,3);
x = interp1(t,x,newTime);
y = interp1(t,y,newTime);
z = interp1(t,z,newTime);
a = [x;y;z]';
t = newTime;

%%Generate a plot of raw data
figure;
p1=plot(t,x,'b');hold on;
p2=plot(t,y,'g');
p3=plot(t,z,'r');hold off;
ylim([-30 20]);

h=[p1;p2;p3];
legend(h,'X-axis','Y-axis','Z-axis');
