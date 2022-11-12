clc;clear all;clear;
format longg;
warning off;

%Initializing parameters

% Detection window length
windowLength = 5;
% Number of windows between detections
detectionInterval = 1;
% resampling bcz phn data isnt sampled
uniformSampleRate = 50; % Hz. 


%Extract features from observation data

fileWalk = timehandeling('walk1.mat'); 
featureWalk = extractTrainingFeature(fileWalk,windowLength,uniformSampleRate);
% 
fileRun = timehandeling('run1.mat'); 
featureRun = extractTrainingFeature(fileRun,windowLength,uniformSampleRate);

fileIdle = timehandeling('idle1.mat'); 
featureIdle = extractTrainingFeature(fileIdle,windowLength,uniformSampleRate);

fileUp = timehandeling('climb1.mat'); 
featureUp = extractTrainingFeature(fileUp,windowLength,uniformSampleRate);

fileDown = timehandeling('down1.mat'); 
featureDown = extractTrainingFeature(fileDown,windowLength,uniformSampleRate);


%%
%Step 2: Normalize training data

data = [featureWalk; featureRun; featureIdle; featureUp; featureDown];

for i = 1:size(data,2)
    range(1,i) = max(data(:,i))-min(data(:,i));
    dMin(1,i) = min(data(:,i));
    data(:,i) = (data(:,i)- dMin(i)) / range(i);
end

%%
%Step 2: Activity indexing
indexIdle =  0;
indexWalk =  2;
indexDown = -1;
indexRun  =  3;
indexUp   =  1;

Idle = indexIdle * zeros(length(featureIdle),1);
Walk = indexWalk * ones(length(featureWalk),1);
Down = indexDown * ones(length(featureDown),1);
Run  = indexRun  * ones(length(featureRun),1);
Up   = indexUp   * ones(length(featureUp),1);


%% preprocessing
X = data;
Y = [Walk;Run;Idle;Up;Down];

%% shuffling
ir=randperm(numel(Y)); % since y is 1D; otherwise use size(y,1)
Y=Y(ir);
X=X(ir,:);
data_new=[X Y];
%Dividing data into train & test set
cv = cvpartition(size(data,1),'HoldOut',0.15);
idx = cv.test;
% Separate to training and test data
dataTrain = data_new(~idx,:);
xtrain=dataTrain(:,1:end-1);
ytrain=dataTrain(:,end);
dataTest  = data_new  (idx,:);
xtest=dataTest(:,1:end-1);
ytest=dataTest(:,end);

% finding k
accuracy=[];
for i=1:40
    mdl=fitcknn(xtrain,ytrain);
    mdl.NumNeighbors=i;
    mk=predict(mdl,xtest);
    accurancy_check=(sum(mk==ytest)/size(dataTest,1))*100;
    accuracy=[accuracy accurancy_check];
end


plot([1:40],accuracy,'-o','LineWidth',2,'Color',[.6 0 0]) 
title('accuracy vs. K Value')
xlabel('K')
ylabel('Accuracy')
[val,pos]=max(accuracy)
%disp("Maximum accuracy:- %f",val,"at K = %f",pos)
print = sprintf('Maximum accuracy: %f at %f',val,pos);
disp(print)

%% confusion matrix

mdl.NumNeighbors=10;
y_pred=predict(mdl,xtest);
fig=figure;
Ac_test=num2cell(ytest);
Ac_test(ytest==0) = {'Idle'};
Ac_test(ytest==2) = {'Walk'};
Ac_test(ytest==-1) = {'Down'};
Ac_test(ytest==3) = {'Run'};
Ac_test(ytest==1) = {'Up'};

Ac_pred=num2cell(y_pred);
Ac_pred(y_pred==0) = {'Idle'};
Ac_pred(y_pred==2) = {'Walk'};
Ac_pred(y_pred==-1) = {'Down'};
Ac_pred(y_pred==3) = {'Run'};
Ac_pred(y_pred==1) = {'Up'};

cm = confusionchart(Ac_test,Ac_pred,'RowSummary','row-normalized','ColumnSummary','column-normalized');
cm.Title='Human Activity Detection using KNN';
%cm.ClassLabels=["Walk","Run","Idle","Up","Down"]
fig_Position = fig.Position;
fig_Position(3) = fig_Position(3)*1.5;
fig.Position = fig_Position;
cm.Normalization = 'row-normalized'; 
sortClasses(cm,'descending-diagonal')
cm.Normalization = 'absolute';

[accuracy,precision,recall,f1_score]=acc_metric(ytest,y_pred)

%% Random forest
%To make predictions on a new predictor column matrix, X, use:   yfit = c.predictFcn(X) replacing 'c' with the name of the variable that is this struct, e.g. 'trainedModel'.  X must contain exactly 6 columns because this model was trained using 6 predictors. X must contain only predictor columns in exactly the same order and format as your training data. Do not include the response column or any columns you did not import into the app.  For more information, see <a href="matlab:helpview(fullfile(docroot, 'stats', 'stats.map'), 'appclassification_exportmodeltoworkspace')">How to predict using an exported model</a>.
% y_pred_forest=treeModel.predictFcn(xtest);
% fig=figure;
% 
% Ac_pred_tree=num2cell(y_pred_forest);
% Ac_pred_tree(y_pred_forest==0) = {'Idle'};
% Ac_pred_tree(y_pred_forest==2) = {'Walk'};
% Ac_pred_tree(y_pred_forest==-1) = {'Down'};
% Ac_pred_tree(y_pred_forest==3) = {'Run'};
% Ac_pred_tree(y_pred_forest==1) = {'Up'};
% 
% cm2 = confusionchart(Ac_test,Ac_pred_tree,'RowSummary','row-normalized','ColumnSummary','column-normalized');
% cm2.Title='Human Activity Detection using KNN';
% %cm.ClassLabels=["Walk","Run","Idle","Up","Down"]
% fig_Position = fig.Position;
% fig_Position(3) = fig_Position(3)*1.5;
% fig.Position = fig_Position;
% cm2.Normalization = 'row-normalized'; 
% sortClasses(cm2,'descending-diagonal')
% cm2.Normalization = 'absolute';
% [accuracy2,precision2,recall2,f1_score2]=acc_metric(ytest,y_pred_forest)
% 
% %%SVM
% y_pred_svm=svmModel.predictFcn(xtest);
% fig=figure;
% 
% Ac_pred_svm=num2cell(y_pred_svm);
% Ac_pred_svm(y_pred_svm==0) = {'Idle'};
% Ac_pred_svm(y_pred_svm==2) = {'Walk'};
% Ac_pred_svm(y_pred_svm==-1) = {'Down'};
% Ac_pred_svm(y_pred_svm==3) = {'Run'};
% Ac_pred_svm(y_pred_svm==1) = {'Up'};
% 
% cm3 = confusionchart(Ac_test,Ac_pred_svm,'RowSummary','row-normalized','ColumnSummary','column-normalized');
% cm3.Title='Human Activity Detection using SVM';
% %cm.ClassLabels=["Walk","Run","Idle","Up","Down"]
% fig_Position = fig.Position;
% fig_Position(3) = fig_Position(3)*1.5;
% fig.Position = fig_Position;
% cm3.Normalization = 'row-normalized'; 
% sortClasses(cm3,'descending-diagonal')
% cm3.Normalization = 'absolute';
% [accuracy3,precision3,recall3,f1_score3]=acc_metric(ytest,y_pred_svm)


%% BIAS VARIANCE TRADEOFF
% as we can see, we get maximum accuracy for k=1.
%1-NN model means our model is really close to your training data and therefore the bias is low. 
% In contrast to this the variance in our model is high, because our model is extremely sensitive and wiggly.
%%
%Plot 
%plot((1:size(dataTrain,1)),accuracy);
% mdl = fitcknn(X,Y);
% knnK = 30; %num of nearest neighbors using in KNN classifier
% mdl.NumNeighbors = knnK;%specify num of nearest neighbors
mdl.NumNeighbors=10; %(Let)

%%
%Step 4: load recorded data and uniformly resample it
testWalk = timehandeling('climb3.mat');
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


%%
%Step 5: Activity Detection
i = 1;
lastFrame = find(t>(t(end)-windowLength-0.005), 1);
% Set default starting activity to idling
lastDetectedActivity = 0;

frameIndex = [];
result = [];
score = [];

% Parse through the data in 5 second windows and detect activity for each 5
% second window
while (i < lastFrame)
    startIndex = i;
    frameIndex(end+1,:) = startIndex;
    t0 = t(startIndex);
    nextFrameIndex = find(t > t0 + detectionInterval);
    nextFrameIndex = nextFrameIndex(1) - 1;
    stopIndex = find(t > t0 + windowLength);
    stopIndex = stopIndex(1) - 1;
    currentFeature = extractFeatures(a(startIndex:stopIndex, :, :),...
                     t(startIndex:stopIndex), uniformSampleRate);
    currentFeature = (currentFeature - dMin) ./ range;
    [tempResult,tempScore] = predict(mdl, currentFeature);
    % Scores reported by KNN classifier is ranging from 0 to 1. Higher score
    % means greater confidence of detection.
    if max(tempScore) < 0.90 || tempResult ~= lastDetectedActivity 
        % Set result to transition
        result(end+1, :) = -10; 
    else
        result(end+1, :) = tempResult;
    end
    lastDetectedActivity = tempResult;
    score(end+1, :) = tempScore;
    i = nextFrameIndex + 1;
end


%%
%Step 6: Generate a plot of raw data and the results
figure;
plot(t,a);
% Raw acceleration data is bounded by +-20, leaving space in bottom of the 
% graph for activity detection markers.
ylim([-30 20]);
hold all;


resWalk =(result == 2);
resRun  =(result == 3);
resIdle =(result == 0);
resDown =(result ==-1);
resUp   =(result == 1);
resUnknown =(result == -10);

sum_w=sum(resWalk);
sum_r=sum(resRun);
sum_i=sum(resIdle);
sum_d=sum(resDown);
sum_up=sum(resUp);
sum_unknown=sum(resUnknown);

res_val = [sum_w sum_r sum_i sum_d sum_up sum_unknown];
activity =["Walking","Running","Idle","Climbing_down","Climbing_Up","Transition"]; 
[m,n]=max(res_val);
fprintf("***********You are %s***************",activity(n));

hold all;

% Plot activity detection markers below the raw acceleration data

% Plot activity detection markers below the raw acceleration data

% hWalk = plot(t(frameIndex(resWalk))+windowLength, 0*result(resWalk)-25, 'kx');
% hRun  = plot(t(frameIndex(resRun))+windowLength, 0*result(resRun)-25, 'r*');
% hIdle = plot(t(frameIndex(resIdle))+windowLength, 0*result(resIdle)-25, 'bo');
% hDown = plot(t(frameIndex(resDown))+windowLength, 0*result(resDown)-25, 'cv');
% hUp   = plot(t(frameIndex(resUp))+windowLength, 0*result(resUp)-25, 'm^');
% hTransition = plot(t(frameIndex(resUnknown))+windowLength, 0*result(resUnknown)-25, 'k.');
% 
% % Increase y-axis limit to include the detected marker
% ylim([-30 20]);
% 
title('',activity(n));
% % Add legend to the graph
% legend([hWalk,hRun,hIdle,hDown, hUp, hTransition],'Walk','Run','Idle','Walking Downstairs','Walking Upstairs','Transition');

