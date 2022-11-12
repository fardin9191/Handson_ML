function featureTraining = extractTrainingFeature(rawData,...
                                                  windowLength,...
                                                  uniformSampleRate)

    format longg;
    %rawData = load(fileName); % load recorded data
    
    %resampling
    tStart = rawData(1,4);
    tStop = rawData(end,4);
    newTime = tStart:1/uniformSampleRate:tStop;
    a_data = rawData(:,1:3);
    t_data = rawData(:,4);
    newData = interp1(t_data, a_data, newTime);

    %find out the starting index of last possilbe window in training data
    frameIndex = find(newTime > (newTime(end) - windowLength...
                      - 2 / uniformSampleRate));
    lastFrame = frameIndex(1);
    
    featureTraining = [];
    %using a sliding window to extract features of recorded data
    for i = 1:lastFrame
        startIndex = i; % window start index
        t0 = newTime(startIndex);
        stopIndex = find(newTime > t0 + windowLength);
        stopIndex = stopIndex(1) - 1; % window stop index
        featureTraining(end+1,:) = extractFeatures(...
                                   newData(startIndex:stopIndex,:,:),...
                                   newTime(startIndex:stopIndex),...
                                   uniformSampleRate);
    end

end