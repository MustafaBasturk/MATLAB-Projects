pretrainedNetwork = 'segnetVGG16CamVid.mat';  
data = load(pretrainedNetwork);
net = data.net;

% Load the video
videoFile = 'traffic_video_720_2.mp4';
videoReader = VideoReader(videoFile);

% Create a VideoWriter object to write the segmented video
outputVideoFile = fullfile('output_videos', 'output_segmented_video_2.mp4');
outputVideo = VideoWriter(outputVideoFile, 'MPEG-4');
outputVideo.FrameRate = videoReader.FrameRate;
open(outputVideo);

inputSize = [360, 480];
% Iterate over each frame in the video
while hasFrame(videoReader)
    % Read the current frame
    frame = readFrame(videoReader);
    frame = imresize(frame, inputSize);

    % Perform semantic segmentation on the frame
    segmentedFrame = semanticseg(frame, net);
    segmentedRGB = label2rgb(segmentedFrame);

    % Write the segmented frame to the output video
    writeVideo(outputVideo, segmentedRGB);
end

% Close the output video file
close(outputVideo);