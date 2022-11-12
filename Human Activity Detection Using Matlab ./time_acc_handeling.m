test=load("testdata.mat");
datetime=test.Acceleration.Timestamp;
time=convertTo(datetime,'excel');
integ=floor(time);
t=time-integ;
a=test.Acceleration{:,1:3}
