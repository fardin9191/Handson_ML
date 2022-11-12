function data=timehandeling(filename)
format longg
test=load(filename)
datetime=test.Acceleration.Timestamp;
t=datenum(datetime);
t1=0;
for i=1:length(t)
    t(i)=t1;
    t1=t1+0.02;
end
% time=convertTo(datetime,'excel');
% integ=floor(time);
% t=time-integ;
a=test.Acceleration{:,1:3};
data = [a t];
