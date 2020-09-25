I participated in IEEE PES DATATHON 2020 few days ago. This was my first kaggle competition.The task was-

Use machine learning to create a model that predicts Global Horizontal Solar Irradiance (GHI) from a set of features(Temparature,pressure,wind speed,wind direction,precipitation,relative_humidity)
Here,\

*global_horizontal_irradiance -> Global Horizontal Solar Irradiance (W/m2)\
*precipitation -> Precipitation (mm)\
*atmospheric_pressure -> Atmospheric Pressure (mm Hg)\
*relative_humidity -> Relative Humidity (%)\
*air_temperature -> Air Temperature (degree celsius)\
*wind_direction -> Wind Direction (degree north)\
*wind_speed -> Wind Speed (m/s)\

I tried various method- randomforest,neural network,gradient boost etc. Among them,i tried to stick with Neural network because of its large optimization,regularization scope.
Morever,I wanted to learn neural network architecture,optimization in details(as a first timer). So that ,i choose neural net over random forest.

Basically, I trained 3 different NN Model with different optimization, Then i stacked those model. And i use weighted avg on those result.\
I was 3rd in public leaderboard,4th in private leaderboard.

Any suggestion on improving accuracy is greatly appreciated.My mail-
fardinpranto005@gmail.com

competition details-https://www.kaggle.com/c/ieee-pes-bdc-datathon-year-2020/overview
