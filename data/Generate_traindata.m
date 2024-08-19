load LS % Load the spectral shape data of the light source
s1=1551:0.001:1552;s1=s1';% Define the wavelength range of FBG1
s2=1551:0.001:1552;s2=s2';% Define the wavelength range of FBG2
% Obtain all possible wavelength combinations of the two FBGs, B
A=[s1 s2];B=sgwc(A);[a,b]=size(B);
% Randomly sample a certain number of combinations from all possible combinations as needed for model training
x=randi(1000000,1000000,1);
D=B(x,:); % % Wavelength combination matrix
lambda=(1550.5:0.001:1552.5)';% Spectrum wavelength range
num=size(D,1);

for i=1:num
    % Reflectivity range of each FBG
    % Please input the corresponding values based on the test results
    x1=randi([100,160],1,1);
    x2=randi([220,310],1,1);
    R=[x1/10000  x2/10000];
    % FWHM range of each FBG
    % Please input the corresponding values based on the test results
    d1=randi([1400,1550],1,1);
    d2=randi([1400,1550],1,1);
    delta=[d1/10000 d2/10000];
    % % Obtain spectra, reflectivity, and FHWM data
    xa=D(i,:)';
    [AS,~]=AS_calculate(xa,LS,lambda,R,delta);
    AS_total(:,i)=AS; % Spectra
    R_total(i,:)=R; % Reflectivity
    delta_total(i,:)=delta; % FWHM
end
AS_total=AS_total';
Xtrain=AS_total;% Training input
Ytrain=[D-1550.5 R_total delta_total];% Training output, subtracting the lower limit of the spectrum wavelength range to prevent large differences in wavelength data compared to reflectance and full width at half maximum data.
save('Xtrain.mat','Xtrain')
save('Ytrain.mat','Ytrain')