%% Used to obtain reflectance spectrum data based on wavelength
function [AS,Ga]=AS_calculate(x,LS,lambda,Ra,delta)
[mm,~]=size(lambda);
xx=x;% wavelength matrix
Fa=LS;
[ynum,~]=size(xx);
Sa=zeros(mm,ynum);
Ga=zeros(mm,ynum);
% Independent reflectance spectra of each FBG
for k=1:ynum
      Sa(:,k)=Ra(k).*exp((-4*log(2).*((lambda-xx(k,1))./delta(k)).^2));
end
[Sra,~]=size(Sa);
% The signal of the ith grating reflecting light reaching the input end of the sensor array.
RSa(:,1)=ones(Sra,1);
for l=2:ynum
    RSa(:,l)=RSa(:,l-1).*(1-Ra(l-1).*Sa(:,l-1));
end
% Obtaining the signals reflected back to the input end from each FBG
for nn = 1:ynum
    Ga(:,nn)=Fa.*Sa(:,nn).*(RSa(:,nn).^2);
end
AS=sum(Ga,2); % Simulate the reflection spectrum