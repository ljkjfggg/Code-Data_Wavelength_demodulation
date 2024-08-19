%% Used to obtain all possible combinations of series grating array
% Applicable when the range of each grating is the same
function [D]=sgwc(A) %Series grating wavelength combination
[m,n]=size(A);% m is the number of values for each grating, n is the number of gratings
for i=2:n
    j=i+2;
    A(:,i)=A(:,i).*j;
end
Q=reshape(A,[],1);% Reshape matrix A into a single column matrix
B=nchoosek(Q,n);% Get all possible combinations without considering the order of gratings in the array
[m2,~]=size(B);% Number of combinations

% Obtain all possible combinations arranged in the order of gratings
% Obtained by filtering matrix B
C=cell(1,m+1);% Create an empty cell array to store combinations
C(1,1)=mat2cell(B,[m2],[n]);% The first element of the cell array is all possible unordered combinations of gratings
% Obtain combinations arranged in the order of gratings using the find function
for k=1:n % Start from the first grating and go until the last one
    for i= 1:m % All possible values of the k-th grating
      j=i+1; % Store the filtering result in other positions of the cell array, keeping C{1,1}
      C{1,j}=C{1,1}(find(C{1,1}(:,k)==A(i,k)),:); % Search through the loop
      % When the range of values for each grating is the same (i.e., each row of A is the same), only k=1 is executed, others are ineffective
      % Therefore, matrix A (excluding the first column) needs to be multiplied by a number to make each column different
    end
    % A series of operations to continuously reduce C{1,1}
    D=C;D{1,1}=[];D=reshape(D,[],1);D=cell2mat(D);C{1,1}=D;
end
D=sortrows(D,1);% D contains all possible combinations of the n gratings in the range of values in series
for i=2:n
    j=i+2;
    D(:,i)=D(:,i)./j;
end
