function rotatemat = rri_bootprocrust(origlv,bootlv)
% Purpose: Computes the procrustean transform between original and bootstrap LVs
% Inputs:
%   origlv: original LVs
%   bootlv: bootstrap LVs
% Output:
%   rotatemat: procrustean transform matrix
%
% v1.0 Nov 2009 Jonas Richiardi
% - initial release
% v1.0.1 Nov 2012 JR
% - fixed doc

%% define coordinate space between original and bootstrap LVs
temp = origlv'*bootlv;

%% orthogonalze space
[V, ~, U] = svd(temp);

%% determine procrustean transform
rotatemat = U*V';
end