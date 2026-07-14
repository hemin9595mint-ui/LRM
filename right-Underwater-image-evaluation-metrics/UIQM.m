function uiqm = UIQM(image, c1, c2, c3)


if nargin < 2
    c1 = 0.0282;
end

if nargin < 3
    c2 = 0.2953;
end

if nargin < 4
    c3 = 3.5753;
end


%% UICM

[~,~,~,~,uicm] = UICM(image);



%% UISM

uism = UISM(image);



%% UIConM

uiconm = UIConM(image);



%% UIQM

uiqm = c1*uicm + c2*uism + c3*uiconm;


end