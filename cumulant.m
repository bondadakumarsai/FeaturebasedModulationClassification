function out = cumulant(X1)
% MAPFEATURE Feature mapping function to polynomial features
%
%   MAPFEATURE(X1, X2) maps the two input features
%   to quadratic features used in the regularization exercise.
%
%   Returns a new feature array with more features, comprising of 
%   X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..
%
%   Inputs X1, X2 must be the same size
%

len = size(X1,1);
cum = [];
for i = 1:len
    X = X1(i,:);
    
    m20 = sum(X.^2)/length(X);
    c20 = m20;
    m21 = sum((abs(X)).^2)/length(X);
    c21 = m21;

    m40 = sum(X.^4)/length(X);
    c40 = m40 - 3 * m20^2;

    m41 = sum((X.^2).*(abs(X).^2))/length(X);
    c41 = m41 - 3 * m21 * m20;

    m42 = sum((abs(X).^4))/length(X);
    c42 = m42 - (abs(m20))^2 - 2 * (m21^2);

    m63 = sum((abs(X).^6))/length(X);
    c63 = m63 - 9 * (c42 * c21) - 6 *(c21^3);

    m60 = sum((X).^6)/length(X);
    c60 = m60 - 15 * (m20 * m40) + 3 * (m20).^3;

    m61 = sum((X.^4).*(abs(X).^2))/length(X);  
    c61 = m61 - 5 * (m21 * m40) - 10 *(m20 * m41) + 30 * ((m20^2)*m21);

    m62 = sum((X.^2).*(abs(X).^4))/length(X);  
    m22 = sum((conj(X)).^2)/length(X);
    c62 = m62 - 6 * (m20 * m42) - 8 *(m21 * m41) - (m22 * m40) + 6 * ((m20^2)*m22) + 24 * ((m21^2)*m20);

    ratio = abs(c63)^2/abs(c42)^3;
   
    cum(i,:) = [ratio abs(c60) abs(c61) abs(c62) abs(c63) abs(m60) abs(m61) abs(m62) abs(m63)];

end

out = cum;
end