function [J grad]=CostFunction(X,Y,theta)
m=length(X);
c1=sigmoid(X*theta);
c1=c1';

d1=1/(1-exp(-X*theta));
J=-Y.*log(c1)-(1-Y).*log(1-c1);
J=sum(J)/(m);
grad = zeros(size(theta));
for i=1:length(grad)
derie=(c1-Y).*X(:,i);
grad(i)=sum(derie)/m;
endfor;
%fprintf("Sum: \n");
%disp((sum((c1-Y).*X(:,1))/m));
%fprintf("\nGradient\n");
%disp(size(grad));
%fprintf("\n");
end;
