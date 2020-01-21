function number = clamp(a,b,c)
%CLAMP
%   Makes sure that the result is b if a<=b<=c or a if b<a or c if b>c
if b<a
    number = a;   
elseif b>c
    number = c;
else
    number = b;
end

end

