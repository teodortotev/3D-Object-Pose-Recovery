    function area = edge_f(v1, v2, p)
%EDGE_F
%   Implements the edge function (Pineda). 

area = (p(1) - v1(1))*(v2(2) - v1(2)) - (p(2) - v1(2))*(v2(1) - v1(1));

end

