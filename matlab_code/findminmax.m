function [newmin,newmax] = findminmax(val,min_,max_)

if val >  max_
    newmax =val;
else
    newmax = max_;
end

if val <  min_
    newmin = val;
else
    newmin = min_;
   
end
