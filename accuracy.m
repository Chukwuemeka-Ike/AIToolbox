function [acc] = accuracy(y, yPred)
    sumAcc = 0;
    if (length(y) == length(yPred))
        for i = 1:length(y)
            if (y(i) == yPred(i))
                sumAcc = sumAcc+1;
            end
        end
    end
    acc = (sumAcc/length(y)) * 100;
end