function [H] = NormLFPSF(H)
for k=1:size(H,3)
    for x1=1:size(H,1)
        for x2=1:size(H,2)
           H{x1,x2,k} = H{x1,x2,k}/sum(sum(H{x1,x2,k})); 
        end
    end
end

