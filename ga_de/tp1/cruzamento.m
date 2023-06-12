function POPnovo = cruzamento(POP,xmin,xmax)
    
    [tamPOP, numVAR] = size(POP);
    POPnovo = zeros(tamPOP,numVAR);
    
    for i = 1:tamPOP
        r = randperm(tamPOP,2);
        POPnovo(i,:) = POP(r(1),:) + (2 * rand(1,numVAR) - 0.5) .* (POP(r(2),:) - POP(r(1),:));
    end
    
    POPnovo = max(POPnovo,xmin);
    POPnovo = min(POPnovo,xmax);
end