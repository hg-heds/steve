function POPnovo = mutacao(POPnovo,xmin,xmax)
    [tamPOP, numVAR] = size(POPnovo);
    
    for i = 1:tamPOP
        if (rand <= 0.2) % Probabilidade de mutação
            POPnovo(i,:) = POPnovo(i,:) + 0.5 * (1 * rand(1,numVAR) - 0.5) .* (xmax - xmin);
        end
    end
    POPnovo = max(POPnovo,xmin);
    POPnovo = min(POPnovo,xmax);    
end
