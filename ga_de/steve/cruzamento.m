function POPnovo = cruzamento(POP,xmin,xmax)
    tipo = 3;

    [tamPOP, numVAR] = size(POP);
    
    POPnovo = zeros(tamPOP,numVAR);

    switch tipo
        case 1
            for i = 1:tamPOP
                r = randperm(tamPOP,1);
                POPnovo(i,:) = (POP(i,:) + POP(r,:)) / 2;
            end
        case 2
            for i = 1:tamPOP
                r = randperm(tamPOP,1);
                POPnovo(i,:) = POP(i,:) + rand(1,numVAR) .* (POP(r,:) - POP(i,:));
            end
        case 3
            for i = 1:tamPOP
                r = randperm(tamPOP,1);
                POPnovo(i,:) = POP(i,:) + (2 * rand(1,numVAR) - 0.5) .* (POP(r,:) - POP(i,:));
            end
    end
            
    POPnovo = max(POPnovo,xmin);
    POPnovo = min(POPnovo,xmax);
end