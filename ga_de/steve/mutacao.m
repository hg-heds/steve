function POPnovo = mutacao(POP,xmin,xmax)
    [tamPOP, numVAR] = size(POP);
    
    POPnovo = zeros(tamPOP,numVAR);

    for i = 1:tamPOP
        tipo = randperm(3,1);
        switch tipo
            case 1
                POPnovo(i,:) = POP(i,:) + (rand - 0.5) * (xmax - xmin);
            case 2
                C = 10;
                POPnovo(i,:) = POP(i,:) + (rand - 0.5) * C;
            case 3
                POPnovo(i,:) = xmin + rand(1,numVAR) .* (xmax - xmin);
        end
    end

    POPnovo = max(POPnovo,xmin);
    POPnovo = min(POPnovo,xmax);
end