function FX = calculaFX(POP)
    tamPOP = size(POP,1);
    
    FX = zeros(tamPOP,1);
    
    for i = 1:tamPOP
        FX(i,:) = rastrigin(POP(i,:));
    end
end