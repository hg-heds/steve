function [POP, FX] = selecao(POP,FX,tamPOP)
    tipo = 3;

    numPOP = size(FX,1);
    
    switch tipo
        case 1 % ELITISMO
            [~, ind] = sort(FX);
            ind = ind(1:tamPOP);
        case 2 % TORNEIO
            for i = 1:tamPOP
                r = randperm(numPOP,2);
                if (FX(r(1)) <= FX(r(2)))
                    ind(i) = r(1);
                else
                    ind(i) = r(2);
                end
            end
        case 3 % ROLETA
            F = 1.1 * max(FX) - FX;
            F = F / sum(F);

            for i = 1:tamPOP
                soma = 0;
                cont = 0;
                r = rand;
                while (soma < r)
                    cont = cont + 1;
                    soma = soma + F(cont);
                end
                ind(i) = cont;
            end
    end
    POP = POP(ind,:);
    FX = FX(ind);
end


