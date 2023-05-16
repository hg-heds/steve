function [POP, FX] = selecao(POP,FX,tamPOP)
	tamELITE = tamPOP;
    
    [POP, FX] = elite(POP,FX,tamELITE);
    [POPtorneio, FXtorneio] = torneio(POP,FX,tamPOP-tamELITE);
    
    POP = [POP; POPtorneio];
    FX = [FX; FXtorneio];
end


function [POP, FX] = elite(POP,FX,tamPOP)

    [~, ind] = sort(FX);
    
    ind = ind(1:tamPOP); % Reduz o tamanho de ind para o valor desejado (tamPOP)
    
    POP = POP(ind,:);
    FX = FX(ind,:);
end

function [POP, FX] = torneio(POP,FX,tamPOP)
    ind = zeros(tamPOP,1);
    
    for i = 1:tamPOP
        r = randperm(size(POP,1),2);
        if (FX(r(1)) <= FX(r(2)))
            ind(i) = r(1);
        else
            ind(i) = r(2);
        end
    end

    POP = POP(ind,:);
    FX = FX(ind,:);    
end

function [POP, FX] = roleta(POP,FX,tamPOP)
    ind = zeros(tamPOP,1);
    
    FXnorm = 1 ./ FX; % Transforma os valores por se tratar de um problema de minimização
    FXnorm = FXnorm / sum(FXnorm); % Normaliza os valores para a roleta
    
    for i = 1:tamPOP
        r = rand;
        soma = 0;
        cont = 0;
        while (r >= soma)
            cont = cont + 1;
            soma = soma + FXnorm(cont);
        end
        ind(i) = cont;
    end

    POP = POP(ind,:);
    FX = FX(ind,:);
end