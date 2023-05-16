clear all; % Limpa todas as variáveis
close all; % Fecha todas figuras
clc; % Limpa a tela

xmin = -5.12; % Específico para a função rastrigin
xmax = 5.12;


%%%%%%%% tamPOP * numGER <= 10000
tamPOP = 20;
numGER = 1000 / tamPOP;
%%%%%%%%

numVAR = 2; % Valor fixo

POP = xmin + rand(tamPOP,numVAR) .* (xmax - xmin);
FX = calculaFX(POP);

for g = 2:numGER
    for i = 1:tamPOP
        j = randperm(numVAR,1);
        C = 0.5 + rand;
        r = randperm(tamPOP,3);
        Pnovo = POP(r(1),:) + C * (POP(r(3),:) - POP(r(2),:));
        
        for d = 1:numVAR
            if ((rand <= 0.5) && (d ~= j))
                Pnovo(d) = POP(i,d);
            end
        end
        
        FXnovo = calculaFX(Pnovo);
        
        if (FXnovo <= FX(i))
            POP(i,:) = Pnovo;
            FX(i) = FXnovo;
        end
    end
    
    plot(POP(:,1),POP(:,2),'ro');
    axis([xmin xmax xmin xmax]);
    xlabel(num2str(g));
    grid on;
    pause(0.1);
end

min(FX)