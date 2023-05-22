clear all; % Limpa todas as variáveis
close all; % Fecha todas figuras
clc; % Limpa a tela

xmin = -5.12; % Específico para a função rastrigin
xmax = 5.12;

resultados = zeros(1,10);


%%%%%%%% tamPOP * numGER <= 10000
tamPOP = 140;
numGER = fix(10000 / tamPOP)+1;
%%%%%%%%

numVAR = 2;

for i = 1:10
    POP = xmin + rand(tamPOP,numVAR) .* (xmax - xmin);
    POPini = POP;
    FX = calculaFX(POP);

    for g = 2:numGER
        POPnovo = cruzamento(POP,xmin,xmax);
        POPnovo = mutacao(POPnovo,xmin,xmax);
        FXnovo = calculaFX(POPnovo);

        POP = [POP; POPnovo];
        FX = [FX; FXnovo];

        [POP, FX] = selecao(POP,FX,tamPOP);
        
        if (numVAR == 2)
            plot(POP(:,1),POP(:,2),'ro');
            axis([xmin xmax xmin xmax]);
            xlabel(num2str(g));
            grid on;
            drawnow; 
        else
            parallelcoords(POP);
            axis([1 numVAR xmin xmax]);
            xlabel(num2str(g));
            grid on;
            drawnow;
        end

    end
    
        
    resultados(1,i) = min(FX);
end

resultados
datastats(resultados')