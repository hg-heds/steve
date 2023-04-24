clear all; % Limpar as variáveis
close all; % Fechar todas as imagens
clc; % Limpar a tela

xmin = -5.12;
xmax = 5.12;

tamPOP = 20;
numVAR = 2;
numGER = 100;
POP = xmin + rand(tamPOP,numVAR) * (xmax - xmin);
FX = calculaFX(POP);
for g = 2:numGER
    POPnovo = cruzamento(POP,xmin,xmax);
%     POPnovo = mutacao(POPnovo,xmin,xmax);
    FXnovo = calculaFX(POPnovo);
    POP = [POP; POPnovo];
    FX = [FX; FXnovo];
    clf; hold on;
    plot(POP(:,1),POP(:,2),'bo','LineWidth',6);
    axis([xmin xmax xmin xmax]);
    grid on;
    pause(0.1);
    drawnow;

    [POP, FX] = selecao(POP,FX,tamPOP);
end
min(FX)