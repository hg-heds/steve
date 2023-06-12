clear all; % Limpar as variáveis
close all; % Fechar todas as imagens
clc; % Limpar a tela

xmin = -100;
xmax = 100;

tamPOP = 50;
numVAR = 1;
numGER = 100;
POP = xmin + rand(tamPOP,numVAR) * (xmax - xmin);
FX = POP .^ 3;
for g = 2:numGER
    POPnovo = cruzamento(POP,xmin,xmax);
    FXnovo = POPnovo .^ 3;
    POP = [POP; POPnovo];
    FX = [FX; FXnovo];
    clf; hold on;
    plot(POP,FX,'bo');
    plot(POPnovo,FXnovo,'rx');
    drawnow;

    [~, ind] = sort(FX);
    ind = ind(1:tamPOP);
    POP = POP(ind);
    FX = FX(ind);
end
min(FX)








