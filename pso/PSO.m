clear all; % Limpar as variáveis
close all; % Fechar todas as imagens
clc; % Limpar a tela

xmin = -5.12;
xmax = 5.12;

tamPOP = 50;
numVAR = 2;
numGER = 100;
Vmax = 0.05 * (xmax - xmin);

POP = xmin + rand(tamPOP,numVAR) * (xmax - xmin);
FX = calculaFX(POP);
V = 0.1 * (rand(tamPOP,numVAR) - 0.5) * (xmax - xmin);

alpha = [1 0.3 0.2]; % atração | pbest | gbest

POPpbest = POP;
FXpbest = FX;

[FXgbest, ind] = min(FX);
POPgbest = POP(ind,:);

for g = 2:numGER
    POP = POP + V;
    POP = min(POP,xmax);
    POP = max(POP,xmin);
    FX = calculaFX(POP);

    for i = 1:tamPOP
        if (FX(i) <= FXpbest(i))
            FXpbest(i) = FX(i);
            POPpbest(i,:) = POP(i,:);
            if (FXpbest(i) <= FXgbest)
                FXgbest = FXpbest(i);
                POPgbest = POPpbest(i,:);
            end
        end
    end

    V = alpha(1) * V + ...
        alpha(2) * (POPpbest - POP) + ...
        alpha(3) * (repmat(POPgbest,tamPOP,1) - POP);

    V = max(V, -Vmax);
    V = min(V, Vmax);
    
    clf; hold on;
    plot(POP(:,1),POP(:,2),'bo','LineWidth',6);
    plot(POPgbest(:,1),POPgbest(:,2),'rx','LineWidth',6);
    axis([xmin xmax xmin xmax]);
    grid on;
    xlabel([num2str(g) " - " num2str(FXgbest)])
    pause(0.1);
    drawnow;
end
FXgbest