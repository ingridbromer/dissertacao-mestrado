Este repositório é composto por três arquivos principais:

1) pre-processamento.py
Responsável pela etapa inicial de preparação dos dados. As imagens originais da base CRIC são recortadas em patches de 90×90 pixels e organizadas em subpastas de acordo com o tipo de lesão. A definição das classes é baseada no arquivo classifications.json.
Tanto as imagens quanto o arquivo JSON estão disponíveis em: https://cricdatabase.com.br/

2) balanceamento.py
Realiza o aumento de dados (data augmentation) de forma direcionada, com o objetivo de balancear o conjunto. O processo pode ser configurado para cenários de 2, 3 ou 6 classes, conforme a estratégia de classificação adotada.

3) treinamento.py
Executa o treinamento de diferentes arquiteturas de redes neurais. Como saída, são gerados:
  Arquivos de modelo treinado (.pth)
  Métricas de desempenho
  Curvas ROC e Precision-Recall (PR)
  Matrizes de confusão
