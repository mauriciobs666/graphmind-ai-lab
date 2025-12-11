# graphmind-ai-lab
Experiments with graphs and LLMs

Falkor DB

docker run -p 6379:6379 -p 3000:3000 -it --rm falkordb/falkordb:edge
#   -p 6379:6379 expõe a porta do Redis/FalkorDB para a máquina host
#   -p 3000:3000 expõe o painel web do FalkorDB
#   -it mantém o container interativo com pseudo-TTY
#   --rm remove o container ao final da execução
#   falkordb/falkordb:edge é a imagem utilizada

