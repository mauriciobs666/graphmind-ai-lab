
from falkordb import FalkorDB

# Conecta-se ao FalkorDB
db = FalkorDB(host='localhost', port=6379)

# Seleciona o grafo desejado
graph = db.select_graph('kg_pastel')

# Limpa qualquer dado pré-existente no grafo
graph.query('MATCH (n) DETACH DELETE n')

# Lista de sabores e seus ingredientes
pastel_recipes = [
    {'sabor': 'Carne', 'ingredientes': ['Carne moída', 'Cebola', 'Azeitona']},
    {'sabor': 'Frango', 'ingredientes': ['Frango desfiado', 'Catupiry', 'Milho']},
    {'sabor': 'Queijo', 'ingredientes': ['Mussarela', 'Provolone', 'Orégano']},
    {'sabor': 'Palmito', 'ingredientes': ['Palmito', 'Tomate', 'Queijo prato']},
    {'sabor': 'Pizza', 'ingredientes': ['Presunto', 'Mussarela', 'Tomate']},
    {'sabor': 'Calabresa', 'ingredientes': ['Calabresa', 'Cebola', 'Queijo']},
    {'sabor': 'Bacalhau', 'ingredientes': ['Bacalhau', 'Batata', 'Pimentão']},
    {'sabor': 'Brócolis', 'ingredientes': ['Brócolis', 'Alho', 'Ricota']},
    {'sabor': 'Carne Seca', 'ingredientes': ['Carne seca', 'Abóbora', 'Cebola roxa']},
    {'sabor': 'Catupiry', 'ingredientes': ['Catupiry', 'Tomate seco', 'Azeitona preta']},
    {'sabor': 'Milho', 'ingredientes': ['Milho', 'Queijo coalho', 'Creme de leite']},
    {'sabor': 'Banana', 'ingredientes': ['Banana', 'Canela', 'Açúcar']},
    {'sabor': 'Chocolate', 'ingredientes': ['Chocolate', 'Granulado', 'Leite condensado']},
    {'sabor': 'Romeu e Julieta', 'ingredientes': ['Goiabada', 'Queijo minas', 'Açúcar de confeiteiro']},
    {'sabor': 'Camarão', 'ingredientes': ['Camarão', 'Catupiry', 'Alho-poró']},
]

# Cria os nós Pastel, os nós Ingrediente e relacionamentos FEITO_DE
graph.query(
    '''
    UNWIND $recipes AS pastel
    CREATE (p:Pastel {sabor: pastel.sabor})
    WITH p, pastel.ingredientes AS ingredientes
    UNWIND ingredientes AS ingrediente
    MERGE (i:Ingrediente {nome: ingrediente})
    CREATE (p)-[:FEITO_DE]->(i)
    ''',
    params={'recipes': pastel_recipes},
)

# Exibe os sabores e ingredientes associados
result = graph.ro_query(
    '''
    MATCH (p:Pastel)-[:FEITO_DE]->(i:Ingrediente)
    RETURN p.sabor AS sabor, collect(i.nome) AS ingredientes
    ORDER BY sabor
    '''
).result_set
for sabor, ingredientes in result:
    ingredientes_list = ', '.join(ingredientes)
    print(f'Pastel sabor: {sabor} | Ingredientes: {ingredientes_list}')

# Resumo do estado atual do grafo
summary = graph.ro_query(
    '''
    MATCH (p:Pastel)-[r:FEITO_DE]->(i:Ingrediente)
    RETURN count(DISTINCT p) AS pasteis, count(DISTINCT i) AS ingredientes, count(r) AS relacoes
    '''
).result_set[0]
print(
    'Resumo do grafo -> '
    f"Pastéis: {summary[0]}, Ingredientes únicos: {summary[1]}, Relações FEITO_DE: {summary[2]}"
)
