from falkordb import FalkorDB
from config import Config

# Connect to FalkorDB using Streamlit secrets.
credentials = Config.get_falkordb_credentials()
db = FalkorDB(
    host=credentials["host"],
    port=credentials["port"],
    username=credentials["username"],
    password=credentials["password"]
)

graph = db.select_graph(Config.get_falkordb_graph())

# Remove any existing data from the graph.
graph.query('MATCH (n) DETACH DELETE n')

# Menu flavors with their ingredients and prices.
pastel_recipes = [
    {'sabor': 'Carne', 'ingredientes': ['Carne moída', 'Cebola', 'Azeitona'], 'preco': 32.0},
    {'sabor': 'Frango', 'ingredientes': ['Frango desfiado', 'Catupiry', 'Milho'], 'preco': 27.0},
    {'sabor': 'Queijo', 'ingredientes': ['Queijo Mussarela', 'Queijo Provolone', 'Orégano'], 'preco': 24.5},
    {'sabor': 'Palmito', 'ingredientes': ['Palmito', 'Tomate', 'Queijo Prato'], 'preco': 36.0},
    {'sabor': 'Pizza', 'ingredientes': ['Presunto', 'Queijo Mussarela', 'Tomate'], 'preco': 29.5},
    {'sabor': 'Calabresa', 'ingredientes': ['Calabresa', 'Cebola', 'Queijo Mussarela'], 'preco': 31.5},
    {'sabor': 'Bacalhau', 'ingredientes': ['Bacalhau', 'Batata', 'Pimentão'], 'preco': 48.0},
    {'sabor': 'Brócolis', 'ingredientes': ['Brócolis', 'Alho', 'Ricota'], 'preco': 26.0},
    {'sabor': 'Carne Seca', 'ingredientes': ['Carne seca', 'Abóbora', 'Cebola roxa'], 'preco': 37.0},
    {'sabor': 'Catupiry', 'ingredientes': ['Catupiry', 'Tomate seco', 'Azeitona preta'], 'preco': 33.5},
    {'sabor': 'Milho', 'ingredientes': ['Milho', 'Queijo Coalho', 'Creme de leite'], 'preco': 22.0},
    {'sabor': 'Banana', 'ingredientes': ['Banana', 'Canela', 'Açúcar'], 'preco': 19.0},
    {'sabor': 'Chocolate', 'ingredientes': ['Chocolate', 'Granulado', 'Leite condensado'], 'preco': 30.0},
    {'sabor': 'Romeu e Julieta', 'ingredientes': ['Goiabada', 'Queijo Branco', 'Açúcar'], 'preco': 25.0},
    {'sabor': 'Camarão', 'ingredientes': ['Camarão', 'Catupiry', 'Alho Poró'], 'preco': 50.0},
]

# Create Pastel/Ingrediente nodes plus the FEITO_DE relationships.
graph.query(
    '''
    UNWIND $recipes AS pastel
    CREATE (p:Pastel {flavor: pastel.sabor, price: pastel.preco})
    WITH p, pastel.ingredientes AS ingredientes
    UNWIND ingredientes AS ingrediente
    MERGE (i:Ingrediente {name: ingrediente})
    CREATE (p)-[:FEITO_DE]->(i)
    ''',
    params={'recipes': pastel_recipes},
)

# Display the created flavors and their ingredients.
result = graph.ro_query(
    '''
    MATCH (p:Pastel)-[:FEITO_DE]->(i:Ingrediente)
    RETURN p.flavor AS flavor, p.price AS price, collect(i.name) AS ingredients
    ORDER BY flavor
    '''
).result_set
for flavor, price, ingredients in result:
    ingredient_list = ', '.join(ingredients)
    print(f'Pastel flavor: {flavor} | Ingredients: {ingredient_list} | Price: R$ {price:.2f}')

# Print a graph summary.
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
