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
    {'name': 'Carne', 'ingredients': ['Carne moída', 'Cebola', 'Azeitona'], 'price': 32.0},
    {'name': 'Frango', 'ingredients': ['Frango desfiado', 'Catupiry', 'Milho'], 'price': 27.0},
    {'name': 'Queijo', 'ingredients': ['Queijo Mussarela', 'Queijo Provolone', 'Orégano'], 'price': 24.5},
    {'name': 'Palmito', 'ingredients': ['Palmito', 'Tomate', 'Queijo Prato'], 'price': 36.0},
    {'name': 'Pizza', 'ingredients': ['Presunto', 'Queijo Mussarela', 'Tomate'], 'price': 29.5},
    {'name': 'Calabresa', 'ingredients': ['Calabresa', 'Cebola', 'Queijo Mussarela'], 'price': 31.5},
    {'name': 'Bacalhau', 'ingredients': ['Bacalhau', 'Batata', 'Pimentão'], 'price': 48.0},
    {'name': 'Brócolis', 'ingredients': ['Brócolis', 'Alho', 'Ricota'], 'price': 26.0},
    {'name': 'Carne Seca', 'ingredients': ['Carne seca', 'Abóbora', 'Cebola roxa'], 'price': 37.0},
    {'name': 'Catupiry', 'ingredients': ['Catupiry', 'Tomate seco', 'Azeitona preta'], 'price': 33.5},
    {'name': 'Milho', 'ingredients': ['Milho', 'Queijo Coalho', 'Creme de leite'], 'price': 22.0},
    {'name': 'Banana', 'ingredients': ['Banana', 'Canela', 'Açúcar'], 'price': 19.0},
    {'name': 'Chocolate', 'ingredients': ['Chocolate', 'Granulado', 'Leite condensado'], 'price': 30.0},
    {'name': 'Romeu e Julieta', 'ingredients': ['Goiabada', 'Queijo Branco', 'Açúcar'], 'price': 25.0},
    {'name': 'Camarão', 'ingredients': ['Camarão', 'Catupiry', 'Alho Poró'], 'price': 50.0},
]

# Create Pastel/Ingrediente nodes plus the FEITO_DE relationships.
graph.query(
    '''
    UNWIND $recipes AS pastel
    CREATE (p:Pastel {name: pastel.name, price: pastel.price})
    WITH p, pastel.ingredients AS ingredients
    UNWIND ingredients AS ingredient
    MERGE (i:Ingrediente {name: ingredient})
    CREATE (p)-[:FEITO_DE]->(i)
    ''',
    params={'recipes': pastel_recipes},
)

# Display the created names and their ingredients.
result = graph.ro_query(
    '''
    MATCH (p:Pastel)-[:FEITO_DE]->(i:Ingrediente)
    RETURN p.name AS name, p.price AS price, collect(i.name) AS ingredients
    ORDER BY name
    '''
).result_set
for name, price, ingredients in result:
    ingredient_list = ', '.join(ingredients)
    print(f'Pastel name: {name} | Ingredients: {ingredient_list} | Price: R$ {price:.2f}')

# Print a graph summary.
summary = graph.ro_query(
    '''
    MATCH (p:Pastel)-[r:FEITO_DE]->(i:Ingrediente)
    RETURN count(DISTINCT p) AS pasteis, count(DISTINCT i) AS ingredientes, count(r) AS relacoes
    '''
).result_set[0]
print(
    'Graph summary -> '
    f"Pastels: {summary[0]}, Unique ingredients: {summary[1]}, FEITO_DE relationships: {summary[2]}"
)
