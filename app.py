import dash
from dash import html, dcc, Input, Output, State
import pandas as pd
import joblib
from xgboost import XGBRanker

# Load the places dataset
places = pd.read_csv("training_dataset_v1.csv")
places['interest'] = places['interest'].str.replace(' ', '_')
places['district'] = places['district'].str.replace(' ', '_')
places['rating'] = places['rating'].fillna(places['rating'].median())
places['user_ratings_total'] = places['user_ratings_total'].fillna(0)
places['max_cost'] = places['max_cost'].fillna(500)

# Load model & encoders
model = XGBRanker()
model.load_model("date_recommender.xgb")
encoder = joblib.load("target_encoder.joblib")
scaler = joblib.load("feature_scaler.joblib")

districts = [  # From your list
    "Bang Bon", "Bang Kapi", "Bang Khae", "Bang Khen", "Bang Kho Laem", "Bang Khun Thian", "Bang Na", "Bang Phlat",
    "Bang Rak", "Bang Sue", "Bangkok Noi", "Bangkok Yai", "Bueng Kum", "Chatuchak", "Chom Thong", "Din Daeng",
    "Don Mueang", "Dusit", "Huai Khwang", "Khan Na Yao", "Khlong Sam Wa", "Khlong San", "Khlong Toei", "Lak Si",
    "Lat Krabang", "Lat Phrao", "Min Buri", "Nong Chok", "Nong Khaem", "Pathum Wan", "Phasi Charoen", "Phaya Thai",
    "Phra Khanong", "Phra Nakhon", "Pom Prap Sattru Phai", "Prawet", "Rat Burana", "Ratchathewi", "Sai Mai",
    "Samphanthawong", "Saphan Sung", "Sathon", "Suan Luang", "Taling Chan", "Thawi Watthana", "Thon Buri",
    "Thung Khru", "Wang Thonglang", "Watthana", "Yan Nawa"
]
interests = [
    "clubbing", "party", "cocktail_bar", "massage_spa", "yoga_studio", "live_music", "karaoke", "sky_bar",
    "fine_dining", "rooftop_bar", "speakeasy", "jazz_bar", "hookah_lounge", "arcade", "escape_room",
    "board_game_cafe", "dessert_cafe", "hidden_bar", "movie_theater", "bowling_alley", "craft_beer_bar",
    "ice_cream_shop", "themed_cafe", "hot_pot_restaurant", "live_performance_venue"
]

# App setup
app = dash.Dash(__name__)
server = app.server  # for deployment

app.layout = html.Div([
    html.H1("Date Spot Recommender 💘"),

    dcc.Dropdown(id='relationship', options=[{'label': r, 'value': r} for r in ['romantic', 'friends', 'family', 'sibling', 'marriage']], placeholder="Relationship Type"),
    dcc.Input(id='age', type='number', placeholder='Your Age'),
    dcc.Input(id='partner_age', type='number', placeholder='Partner Age'),
    dcc.Dropdown(id='gender', options=[{'label': g, 'value': g} for g in ['male', 'female', 'other']], placeholder="Your Gender"),
    dcc.Dropdown(id='partner_gender', options=[{'label': g, 'value': g} for g in ['male', 'female', 'other']], placeholder="Partner Gender"),
    dcc.Dropdown(id='interest', options=[{'label': i.replace('_', ' ').title(), 'value': i} for i in interests], placeholder="Interest"),
    dcc.Dropdown(id='district', options=[{'label': d, 'value': d.replace(" ", "_")} for d in districts], placeholder="District"),
    dcc.Slider(id='max_cost', min=100, max=3000, step=100, value=1000, marks={i: f'{i}' for i in range(100, 3100, 500)}),

    html.Button('Get Recommendations', id='submit', n_clicks=0),

    html.Hr(),
    html.Div(id='recommendations')
])

@app.callback(
    Output('recommendations', 'children'),
    Input('submit', 'n_clicks'),
    State('relationship', 'value'),
    State('age', 'value'),
    State('partner_age', 'value'),
    State('gender', 'value'),
    State('partner_gender', 'value'),
    State('interest', 'value'),
    State('district', 'value'),
    State('max_cost', 'value'),
)
def recommend(n_clicks, relationship, age, partner_age, gender, partner_gender, interest, district, max_cost):
    if not all([relationship, age, partner_age, gender, partner_gender, interest, district, max_cost]):
        return "Please fill in all fields."

    user_input = {
        'relationship_type': relationship,
        'age': int(age),
        'partner_age': int(partner_age),
        'gender': gender,
        'partner_gender': partner_gender,
        'interest': interest,
        'district': district,
        'max_cost': max_cost
    }

    eligible = places[(places['district'] == user_input['district']) & (places['max_cost'] <= user_input['max_cost'])].copy()
    if eligible.empty:
        return "No places found for this selection."

    test_df = pd.DataFrame({
        'interest': [user_input['interest']] * len(eligible),
        'district': [user_input['district']] * len(eligible),
        'age': user_input['age'],
        'partner_age': user_input['partner_age'],
        'rating': eligible['rating'].values,
        'user_ratings_total': eligible['user_ratings_total'].values,
        'max_cost': eligible['max_cost'].values
    })

    num_features = ['age', 'partner_age', 'rating', 'user_ratings_total', 'max_cost']
    encoded = encoder.transform(test_df[['interest', 'district']])
    encoded[num_features] = scaler.transform(test_df[num_features])

    eligible['score'] = model.predict(encoded)
    top = eligible.nlargest(3, 'score')[['name', 'interest', 'rating', 'max_cost']]

    return html.Ul([html.Li(f"{row['name']} - {row['interest'].replace('_',' ').title()} - Rating: {row['rating']} - Cost: {row['max_cost']}") for _, row in top.iterrows()])

if __name__ == '__main__':
    app.run_server(debug=True)
