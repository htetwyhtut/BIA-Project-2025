import dash
from dash import html, dcc, Input, Output, State, dash_table
import pandas as pd
import joblib
from xgboost import XGBRanker

app = dash.Dash(__name__)
server = app.server

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

districts = [
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

app.layout = html.Div(style={
    'fontFamily': 'Segoe UI, sans-serif',
    'padding': '2rem',
    'minHeight': '100vh'
}, children=[
    html.Div(style={
        'maxWidth': '800px',
        'margin': '0 auto',
        'backgroundColor': '#fff',
        'padding': '2.5rem',
        'borderRadius': '16px',
        'boxShadow': '0 6px 20px rgba(0, 0, 0, 0.1)'
    }, children=[
        html.H1("AI Date Planner", style={'textAlign': 'center'}),
        html.H4("Plan your perfect date in Bangkok", style={'textAlign': 'center', 'marginBottom': '2rem'}),

        html.Div([
            html.Label("Relationship Type", style={'fontWeight': 'bold'}),
            dcc.RadioItems(
                id='relationship',
                options=[{'label': r.title(), 'value': r} for r in ['romantic', 'friends', 'family', 'sibling', 'marriage']],
                labelStyle={'display': 'inline-block', 'marginRight': '1rem', 'marginBottom': '1rem'},
                inputStyle={"marginRight": "0.5rem"}
            )
        ], style={
            'border': '1px solid #e5e5e5',
            'borderRadius': '10px',
            'padding': '1rem',
            'marginBottom': '1.5rem',
            'backgroundColor': 'rgba(255, 192, 203, 0.95)'
        }),

        html.Div([
            html.Label("Your Age", style={'fontWeight': 'bold'}),
            dcc.Slider(
                id='age',
                min=18, max=100, step=1, value=25,
                marks={i: str(i) for i in range(18, 101, 10)},
                tooltip={"placement": "bottom", "always_visible": True}
            )
        ], style={
            'border': '1px solid #e5e5e5',
            'borderRadius': '10px',
            'padding': '1rem',
            'marginBottom': '1.5rem',
            'backgroundColor': 'rgba(255, 192, 203, 0.95)'
        }),

        html.Div([
            html.Label("Partner Age", style={'fontWeight': 'bold'}),
            dcc.Slider(
                id='partner_age',
                min=18, max=100, step=1, value=25,
                marks={i: str(i) for i in range(18, 101, 10)},
                tooltip={"placement": "bottom", "always_visible": True}
            )
        ], style={
            'border': '1px solid #e5e5e5',
            'borderRadius': '10px',
            'padding': '1rem',
            'marginBottom': '1.5rem',
            'backgroundColor': 'rgba(255, 192, 203, 0.95)'
        }),

        html.Div([
            html.Label("Your Gender", style={'fontWeight': 'bold'}),
            dcc.RadioItems(
                id='gender',
                options=[{'label': g.title(), 'value': g} for g in ['male', 'female', 'other']],
                labelStyle={'display': 'inline-block', 'marginRight': '1rem'},
                inputStyle={"marginRight": "0.5rem"}
            )
        ], style={
            'border': '1px solid #e5e5e5',
            'borderRadius': '10px',
            'padding': '1rem',
            'marginBottom': '1.5rem',
            'backgroundColor': 'rgba(255, 192, 203, 0.95)'
        }),

        html.Div([
            html.Label("Partner Gender", style={'fontWeight': 'bold'}),
            dcc.RadioItems(
                id='partner_gender',
                options=[{'label': g.title(), 'value': g} for g in ['male', 'female', 'other']],
                labelStyle={'display': 'inline-block', 'marginRight': '1rem'},
                inputStyle={"marginRight": "0.5rem"}
            )
        ], style={
            'border': '1px solid #e5e5e5',
            'borderRadius': '10px',
            'padding': '1rem',
            'marginBottom': '1.5rem',
            'backgroundColor': 'rgba(255, 192, 203, 0.95)'
        }),

        html.Div([
            html.Label("Select Interests", style={'fontWeight': 'bold'}),
            dcc.Checklist(
                id='interest',
                options=[{'label': i.replace('_', ' ').title(), 'value': i} for i in interests],
                labelStyle={'display': 'inline-block', 'margin': '0.35rem 1rem 0.35rem 0'}
            )
        ], style={
            'border': '1px solid #e5e5e5',
            'borderRadius': '10px',
            'padding': '1rem',
            'marginBottom': '1.5rem',
            'backgroundColor': 'rgba(255, 192, 203, 0.95)'
        }),

        html.Div([
            html.Label("Select District", style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='district',
                options=[{'label': d, 'value': d.replace(" ", "_")} for d in districts],
                placeholder="District",
                style={'width': '100%'}
            )
        ], style={
            'border': '1px solid #e5e5e5',
            'borderRadius': '10px',
            'padding': '1rem',
            'marginBottom': '1.5rem',
            'backgroundColor': 'rgba(255, 192, 203, 0.95)'
        }),

        html.Div([
            html.Label("Budget (THB)", style={'fontWeight': 'bold'}),
            dcc.Slider(
                id='max_cost',
                min=500,
                max=5000,
                step=500,
                value=2000,
                marks={i: str(i) for i in range(500, 5001, 500)},
                tooltip={"placement": "bottom", "always_visible": True}
            )
        ], style={
            'border': '1px solid #e5e5e5',
            'borderRadius': '10px',
            'padding': '1rem',
            'marginBottom': '1.5rem',
            'backgroundColor': 'rgba(255, 192, 203, 0.95)'
        }),

        html.Button(
            'Get Recommendations',
            id='submit',
            n_clicks=0,
            style={
                'width': '100%',
                'padding': '1rem',
                'fontSize': '1rem',
                'backgroundColor': '#007BFF',
                'color': 'white',
                'border': 'none',
                'borderRadius': '10px',
                'fontWeight': 'bold'
            }
        ),

        html.Hr(),

        html.Div(id='recommendations', style={'marginTop': '2rem'})
    ])
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

    age_avg = sum(age) / 2 if isinstance(age, list) else age
    partner_age_avg = sum(partner_age) / 2 if isinstance(partner_age, list) else partner_age
    max_budget = max_cost[1] if isinstance(max_cost, list) else max_cost

    eligible = places[(places['district'] == district) & (places['max_cost'] <= max_budget)].copy()
    if eligible.empty:
        return "No places found for this selection."

    interest_used = interest[0] if isinstance(interest, list) and interest else interest

    test_df = pd.DataFrame({
        'interest': [interest_used] * len(eligible),
        'district': [district] * len(eligible),
        'age': age_avg,
        'partner_age': partner_age_avg,
        'rating': eligible['rating'].values,
        'user_ratings_total': eligible['user_ratings_total'].values,
        'max_cost': eligible['max_cost'].values
    })

    num_features = ['age', 'partner_age', 'rating', 'user_ratings_total', 'max_cost']
    encoded = encoder.transform(test_df[['interest', 'district']])
    encoded[num_features] = scaler.transform(test_df[num_features])

    eligible['score'] = model.predict(encoded)

    top = eligible.nlargest(3, 'score')[['name', 'interest', 'rating', 'max_cost']].copy()
    top['Google Maps Link'] = top['name'].apply(
        lambda name: f"https://www.google.com/maps/search/?api=1&query={name.replace(' ', '+')}"
    )
    top.columns = ['Place Name', 'Place Type', 'Google Maps Ratings', 'Estimated Cost', 'Google Maps Link']

    return dash_table.DataTable(
        columns=[
            {"name": col, "id": col, "presentation": "markdown"} if "Link" in col else {"name": col, "id": col}
            for col in top.columns
        ],
        data=[
            {**row, "Google Maps Link": f"[Open in Maps]({row['Google Maps Link']})"} for row in top.to_dict('records')
        ],
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'left', 'padding': '10px'},
        style_header={'backgroundColor': '#f1f1f1', 'fontWeight': 'bold'},
        style_data={'backgroundColor': 'white'}
    )

if __name__ == '__main__':
    app.run(debug=True)
