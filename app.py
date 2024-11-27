from flask import Flask, request, render_template
import joblib 
import pandas as pd

app = Flask(__name__)
modelo = joblib.load('modelo_imoveis.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/prever', methods=['POST'])
def prever():
    Tamanho = float(request.form['Tamanho'])
    N_Quartos = int(request.form['N_Quartos'])
    N_Banheiros = int(request.form['N_Banheiros'])
    Distancia_Min = int(request.form['Distancia_Min'])
    Distancia_Km = float(request.form['Distancia_Km'])
    Piscina = (request.form['Piscina'])
    Piscina_ = Piscina.upper()
    Garagem = int(request.form['Garagem'])
    
    entrada = pd.DataFrame([[Tamanho, N_Quartos, N_Banheiros, Distancia_Min, Distancia_Km, Piscina_, Garagem]], columns=['Tamanho', 'N_Quartos', 'N_Banheiros', 'Distancia_Min', 'Distancia_Km', 'Piscina_', 'Garagem'])
    preco_previsto = modelo.predict(entrada)[0]
    
    return render_template('resultado.html', preco=preco_previsto)

if __name__ == '__main__':
    app.run(debug=True)

  