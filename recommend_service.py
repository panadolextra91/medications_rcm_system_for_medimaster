from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle

# ─────────────────────────────────────────────────────────────────────────────
# Load filtered CSVs (ensure these files are in the same directory as this script)
# ─────────────────────────────────────────────────────────────────────────────
sym_des      = pd.read_csv("symtoms_df_filtered.csv")
precautions  = pd.read_csv("precautions_df_filtered.csv")
workout      = pd.read_csv("workout_df_filtered.csv")
description  = pd.read_csv("description_filtered.csv")
medications  = pd.read_csv("medications_categories_updated.csv")
diets        = pd.read_csv("diets_filtered.csv")

svc = pickle.load(open('svc_2.pkl', 'rb'))

def helper(dis):
    # 1) Description
    desc_list = description.loc[description['Disease'] == dis, 'Description'].tolist()
    desc = " ".join(desc_list)

    # 2) Precautions – use tolist() to get a pure Python list
    pre_df = precautions.loc[
        precautions['Disease'] == dis,
        ['Precaution_1','Precaution_2','Precaution_3','Precaution_4']
    ]
    pre = pre_df.values.tolist()[0] if not pre_df.empty else []

    # 3) Medications – split comma-separated strings into a list
    med_series = medications.loc[medications['Disease'] == dis, 'Medication'].iloc[0]
    if isinstance(med_series, str):
        med = [m.strip() for m in med_series.split(',')]
    else:
        # if it were already a list
        med = list(med_series)

    # 4) Diets
    die = diets.loc[diets['Disease'] == dis, 'Diet'].tolist()

    # 5) Workouts
    wrkout = workout.loc[workout['disease'] == dis, 'workout'].tolist()

    return desc, pre, med, die, wrkout


symptoms_dict = {
    'itching': 0,
    'skin_rash': 1,
    'nodal_skin_eruptions': 2,
    'continuous_sneezing': 3,
    'shivering': 4,
    'chills': 5,
    'joint_pain': 6,
    'stomach_pain': 7,
    'acidity': 8,
    'ulcers_on_tongue': 9,
    'vomiting': 10,
    'burning_micturition': 11,
    'spotting_ urination': 12,
    'fatigue': 13,
    'anxiety': 14,
    'weight_loss': 15,
    'lethargy': 16,
    'cough': 17,
    'high_fever': 18,
    'sunken_eyes': 19,
    'breathlessness': 20,
    'sweating': 21,
    'dehydration': 22,
    'indigestion': 23,
    'headache': 24,
    'nausea': 25,
    'loss_of_appetite': 26,
    'pain_behind_the_eyes': 27,
    'back_pain': 28,
    'diarrhoea': 29,
    'mild_fever': 30,
    'yellowing_of_eyes': 31,
    'swelled_lymph_nodes': 32,
    'malaise': 33,
    'blurred_and_distorted_vision': 34,
    'phlegm': 35,
    'chest_pain': 36,
    'weakness_in_limbs': 37,
    'fast_heart_rate': 38,
    'neck_pain': 39,
    'dizziness': 40,
    'excessive_hunger': 41,
    'drying_and_tingling_lips': 42,
    'slurred_speech': 43,
    'stiff_neck': 44,
    'loss_of_balance': 45,
    'bladder_discomfort': 46,
    'foul_smell_of urine': 47,
    'continuous_feel_of_urine': 48,
    'depression': 49,
    'irritability': 50,
    'muscle_pain': 51,
    'red_spots_over_body': 52,
    'dischromic _patches': 53,
    'watering_from_eyes': 54,
    'rusty_sputum': 55,
    'visual_disturbances': 56,
    'blood_in_sputum': 57,
    'palpitations': 58,
    'pus_filled_pimples': 59,
    'blackheads': 60,
    'scurring': 61,
    'skin_peeling': 62,
    'silver_like_dusting': 63,
    'small_dents_in_nails': 64,
    'inflammatory_nails': 65,
    'blister': 66,
    'red_sore_around_nose': 67,
    'yellow_crust_ooze': 68
}
diseases_list = {6: 'Fungal infection', 1: 'Allergy', 7: 'GERD', 5: 'Drug Reaction',
       8: 'Gastroenteritis', 12: 'Migraine', 2: 'Cervical spondylosis', 11: 'Malaria',
       3: 'Chicken pox', 4: 'Dengue', 15: 'Tuberculosis', 13: 'Pneumonia',
       9: 'Hypoglycemia', 0:'Acne', 16: 'Urinary tract infection', 14: 'Psoriasis',
       10: 'Impetigo'}

# Model Prediction function
def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    for item in patient_symptoms:
        input_vector[symptoms_dict[item]] = 1
    return diseases_list[svc.predict([input_vector])[0]]

# ─────────────────────────────────────────────────────────────────────────────
# Flask application
# ─────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    symptoms = data.get('symptoms', [])
    try:
        disease = get_predicted_value(symptoms)
        desc, pre, med, die, wrk = helper(disease)
        return jsonify({
            'disease': disease,
            'description': desc,
            'precautions': pre,
            'medications': med,
            'diets': die,
            'workouts': wrk
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # Listen on port 5000
    app.run(host='0.0.0.0', port=5001, debug=True)
