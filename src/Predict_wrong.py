import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from rdkit import DataStructs
import joblib
import os
import traceback

# Get the directory of the current script (Predict.py)
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the trained model and scaler using paths relative to Predict.py
model_path = os.path.join(_CURRENT_DIR, "models_data/CuMOF_XGBoost_best_model.pkl")
scaler_path = os.path.join(_CURRENT_DIR, "models_data/scaler.pkl")

print(f"[DEBUG] Attempting to load model from: {model_path}")
model = joblib.load(model_path)
print(f"[DEBUG] Model loaded successfully.")
print(f"[DEBUG] Attempting to load scaler from: {scaler_path}")
scaler = joblib.load(scaler_path)
print(f"[DEBUG] Scaler loaded successfully.")
print(f"[DEBUG] Scaler expects {scaler.n_features_in_} features.")

# Load label mappings
label_mapping = {"Paddle-wheel": 0, "Other": 1, "rod": 2}
reverse_label_mapping = {v: k for k, v in label_mapping.items()}

# Function to compute molecular feature vector
def featurize_smiles(smiles):
    """
    Converts a SMILES string into a feature vector for model input.
    """
    print(f"[DEBUG] Featurizing SMILES: {smiles}")
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print("[DEBUG] MolFromSmiles returned None")
            return None
        
        # Calculate chemical descriptors
        descriptors_raw_values = []
        for desc_name, desc_func in Descriptors._descList:
            try:
                value = desc_func(mol)
                if isinstance(value, (int, float)):
                    descriptors_raw_values.append(float(value))
                else:
                    descriptors_raw_values.append(0.0) # Default for non-numeric/None
            except Exception:
                descriptors_raw_values.append(0.0) # Default for errors
        
        print(f"[DEBUG] Raw RDKit descriptors count: {len(descriptors_raw_values)}")

        # Adjust descriptor list to 210 elements. If 208, append two zeros.
        # Otherwise, pad or truncate to 210.
        num_target_descriptors = 210
        current_num_descriptors = len(descriptors_raw_values)

        if current_num_descriptors == 208: # Specific case for current environment, user wants to append zeros
            descriptors_padded_list = descriptors_raw_values + [0.0, 0.0]
            print(f"[DEBUG] Padded 208 descriptors to {len(descriptors_padded_list)} by appending 2 zeros.")
        elif current_num_descriptors < num_target_descriptors:
            padding_size = num_target_descriptors - current_num_descriptors
            descriptors_padded_list = descriptors_raw_values + [0.0] * padding_size
            print(f"[DEBUG] Padded {current_num_descriptors} descriptors to {len(descriptors_padded_list)} by appending {padding_size} zeros.")
        elif current_num_descriptors > num_target_descriptors:
            descriptors_padded_list = descriptors_raw_values[:num_target_descriptors]
            print(f"[DEBUG] Truncated {current_num_descriptors} descriptors to {len(descriptors_padded_list)}.")
        else: # current_num_descriptors == num_target_descriptors
            descriptors_padded_list = descriptors_raw_values
            print(f"[DEBUG] Descriptor count is already {len(descriptors_padded_list)}, no adjustment needed.")

        descriptors_final_np = np.array(descriptors_padded_list, dtype=float)
        descriptors_final_np = np.nan_to_num(descriptors_final_np, nan=0.0, posinf=np.finfo(np.float64).max, neginf=np.finfo(np.float64).min)
        print(f"[DEBUG] Final RDKit descriptors count after adjustment: {len(descriptors_final_np)}")
        
        # Calculate molecular fingerprints
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        fingerprint_array = np.zeros((2048,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fingerprint, fingerprint_array)
        fingerprint_array = fingerprint_array.astype(float)
        print(f"[DEBUG] Fingerprint array length: {len(fingerprint_array)}")
        
        # Calculate SMILES string length
        smiles_length = float(len(smiles))
        print(f"[DEBUG] SMILES length: {smiles_length}")
        
        # Combine features
        features = np.concatenate([descriptors_final_np, fingerprint_array, [smiles_length]])
        features = np.nan_to_num(features, nan=0.0, posinf=np.finfo(np.float64).max, neginf=np.finfo(np.float64).min)
        print(f"[DEBUG] Total features combined: {len(features)}")
        return features
    except Exception as e:
        print(f"[DEBUG] Feature extraction failed with error: {e}")
        print(traceback.format_exc())
        return None

# Prediction function
def predict_smiles(smiles_string):
    """
    Takes a SMILES string as input and returns the predicted label or a generic unavailability message.
    """
    generic_unavailable_message = "Prediction unavailable"
    print(f"[DEBUG] Predicting for SMILES: {smiles_string}")
    try:
        if not smiles_string or not isinstance(smiles_string, str):
             print("[DEBUG] Invalid input: not a string or empty")
             return generic_unavailable_message
        smiles_string = smiles_string.strip()
        if not smiles_string:
            print("[DEBUG] Invalid input: empty after stripping")
            return generic_unavailable_message

        features = featurize_smiles(smiles_string)
        if features is None:
            print("[DEBUG] Featurization returned None, prediction unavailable.")
            return generic_unavailable_message
        
        print(f"[DEBUG] Features vector length for scaler: {len(features)}")
        if len(features) != scaler.n_features_in_:
            print(f"[DEBUG] CRITICAL: Feature length mismatch. Got {len(features)}, Scaler expects {scaler.n_features_in_}. Prediction unavailable.")
            return generic_unavailable_message
            
        features_reshaped = features.reshape(1, -1)
        print(f"[DEBUG] Features reshaped to: {features_reshaped.shape}")
        
        features_scaled = scaler.transform(features_reshaped)
        print(f"[DEBUG] Features scaled shape: {features_scaled.shape}")
        
        prediction_numeric = model.predict(features_scaled)[0]
        print(f"[DEBUG] Numeric prediction: {prediction_numeric}")
        predicted_label = reverse_label_mapping[prediction_numeric]
        print(f"[DEBUG] Predicted label: {predicted_label}")
        return predicted_label
    except Exception as e:
        print(f"[DEBUG] Prediction failed in predict_smiles function: {e}")
        print(traceback.format_exc())
        return generic_unavailable_message

if __name__ == "__main__":
    print("[DEBUG] Running Predict.py directly for testing...")
    print(f"[DEBUG] Scaler (loaded in __main__): Expects {scaler.n_features_in_} features.")
    example_smiles_list = [
        "O=C(O)C=1C=CC(=CC1)C2=C(C(C=3C=CC(=CC3)C(=O)O)=C(C(C=4C=CC(=CC4)C(=O)O)=C2C)C)C", # Valid example 1
        "O=C(O)C=1C=CC(=CC1)C2=C(C3=CC=C(C=C3)C(C)(C)C)C(C=4C=CC(=CC4)C(=O)O)=C(C5=CC=C(C=C5)C(C)(C)C)C(C=6C=CC(=CC6)C(=O)O)=C2C7=CC=C(C=C7)C(C)(C)C", # User specified SMILES
        "CC", # Simpler valid example
        "ThisIsAnInvalidSMILESString", # Invalid example
    ]

    print("\nPrediction results (direct script run):")
    for i, s in enumerate(example_smiles_list):
        print(f"\n--- Example {i+1} ---SMILES: {repr(s)}")
        result = predict_smiles(s)
        print(f"  Predicted Type/Error: {result}")

