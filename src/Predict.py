import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from rdkit import DataStructs
import joblib
import os

# 获取当前脚本路径
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 加载模型与标准化器
model_path = os.path.join(_CURRENT_DIR, "models_data/CuMOF_XGBoost_best_model.pkl")
scaler_path = os.path.join(_CURRENT_DIR, "models_data/scaler.pkl")
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# 标签映射
label_mapping = {'Paddle-wheel': 0, 'Other': 1, 'rod': 2}
reverse_label_mapping = {v: k for k, v in label_mapping.items()}

# 特征提取函数
def featurize_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        # RDKit描述符
        descriptors_raw = []
        for name, func in Descriptors._descList:
            try:
                val = func(mol)
                descriptors_raw.append(float(val) if isinstance(val, (int, float)) else 0.0)
            except Exception:
                descriptors_raw.append(0.0)

        # 保证维度为210（默认208 + 2个0补齐）
        if len(descriptors_raw) == 208:
            descriptors_raw += [0.0, 0.0]
        elif len(descriptors_raw) < 210:
            descriptors_raw += [0.0] * (210 - len(descriptors_raw))
        elif len(descriptors_raw) > 210:
            descriptors_raw = descriptors_raw[:210]

        descriptors_np = np.array(descriptors_raw, dtype=float)
        descriptors_np = np.nan_to_num(descriptors_np, nan=0.0, posinf=np.finfo(np.float64).max, neginf=np.finfo(np.float64).min)

        # Morgan指纹（2048位）
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        fingerprint_array = np.zeros((2048,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fingerprint, fingerprint_array)
        fingerprint_array = fingerprint_array.astype(float)

        # SMILES长度
        smiles_len = float(len(smiles))

        # 合并特征：210 + 2048 + 1 = 2259
        features = np.concatenate([descriptors_np, fingerprint_array, [smiles_len]])
        features = np.nan_to_num(features, nan=0.0, posinf=np.finfo(np.float64).max, neginf=np.finfo(np.float64).min)
        return features
    except Exception as e:
        print(f"[ERROR] Feature extraction failed: {e}")
        return None

# 预测函数
def predict_smiles(smiles):
    features = featurize_smiles(smiles)
    if features is None or len(features) != scaler.n_features_in_:
        return "Prediction unavailable"
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)[0]
    return reverse_label_mapping.get(prediction, "Unknown")

# 示例测试
if __name__ == "__main__":
    example_smiles_list = [
        "O=C(O)C=1C=CC(=CC1)C2=C(C(C=3C=CC(=CC3)C(=O)O)=C(C(C=4C=CC(=CC4)C(=O)O)=C2C)C)C",
        "O=C(O)C=1C=CC(=CC1)C2=C(C3=CC=C(C=C3)C(C)(C)C)C(C=4C=CC(=CC4)C(=O)O)=C(C5=CC=C(C=C5)C(C)(C)C)C(C=6C=CC(=CC6)C(=O)O)=C2C7=CC=C(C=C7)C(C)(C)C",
        "CC",
        "ThisIsAnInvalidSMILESString"
    ]
    print("Prediction results:")
    for smiles in example_smiles_list:
        result = predict_smiles(smiles)
        print(f"SMILES: {smiles} -> Predicted Type: {result}")
