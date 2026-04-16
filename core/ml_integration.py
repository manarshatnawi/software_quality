import pickle
from pathlib import Path
from typing import Optional


class DLQualityPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self._load()

    def _load(self):
        try:
            import numpy as np
            import tensorflow as tf
        except ImportError:
            print("Warning: TensorFlow/numpy not available, ML features disabled")
            return

        model_path = Path("models/final_model.keras")
        scaler_path = Path("dataset/scaler.pkl")
        features_path = Path("dataset/feature_names.pkl")
        
        if model_path.exists():
            self.model = tf.keras.models.load_model(model_path)
        if scaler_path.exists():
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
        if features_path.exists():
            with open(features_path, 'rb') as f:
                self.feature_names = pickle.load(f)

    def predict(self, feature_vector):
        if self.model is None or self.scaler is None or self.feature_names is None:
            return None

        try:
            import numpy as np
        except ImportError:
            return None

        fv = feature_vector
        features_dict = {
            'loc': fv.lines_of_code,
            'lloc': fv.lines_of_code,
            'sloc': fv.lines_of_code,
            'comments': int(fv.comment_density * fv.lines_of_code),
            'blank_lines': 0,
            'num_functions': fv.num_functions,
            'num_classes': fv.num_classes,
            'total_complexity': fv.cyclomatic_complexity * max(1, fv.num_functions),
            'avg_complexity': fv.cyclomatic_complexity,
            'max_complexity': fv.cyclomatic_complexity,
            'high_complexity_methods': 1 if fv.cyclomatic_complexity > 10 else 0,
            'maintainability_index': 50.0,
            'comment_density': fv.comment_density * 100,
            'magic_numbers': fv.magic_numbers_count,
            'avg_function_length': fv.avg_function_lines,
            'max_nesting_depth': fv.max_nesting_depth,
            'code_smells': 0,
            'pep8_score': (1 - fv.long_lines_ratio) * 100,
            'duplicate_ratio': fv.duplicate_code_score * 100,
        }
        try:
            X = np.array([[features_dict.get(name, 0) for name in self.feature_names]])
            X_scaled = self.scaler.transform(X)
            pred = self.model.predict(X_scaled, verbose=0)[0][0]
            return max(0, min(100, pred))
        except Exception as e:
            print(f"[DL] error: {e}")
            return None


class GNNClassifier:
    def __init__(self):
        self.model = None
        self._load()

    def _load(self):
        model_path = Path("models/gnn_model.pt")
        if model_path.exists():
            try:
                import torch
                try:
                    from .models import CodeQualityGNN
                except ImportError:
                    from models import CodeQualityGNN
                self.model = CodeQualityGNN(vocab_size=79, hidden_dim=64, num_classes=3)
                self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
                self.model.eval()
            except Exception as e:
                print(f"[GNN] load error: {e}")

    def predict(self, code: str):
        if self.model is None:
            return None
        try:
            import torch
            from torch_geometric.data import Data
            
            # استخراج العقد والحواف محلياً بدلاً من الاستيراد الخارجي
            nodes, edges = self._get_nodes_edges(code)
            
            if not nodes:
                return None
            
            x = torch.tensor(nodes, dtype=torch.long).unsqueeze(1)
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            batch = torch.zeros(len(nodes), dtype=torch.long)
            
            with torch.no_grad():
                logits = self.model(x, edge_index, batch)
                pred_class = logits.argmax(dim=1).item()
            
            mapping = {0: "clean_code", 1: "bad_naming", 2: "complexity_overload"}
            return mapping.get(pred_class, "clean_code")
        except Exception as e:
            print(f"[GNN] predict error: {e}")
            return None

    def _get_nodes_edges(self, code: str):
        """استخراج العقد والحواف من الكود باستخدام AST"""
        import ast
        
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return [], []
        
        nodes = []
        edges = []
        node_map = {}
        
        def traverse(node, parent_idx=None):
            node_type = type(node).__name__
            if node_type not in node_map:
                node_map[node_type] = len(nodes)
                nodes.append(hash(node_type) % 79)  # vocab_size=79
            curr_idx = node_map[node_type]
            
            if parent_idx is not None:
                edges.append((parent_idx, curr_idx))
            
            for child in ast.iter_child_nodes(node):
                traverse(child, curr_idx)
        
        traverse(tree)
        return nodes, edges


# Lazy instantiation
_dl_predictor = None
_gnn_classifier = None


def dl_predictor():
    global _dl_predictor
    if _dl_predictor is None:
        _dl_predictor = DLQualityPredictor()
    return _dl_predictor


def gnn_classifier():
    global _gnn_classifier
    if _gnn_classifier is None:
        _gnn_classifier = GNNClassifier()
    return _gnn_classifier