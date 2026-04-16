"""
models.py - نماذج التعلم الآلي والحلقات الذكية
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch.nn import Linear
import tree_sitter_python as tspython
from tree_sitter import Language, Parser
import radon.complexity as cc
import re
import time
import pandas as pd
from torch_geometric.data import Data

# ==========================================
# 1. محرك التحليل اللغوي (Parser)
# ==========================================
PY_LANGUAGE = Language(tspython.language())
parser = Parser(PY_LANGUAGE)

vocab = {}
label_map = {"Clean_Code": 0, "Bad_Naming": 1, "High_Complexity": 2}

def code_to_graph_data(code_snippet):
    tree = parser.parse(bytes(code_snippet, "utf8"))
    nodes, edges = [], []
    def traverse(node, parent_idx=None):
        curr_idx = len(nodes)
        if node.type not in vocab: vocab[node.type] = len(vocab)
        nodes.append(vocab[node.type])
        if parent_idx is not None: edges.append((parent_idx, curr_idx))
        for child in node.children: traverse(child, curr_idx)
    traverse(tree.root_node)
    if not nodes: return None
    x = torch.tensor(nodes, dtype=torch.long).unsqueeze(1)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return Data(x=x, edge_index=edge_index)

# ==========================================
# 2. الـ GNN Model
# ==========================================
class CodeQualityGNN(torch.nn.Module):
    def __init__(self, vocab_size=79, hidden_dim=32, num_classes=3):
        """
        vocab_size: عدد أنواع العقد التي وجدناها (79)
        hidden_dim: حجم الذاكرة المخفية للموديل (32)
        num_classes: عدد التصنيفات (3: نظيف، تسمية سيئة، معقد)
        """
        super(CodeQualityGNN, self).__init__()

        # 1. طبقة التضمين (Embedding Layer)
        # تحول الرقم الواحد (مثل رقم 5 الذي يمثل 'identifier') إلى متجه من 32 رقماً
        self.embedding = torch.nn.Embedding(vocab_size, hidden_dim)

        # 2. طبقات الشبكة العصبية الرسومية (Graph Convolutional Layers)
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        # 3. طبقة التصنيف النهائية (Classifier)
        self.classifier = Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index, batch):
        """
        هذه الدالة تُسمى تلقائياً عند إعطاء الكود للموديل
        """
        # استخراج الخصائص من الأرقام
        x = self.embedding(x.squeeze(1))

        # المرور عبر الطبقة الأولى + دالة تنشيط (ReLU) لكسر الخطية
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        # المرور عبر الطبقة الثانية
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # 4. التجميع (Pooling)
        # نجمع معلومات كل العقد الـ 26 في متجه واحد يمثل الكود بأكمله
        x = global_mean_pool(x, batch)

        # 5. التصنيف النهائي
        out = self.classifier(x)

        return out # ستخرج 3 أرقام تمثل احتمالات كل فئة

# ==========================================
# 3. محرك القواعد الذكي (The Rule-Based Critic)
# ==========================================
def rule_based_critic(code_snippet):
    # فحص التعقيد
    blocks = cc.cc_visit(code_snippet)
    max_complexity = 0
    for block in blocks:
        if block.complexity > max_complexity:
            max_complexity = block.complexity
    if max_complexity > 3: return "High_Complexity"

    # فحص التسميات
    bad_names = re.findall(r'\b([a-zA-Z]{1,2})\b\s*=', code_snippet)
    bad_func_names = re.findall(r'def\s+([a-zA-Z]{1,2})\s*\(', code_snippet)
    if len(bad_names) >= 2 or len(bad_func_names) >= 1:
        return "Bad_Naming"

    return "Clean_Code"

# ==========================================
# 4. الدالة الوهمية للإصلاح
# ==========================================
def local_ai_fixer(code, predicted_label_name):
    if predicted_label_name == "Bad_Naming":
        return "def calculate_sum(first_number, second_number):\n    total_sum = first_number + second_number\n    return total_sum\n"
    elif predicted_label_name == "High_Complexity":
        return "def process_data(data):\n    # Refactored by AI to reduce complexity\n    return [x for x in data if x > 0]\n"
    return code

# ==========================================
# 5. الحلقة الذكية (النظام الهجين)
# ==========================================
def autonomous_refine_loop(initial_code, model):
    print("\n" + "="*50)
    print("🤖 بدء الحلقة الذكية (النظام الهجين: GNN + Rule-Based)...")
    print("="*50)

    current_code = initial_code
    max_iterations = 3

    for i in range(max_iterations):
        print(f"\n--- التكرار رقم {i+1} ---")

        # استخدام محرك القواعد الدقيق لاتخاذ القرار
        predicted_label = rule_based_critic(current_code)

        print(f"🔴 تشخيص النظام: [{predicted_label}]")

        if predicted_label == "Clean_Code":
            print("✅ نجاح! تم الوصول لفايب كود احترافي بأعلى جودة.")
            print("="*50)
            return current_code

        print(f"🛠️ إرسال التحسينات المطلوبة للـ AI...")
        current_code = local_ai_fixer(current_code, predicted_label)
        time.sleep(1)

    print("\n⚠️ انتهت التكرارات القصوى.")
    print("="*50)
    return current_code

# ==========================================
# 6. بناء مجموعة البيانات للتدريب
# ==========================================
def build_graph_dataset(csv_path='dataset/dataset.csv'):
    """بناء مجموعة البيانات الرسومية من CSV"""
    df = pd.read_csv(csv_path)
    graph_dataset = []

    # بناء القاموس أولاً
    for code in df['Code']:
        code_to_graph_data(code)

    # ثم بناء الرسوم
    for index, row in df.iterrows():
        graph = code_to_graph_data(row['Code'])
        if graph:
            graph.y = torch.tensor(label_map[row['Label']], dtype=torch.long)
            graph_dataset.append(graph)

    return graph_dataset