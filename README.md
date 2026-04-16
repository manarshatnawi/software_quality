# 🔬 QualityAnalyzPromet

نظام ذكي لتحليل وإصلاح جودة كود Python بشكل تكراري عبر **AST + DL Model + Groq API**.

---

## 🗂️ هيكل المشروع

```
QualityAnalyzPromet/
│
├── core/                        ← حزمة المنطق الأساسي
│   ├── __init__.py              ← واجهة عامة نظيفة
│   ├── models.py                ← FeatureVector · QualityReport · IterationRecord
│   ├── classifier.py            ← ProblemClassifier (DL model + rules)
│   ├── scorer.py                ← QualityScorer (حساب الدرجات)
│   └── config.py                ← الإعدادات والثوابت المركزية
│
├── tests/                       ← اختبارات الوحدة
│   ├── test_classifier.py       ← اختبارات التصنيف
│   ├── test_scorer.py           ← اختبارات الدرجات
│   └── ...
│
├── models/                      ← نماذج التعلم العميق
│   ├── final_model.keras        ← النموذج المدرب
│   └── model_performance.txt    ← أداء النموذج
│
├── dataset/                     ← بيانات التدريب
│   ├── X_train.npy
│   ├── y_train.npy
│   └── ...
│
├── app.py                       ← واجهة Streamlit
├── requirements.txt
├── .env.example
└── README.md
```

---

## ⚙️ التثبيت

```bash
pip install -r requirements.txt
cp .env.example .env   # أضف GROQ_API_KEY
```

---

## 🚀 التشغيل

```bash
streamlit run app.py
```

---

## 🔄 Pipeline

```
User Code
  ↓  ASTAnalyzer          — استخراج 20+ مقياس
  ↓  ProblemClassifier    — DL Model للتنبؤ بالمشاكل (أو قواعد احتياطية)
  ↓  QualityScorer        — حساب الدرجة والـ grade
  ↓  RepairPromptBuilder  — بناء prompt مستهدف
  ↓  Groq API (llama-3.3-70b)
  ↓  Re-analyze → Compare → Loop
  ↓  IterativeRefiner     — يتوقف عند بلوغ الهدف
```

---

## 📊 أبعاد التقييم

| البُعد            | الوزن | ما يقيسه                          |
|-------------------|-------|-----------------------------------|
| Complexity        | 20%   | Cyclomatic · Cognitive · Nesting   |
| Readability       | 20%   | التسمية · طول الأسطر              |
| Documentation     | 20%   | Docstrings · التعليقات             |
| Best Practices    | 20%   | Type hints · Exception handling    |
| Maintainability   | 20%   | متوسط الأبعاد الأربعة             |

---

## 🧪 الاختبارات

```bash
python -m pytest tests/
```

---

## ⚙️ تخصيص الإعدادات

```python
# config.py — كل الثوابت في مكان واحد
THRESHOLDS.max_cyclomatic_complexity = 10
WEIGHTS.high_penalty = 8
```

---

## 🏗️ مبادئ التصميم

| المبدأ | التطبيق |
|--------|---------|
| **Single Responsibility** | كل ملف مسؤولية واحدة |
| **No Magic Numbers** | جميع الثوابت في `config.py` |
| **Open/Closed** | يمكن استبدال `ProblemClassifier` بنموذج ML |
| **Testable** | كل وحدة مستقلة قابلة للاختبار |</content>
<parameter name="filePath">c:\Users\RB\Desktop\All Codes system\QualityAnalyzPromet\README.md