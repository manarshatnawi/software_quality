"""
Smart Code Quality Analyzer — Full Streamlit Interface
يعمل مع: code_quality_system.py (في نفس المجلد)
"""

import streamlit as st
import plotly.graph_objects as go
import math
import os
import difflib
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from core import (
    ASTAnalyzer, ProblemClassifier, QualityScorer,
    IterativeRefiner, FeatureVector, QualityReport, IterationRecord
)

# ─────────────────────────────────────────────────────────────────────
# Language Support
# ─────────────────────────────────────────────────────────────────────
LANGUAGES = {
    "العربية": {
        "title": "محلل جودة الكود الذكي",
        "subtitle": "نظام تحليل وإصلاح الكود بشكل تكراري ذكي",
        "input_title": "📝 أدخل الكود",
        "example_button": "📋 مثال",
        "analyze_button": "🔍 تحليل",
        "refine_button": "✨ إصلاح",
        "results_title": "📊 النتائج",
        "quality_score": "درجة الجودة",
        "problems_found": "المشاكل المكتشفة",
        "improved_code": "الكود المحسن",
        "iterations": "التكرارات",
        "grade": "التقدير",
        "loading": "جاري التحليل...",
        "success": "تم التحليل بنجاح!",
        "error": "حدث خطأ أثناء التحليل"
    },
    "English": {
        "title": "Smart Code Quality Analyzer",
        "subtitle": "Intelligent iterative code analysis and refinement system",
        "input_title": "📝 Enter Code",
        "example_button": "📋 Example",
        "analyze_button": "🔍 Analyze",
        "refine_button": "✨ Refine",
        "results_title": "📊 Results",
        "quality_score": "Quality Score",
        "problems_found": "Problems Found",
        "improved_code": "Improved Code",
        "iterations": "Iterations",
        "grade": "Grade",
        "loading": "Analyzing...",
        "success": "Analysis completed successfully!",
        "error": "Error during analysis"
    }
}

# Language selector
if 'language' not in st.session_state:
    st.session_state.language = "العربية"

col1, col2 = st.columns([6, 1])
with col2:
    lang = st.selectbox("", ["العربية", "English"], 
                       index=["العربية", "English"].index(st.session_state.language),
                       key="lang_selector")
    st.session_state.language = lang

texts = LANGUAGES[st.session_state.language]

# ─────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=texts["title"],
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────
# Enhanced CSS with Modern Design
# ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Cairo:wght@400;600;700&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {{ 
    font-family: {'Cairo' if st.session_state.language == 'العربية' else 'Poppins'}, sans-serif; 
    background: 
        radial-gradient(circle at 20% 50%, rgba(120, 119, 198, 0.3) 0%, transparent 50%),
        radial-gradient(circle at 80% 20%, rgba(255, 119, 198, 0.3) 0%, transparent 50%),
        radial-gradient(circle at 40% 80%, rgba(120, 219, 255, 0.3) 0%, transparent 50%),
        linear-gradient(135deg, #667eea 0%, #764ba2 25%, #f093fb 50%, #f5576c 75%, #4facfe 100%);
    background-size: 100% 100%, 100% 100%, 100% 100%, 400% 400%;
    background-attachment: fixed;
    animation: gradientShift 15s ease infinite;
    color: #333;
    min-height: 100vh;
    position: relative;
}}

@keyframes gradientShift {{
    0% {{ background-position: 0% 50%; }}
    50% {{ background-position: 100% 50%; }}
    100% {{ background-position: 0% 50%; }}
}}

html::before {{
    content: '';
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: 
        url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 1000"><defs><pattern id="stars" x="0" y="0" width="100" height="100" patternUnits="userSpaceOnUse"><circle cx="20" cy="20" r="1" fill="rgba(255,255,255,0.1)"/><circle cx="80" cy="80" r="0.5" fill="rgba(255,255,255,0.08)"/><circle cx="50" cy="10" r="0.8" fill="rgba(255,255,255,0.06)"/><circle cx="10" cy="90" r="0.3" fill="rgba(255,255,255,0.05)"/></pattern></defs><rect width="1000" height="1000" fill="url(%23stars)"/></svg>'),
        radial-gradient(circle at 30% 70%, rgba(255,255,255,0.1) 0%, transparent 50%),
        radial-gradient(circle at 70% 30%, rgba(255,255,255,0.05) 0%, transparent 50%);
    pointer-events: none;
    z-index: -1;
}}
code, pre {{ 
    font-family: 'JetBrains Mono', monospace !important; 
    font-size: .85rem !important; 
    background: #f8f9fa !important;
    border-radius: 8px !important;
    padding: 12px !important;
}}

.hero {{
    background: 
        linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(255,255,255,0.9) 50%, rgba(255,255,255,0.85) 100%),
        linear-gradient(45deg, #667eea, #764ba2, #f093fb, #f5576c);
    background-size: 100% 100%, 300% 300%;
    background-position: 0% 0%, 0% 0%;
    animation: heroGlow 8s ease-in-out infinite alternate;
    padding: 3rem 2.5rem 2.5rem; 
    border-radius: 24px; 
    margin-bottom: 2.5rem;
    color: #333; 
    text-align: center; 
    box-shadow: 
        0 20px 60px rgba(0,0,0,.15),
        inset 0 1px 0 rgba(255,255,255,0.6);
    position: relative;
    overflow: hidden;
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255,255,255,0.3);
}}

@keyframes heroGlow {{
    0% {{ 
        background-position: 0% 0%, 0% 0%;
        box-shadow: 0 20px 60px rgba(0,0,0,.15), inset 0 1px 0 rgba(255,255,255,0.6);
    }}
    100% {{ 
        background-position: 0% 0%, 100% 100%;
        box-shadow: 0 25px 80px rgba(0,0,0,.2), inset 0 1px 0 rgba(255,255,255,0.8);
    }}
}}

.hero::before {{
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: 
        radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%),
        conic-gradient(from 0deg, transparent, rgba(255,119,198,0.1), transparent);
    animation: rotate 20s linear infinite;
    pointer-events: none;
}}

@keyframes rotate {{
    from {{ transform: rotate(0deg); }}
    to {{ transform: rotate(360deg); }}
}}

.hero h1 {{ 
    font-size: 2.5rem; 
    margin: 0; 
    letter-spacing: 1px;
    font-weight: 700;
    text-shadow: 0 2px 4px rgba(0,0,0,.1);
    background: linear-gradient(45deg, #667eea, #764ba2, #f093fb);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}}
.hero .sub {{ 
    opacity: .8; 
    margin: 1rem 0 0; 
    font-size: 1.1rem;
    font-weight: 500;
    color: #555;
}}

.sec-title {{
    font-size: 1.4rem;
    font-weight: 600;
    color: #4a5568;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 3px solid #667eea;
    display: inline-block;
}}

.stTextArea textarea {{
    border-radius: 12px !important;
    border: 2px solid #e2e8f0 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 14px !important;
    background: #ffffff !important;
    box-shadow: 0 4px 12px rgba(0,0,0,.1) !important;
}}

.stButton button {{
    border-radius: 12px !important;
    font-weight: 600 !important;
    padding: 12px 24px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 12px rgba(0,0,0,.15) !important;
}}

.stButton button:hover {{
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(0,0,0,.25) !important;
}}

.metric-card {{
    background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
    padding: 1.5rem;
    border-radius: 16px;
    box-shadow: 0 8px 24px rgba(0,0,0,.12);
    border: 1px solid #e2e8f0;
    margin-bottom: 1rem;
}}

.problem-card {{
    background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
    border-left: 4px solid #ef4444;
    padding: 1rem;
    border-radius: 12px;
    margin-bottom: 0.5rem;
}}

.success-card {{
    background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
    border-left: 4px solid #10b981;
    padding: 1rem;
    border-radius: 12px;
    margin-bottom: 0.5rem;
}}

.stProgress .st-bo {{
    background-color: #667eea !important;
}}

.stTabs [data-baseweb="tab-list"] {{
    gap: 8px;
}}

.stTabs [data-baseweb="tab"] {{
    background-color: #f1f5f9;
    border-radius: 8px 8px 0 0;
    border: none;
    color: #64748b;
    font-weight: 500;
}}

.stTabs [data-baseweb="tab"][aria-selected="true"] {{
    background-color: #667eea;
    color: white;
}}

.stAlert {{
    border-radius: 12px !important;
    border: none !important;
}}

.language-selector {{
    background: rgba(255,255,255,0.9);
    backdrop-filter: blur(10px);
    border-radius: 12px;
    padding: 8px;
    box-shadow: 0 4px 12px rgba(0,0,0,.1);
}}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────
# Hero with University Logo
# ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero">
  <div class="university-logo">🏛️ {'جامعة أربد الأهلية' if st.session_state.language == 'العربية' else 'Al-Ahliyya Amman University'}</div>
  <h1>🔬 {texts['title']}</h1>
  <div class="sub">{'تحليل AST → استخراج الميزات → تصنيف المشاكل → إصلاح ذكي تكراري عبر Groq' if st.session_state.language == 'العربية' else 'AST Parsing → Feature Extraction → Problem Classification → Iterative AI Repair via Groq'}</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"### ⚙️ {'الإعدادات' if st.session_state.language == 'العربية' else 'Settings'}")
    st.divider()
    st.markdown(f"**🔁 {'إعدادات الحلقة التكرارية' if st.session_state.language == 'العربية' else 'Iteration Settings'}**")
    max_iter = st.slider(f"{'أقصى عدد تكرارات' if st.session_state.language == 'العربية' else 'Max Iterations'}", 1, 8, 4)
    target   = st.slider(f"{'درجة الهدف' if st.session_state.language == 'العربية' else 'Target Score'}", 50, 100, 85)
    patience = st.slider(f"{'الصبر' if st.session_state.language == 'العربية' else 'Patience'}", 1, 4, 2)
    min_imp  = st.slider(f"{'أدنى تحسن مقبول' if st.session_state.language == 'العربية' else 'Min Improvement'}", 1.0, 10.0, 3.0, .5)
    st.divider()
    analyze_only = st.checkbox(f"{'تحليل فقط (بدون إصلاح)' if st.session_state.language == 'العربية' else 'Analyze Only (No Refinement)'}", value=False)
    

# قراءة مفتاح Groq من secrets.toml (مخفي عن المستخدم)
try:
    groq_key = st.secrets["GROQ_API_KEY"]
except:
    groq_key = os.environ.get("GROQ_API_KEY", "")
    if not groq_key:
        warning_msg = """
⚠️ **مفتاح Groq API غير موجود**

يمكنك استخدام وضع **تحليل فقط** بدون المفتاح.
أما وضع الإصلاح التكراري فيحتاج إلى `GROQ_API_KEY` داخل `.streamlit/secrets.toml`.
        """ if st.session_state.language == "العربية" else """
⚠️ **Groq API Key Not Found**

You can use **Analyze Only** mode without the key.
Refinement mode requires `GROQ_API_KEY` in `.streamlit/secrets.toml`.
        """
        st.warning(warning_msg)

# ─────────────────────────────────────────────────────────────────────
# Example Code (Very Bad Code)
# ─────────────────────────────────────────────────────────────────────
EXAMPLE = '''\
import os
x=1
y=2
def f(a,b):
    if a>0:
        return a+b
    else:
        return a-b
z=x+y
print(z)
'''

# ─────────────────────────────────────────────────────────────────────
# Input Area
# ─────────────────────────────────────────────────────────────────────
c1, c2 = st.columns([4,1])
with c1:
    st.markdown(f'<div class="sec-title">{texts["input_title"]}</div>', unsafe_allow_html=True)
with c2:
    st.write("")
    if st.button(texts["example_button"], use_container_width=True):
        st.session_state["load_ex"] = True

placeholder_text = "الصق كودك Python هنا …" if st.session_state.language == "العربية" else "Paste your Python code here..."
user_code = st.text_area(texts["input_title"], value=EXAMPLE if st.session_state.get("load_ex") else "",
                          height=260, placeholder=placeholder_text,
                          label_visibility="collapsed")

button_text = f"🚀 {'تشغيل التحليل والإصلاح' if st.session_state.language == 'العربية' else 'Run Analysis & Refinement'}"
run = st.button(button_text, type="primary", use_container_width=True)

# ─────────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────────
def gc(grade):
    return {"A":"#00c853","B":"#64dd17","C":"#ffd600","D":"#ff6d00","F":"#d50000"}.get(grade,"#888")

def render_grade(grade, score):
    st.markdown(f"""
    <div class="grade-wrap">
      <span class="grade-badge g{grade}">{grade}</span>
      <div style="color:#aaa;margin-top:.4rem;font-size:.9rem;">
        <b style="color:#03dac6">{score:.1f}</b> / 100
      </div>
    </div>""", unsafe_allow_html=True)

def render_score_cards(rep):
    dims = [("🏆 Overall",rep.overall_score),("📖 Readability",rep.readability_score),
            ("🔧 Maintain.",rep.maintainability_score),("🧩 Complexity",rep.complexity_score),
            ("📚 Docs",rep.documentation_score),("✅ Best Prac.",rep.best_practices_score)]
    cols = st.columns(6)
    for i,(lbl,val) in enumerate(dims):
        color = "#03dac6" if val>=70 else "#ffa726" if val>=50 else "#ef5350"
        with cols[i]:
            st.markdown(f'<div class="sc"><div class="val" style="color:{color}">{val:.0f}</div>'
                        f'<div class="lbl">{lbl}</div></div>', unsafe_allow_html=True)

def render_problems(problems):
    if not problems:
        st.success("✅ لا مشاكل مكتشفة")
        return
    icons = {"high":"🔴","medium":"🟡","low":"🔵"}
    for p in problems:
        st.markdown(f"""
        <div class="pb pb-{p['severity']}">
          <div class="pb-title">{icons.get(p['severity'],'⚪')} [{p['severity'].upper()}] {p['type']}</div>
          <div class="pb-desc">{p['description']}</div>
          <div class="pb-fix">💡 {p['suggestion']}</div>
        </div>""", unsafe_allow_html=True)

def radar_chart(rep, title=""):
    cats = ["Readability","Complexity","Docs","Best Practices","Maintainability"]
    vals = [rep.readability_score, rep.complexity_score, rep.documentation_score,
            rep.best_practices_score, rep.maintainability_score]
    fig = go.Figure(go.Scatterpolar(
        r=vals+[vals[0]], theta=cats+[cats[0]],
        fill='toself', line_color='#bb86fc', fillcolor='rgba(187,134,252,0.18)',
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True,range=[0,100],color="#555"),bgcolor="#1e1e2e"),
        paper_bgcolor="#0d0d1a", plot_bgcolor="#0d0d1a", font=dict(color="white"),
        margin=dict(t=35,b=20,l=20,r=20), height=290,
        title=dict(text=title,font=dict(size=13,color="#bb86fc"),x=.5),
    )
    return fig

def progress_chart(history):
    iters  = [r["iteration"] for r in history]
    scores = [r["score"] for r in history]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=iters, y=scores, mode='lines+markers+text',
        line=dict(color='#03dac6',width=3),
        marker=dict(size=11,color='#bb86fc',line=dict(color='#fff',width=2)),
        text=[f"{s:.0f}" for s in scores], textposition="top center",
        textfont=dict(color="#fff",size=11),
    ))
    fig.add_hline(y=target, line_dash="dash", line_color="#ffa726",
                  annotation_text=f"Target ({target})", annotation_position="top right",
                  annotation_font_color="#ffa726")
    fig.update_layout(
        xaxis=dict(title="Iteration",color="#aaa",tickmode='linear',dtick=1),
        yaxis=dict(title="Score",range=[0,108],color="#aaa"),
        paper_bgcolor="#0d0d1a", plot_bgcolor="#1e1e2e", font=dict(color="white"),
        margin=dict(t=30,b=40,l=40,r=20), height=300, showlegend=False,
    )
    return fig

def bar_compare(rep0, repf):
    cats   = ["Readability","Complexity","Docs","Best Practices","Maintainability","Overall"]
    before = [rep0.readability_score,rep0.complexity_score,rep0.documentation_score,
              rep0.best_practices_score,rep0.maintainability_score,rep0.overall_score]
    after  = [repf.readability_score,repf.complexity_score,repf.documentation_score,
              repf.best_practices_score,repf.maintainability_score,repf.overall_score]
    fig = go.Figure()
    fig.add_trace(go.Bar(name='قبل',x=cats,y=before,marker_color='#ef5350',opacity=.85))
    fig.add_trace(go.Bar(name='بعد', x=cats,y=after, marker_color='#03dac6',opacity=.9))
    fig.update_layout(
        barmode='group', xaxis=dict(color="#aaa"), yaxis=dict(range=[0,115],color="#aaa"),
        paper_bgcolor="#0d0d1a", plot_bgcolor="#1e1e2e", font=dict(color="white"),
        legend=dict(bgcolor="#1e1e2e"), margin=dict(t=20,b=40,l=40,r=20), height=310,
    )
    return fig

def feature_table(fv):
    rows = [
        ("Lines of Code",fv.lines_of_code),("Functions",fv.num_functions),
        ("Classes",fv.num_classes),(f"Cyclomatic Complexity",f"{fv.cyclomatic_complexity:.1f}"),
        ("Cognitive Complexity",f"{fv.cognitive_complexity:.1f}"),
        ("Max Nesting Depth",fv.max_nesting_depth),
        ("Avg Function Lines",f"{fv.avg_function_lines:.1f}"),
        ("Max Function Lines",fv.max_function_lines),
        ("Short Names Ratio",f"{fv.short_names_ratio:.0%}"),
        ("Naming Convention",f"{fv.naming_convention_score:.0%}"),
        ("Docstring Coverage",f"{fv.docstring_coverage:.0%}"),
        ("Comment Density",f"{fv.comment_density:.0%}"),
        ("Bare Excepts",fv.bare_except_count),
        ("Magic Numbers",fv.magic_numbers_count),
        ("Global Vars",fv.global_vars_count),
        ("Long Lines Ratio",f"{fv.long_lines_ratio:.0%}"),
        ("Type Hints","✅" if fv.uses_type_hints else "❌"),
        ("Type Hint Coverage",f"{fv.type_hint_coverage:.0%}"),
        ("List Comprehensions","✅" if fv.uses_list_comp else "❌"),
        ("Generators","✅" if fv.uses_generators else "❌"),
    ]
    html = '<table class="ftable"><tr><th>Feature</th><th>Value</th></tr>'
    for name,val in rows:
        html += f"<tr><td>{name}</td><td><b>{val}</b></td></tr>"
    html += "</table>"
    st.markdown(html, unsafe_allow_html=True)

def diff_view(old, new):
    diff = list(difflib.unified_diff(old.splitlines(), new.splitlines(), lineterm='', n=3))
    if not diff:
        st.info("لا تغييرات في الكود.")
        return
    html = '<div style="background:#0d0d1a;border-radius:8px;padding:.8rem;overflow-x:auto;max-height:400px;font-family:JetBrains Mono,monospace;font-size:.78rem;">'
    for line in diff[2:]:
        if line.startswith('+'):
            html += f'<div style="background:#0d2b1a;color:#80cbc4;white-space:pre;padding:.05rem .3rem;">{line}</div>'
        elif line.startswith('-'):
            html += f'<div style="background:#2b0d0d;color:#ef9a9a;white-space:pre;padding:.05rem .3rem;text-decoration:line-through;">{line}</div>'
        elif line.startswith('@@'):
            html += f'<div style="color:#555;white-space:pre;padding:.05rem .3rem;">{"─"*50}</div>'
        else:
            html += f'<div style="color:#555;white-space:pre;padding:.05rem .3rem;">{line}</div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────
# Main Logic
# ─────────────────────────────────────────────────────────────────────
if run:
    if not user_code.strip():
        warning_msg = "⚠️ أدخل كوداً أولاً." if st.session_state.language == "العربية" else "⚠️ Enter code first."
        st.warning(warning_msg)
        st.stop()
    if not analyze_only and not groq_key:
        error_msg = "❌ لا يمكن الإصلاح بدون مفتاح Groq API." if st.session_state.language == "العربية" else "❌ Cannot refine without Groq API key."
        st.error(error_msg)
        st.stop()
    if groq_key:
        os.environ["GROQ_API_KEY"] = groq_key

    # ════════════════════════════════════════
    # 1. Initial Analysis
    # ════════════════════════════════════════
    analysis_title = "📊 التحليل الأولي" if st.session_state.language == "العربية" else "📊 Initial Analysis"
    st.markdown(f'<div class="sec-title">{analysis_title}</div>', unsafe_allow_html=True)
    try:
        fv0 = ASTAnalyzer(user_code).build_feature_vector()
        cat0, p0 = ProblemClassifier().classify(fv0)
        rep0 = QualityScorer().score(fv0, p0)
        rep0.problem_category = cat0
    except SyntaxError as e:
        st.error(f"❌ خطأ في بنية الكود: {e}")
        st.stop()

    cg, cr, cp = st.columns([1,2,2])
    with cg:
        render_grade(rep0.grade, rep0.overall_score)
        hi = sum(1 for p in rep0.problems if p['severity']=='high')
        me = sum(1 for p in rep0.problems if p['severity']=='medium')
        lo = sum(1 for p in rep0.problems if p['severity']=='low')
        st.markdown(f"""<div style="text-align:center;color:#aaa;font-size:.85rem;margin-top:.3rem;">
          <div>📂 <b style="color:#bb86fc">{rep0.problem_category}</b></div>
          <div style="margin-top:.3rem;">🔴 {hi} &nbsp; 🟡 {me} &nbsp; 🔵 {lo}</div>
        </div>""", unsafe_allow_html=True)
    with cr:
        st.plotly_chart(radar_chart(rep0, "الكود الأصلي"), use_container_width=True)
    with cp:
        st.markdown("**المشاكل:**")
        render_problems(rep0.problems)

    render_score_cards(rep0)

    with st.expander("🧬 Feature Vector الكامل"):
        feature_table(fv0)

    if analyze_only:
        st.info("ℹ️ وضع التحليل فقط.")
        st.stop()

    # ════════════════════════════════════════
    # 2. Iterative Refinement
    # ════════════════════════════════════════
    st.markdown('<div class="sec-title">🔁 الإصلاح التكراري</div>', unsafe_allow_html=True)
    prog    = st.progress(0, text="جاري التهيئة …")
    status  = st.empty()
    log_box = st.container()

    refiner      = IterativeRefiner(max_iterations=max_iter, target_score=float(target),
                                     min_improvement=min_imp, patience=patience,
                                     api_key=groq_key or None)
    current_code = user_code
    no_improve   = 0
    prev_score   = -math.inf
    history_ui   = []

    for i in range(1, max_iter + 1):
        prog.progress(i / max_iter, text=f"التكرار {i}/{max_iter}")
        status.info(f"🔄 تحليل التكرار {i} …")

        try:
            fv_i = ASTAnalyzer(current_code).build_feature_vector()
        except SyntaxError:
            status.error("❌ خطأ بنيوي في الكود الناتج — إيقاف.")
            break

        cat_i, probs_i = ProblemClassifier().classify(fv_i)
        rep_i = QualityScorer().score(fv_i, probs_i)
        rep_i.problem_category = cat_i
        delta = rep_i.overall_score - prev_score if prev_score > -math.inf else 0.0

        history_ui.append({"iteration":i,"score":rep_i.overall_score,
                            "grade":rep_i.grade,"category":rep_i.problem_category,
                            "problems":len(probs_i),"delta":delta})
        refiner.history.append(IterationRecord(
            iteration=i,code=current_code,feature_vector=fv_i,
            quality_report=rep_i,score_delta=delta))

        sign  = "+" if delta>0 else ""
        dclr  = "#03dac6" if delta>=0 else "#ef5350"
        with log_box:
            st.markdown(f"""
            <div class="it-row">
              <span class="it-num">تكرار {i}</span>
              <span class="it-score">{rep_i.overall_score:.1f}</span>
              <span class="it-grade" style="color:{gc(rep_i.grade)}">{rep_i.grade}</span>
              <span class="it-cat">{rep_i.problem_category} · {len(probs_i)} مشكلة</span>
              <span class="it-delta" style="color:{dclr}">{sign}{delta:.1f}</span>
            </div>""", unsafe_allow_html=True)

        if rep_i.overall_score >= target and i > 1:
            status.success(f"✅ بلغنا الهدف {target}!")
            break
        if rep_i.problem_category == "clean_code" and i > 1:
            status.success("✅ الكود أصبح نظيفاً!")
            break
        if i > 1 and delta < min_imp:
            no_improve += 1
            if no_improve >= patience:
                status.warning("⚠️ لا تحسن كافٍ — إيقاف.")
                break
        else:
            no_improve = 0
        if i == max_iter:
            status.info("ℹ️ وصلنا الحد الأقصى.")
            break

        status.info(f"🤖 إرسال الكود لـ Groq (تكرار {i}) …")
        prompt   = refiner._prompt_builder.build(current_code, rep_i, fv_i, i)
        improved, api_error = refiner.call_api_with_error(prompt)
        if api_error:
            status.error(f"âŒ Groq API error: {api_error}")
            break
        if not improved or improved.strip() == current_code.strip():
            status.warning("⚠️ لا تغييرات من API — إيقاف.")
            break

        current_code = improved
        prev_score   = rep_i.overall_score

    prog.progress(1.0, text="✅ اكتمل!")

    # Final analysis
    try:
        fv_f = ASTAnalyzer(current_code).build_feature_vector()
        cat_f, pf = ProblemClassifier().classify(fv_f)
        rep_f = QualityScorer().score(fv_f, pf)
        rep_f.problem_category = cat_f
    except SyntaxError:
        rep_f = rep_i
        fv_f  = fv_i

    # ════════════════════════════════════════
    # 3. Improvement Chart
    # ════════════════════════════════════════
    st.markdown('<div class="sec-title">📈 مسار التحسن عبر التكرارات</div>', unsafe_allow_html=True)
    if len(history_ui) > 1:
        st.plotly_chart(progress_chart(history_ui), use_container_width=True)
    else:
        st.info("تكرار واحد فقط — لا رسم.")

    # ════════════════════════════════════════
    # 4. Comparison Before/After
    # ════════════════════════════════════════
    st.markdown('<div class="sec-title">⚖️ المقارنة: قبل وبعد</div>', unsafe_allow_html=True)

    ca, cb = st.columns(2)
    with ca:
        st.markdown('<div class="side-header side-before">📌 قبل الإصلاح</div>', unsafe_allow_html=True)
        render_grade(rep0.grade, rep0.overall_score)
    with cb:
        st.markdown('<div class="side-header side-after">✨ بعد الإصلاح</div>', unsafe_allow_html=True)
        render_grade(rep_f.grade, rep_f.overall_score)

    improvement = rep_f.overall_score - rep0.overall_score
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("تحسن الدرجة", f"{rep_f.overall_score:.1f}",
              f"{'+' if improvement>=0 else ''}{improvement:.1f}")
    m2.metric("مشاكل قبل", len(rep0.problems))
    m3.metric("مشاكل بعد", len(rep_f.problems), len(rep_f.problems)-len(rep0.problems))
    m4.metric("عدد التكرارات", len(history_ui))

    st.plotly_chart(bar_compare(rep0, rep_f), use_container_width=True)

    r1, r2 = st.columns(2)
    with r1:
        st.plotly_chart(radar_chart(rep0,"قبل"), use_container_width=True)
    with r2:
        st.plotly_chart(radar_chart(rep_f,"بعد"), use_container_width=True)

    # ════════════════════════════════════════
    # 5. Diff
    # ════════════════════════════════════════
    st.markdown('<div class="sec-title">🔀 الفرق بين النسختين (Diff)</div>', unsafe_allow_html=True)
    diff_view(user_code, current_code)

    # ════════════════════════════════════════
    # 6. Code Side by Side
    # ════════════════════════════════════════
    st.markdown('<div class="sec-title">📄 الكود: قبل وبعد</div>', unsafe_allow_html=True)
    cc1, cc2 = st.columns(2)
    with cc1:
        st.markdown('<div class="side-header side-before">📌 الكود الأصلي</div>', unsafe_allow_html=True)
        st.code(user_code, language="python")
    with cc2:
        st.markdown('<div class="side-header side-after">✨ الكود المحسَّن</div>', unsafe_allow_html=True)
        st.code(current_code, language="python")

    # ════════════════════════════════════════
    # 7. Full Report (without JSON)
    # ════════════════════════════════════════
    st.markdown('<div class="sec-title">📋 التقرير الكامل</div>', unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["🔍 المشاكل المتبقية","🧬 Feature Vector","🔁 سجل التكرارات"])

    with tab1:
        render_problems(rep_f.problems)

    with tab2:
        fc1, fc2 = st.columns(2)
        with fc1:
            st.markdown("**قبل:**")
            feature_table(fv0)
        with fc2:
            st.markdown("**بعد:**")
            feature_table(fv_f)

    with tab3:
        for r in history_ui:
            sign = "+" if r['delta']>0 else ""
            dclr = "#03dac6" if r['delta']>=0 else "#ef5350"
            st.markdown(f"""
            <div class="it-row">
              <span class="it-num">تكرار {r['iteration']}</span>
              <span class="it-score">{r['score']:.1f}</span>
              <span class="it-grade" style="color:{gc(r['grade'])}">{r['grade']}</span>
              <span class="it-cat">{r['category']} · {r['problems']} مشكلة</span>
              <span class="it-delta" style="color:{dclr}">{sign}{r['delta']:.1f}</span>
            </div>""", unsafe_allow_html=True)

    # ════════════════════════════════════════
    # 8. Download only the improved code (no JSON)
    # ════════════════════════════════════════
    st.divider()
    st.download_button("⬇️ تحميل الكود المحسَّن (.py)",
                       data=current_code, file_name="refined_code.py",
                       mime="text/plain", use_container_width=True)

# ─────────────────────────────────────────────────────────────────────
# Footer (Team Name)
# ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  <p>👨‍💻 فريق العمل: [حسين النصار] | [منار شطناوي] | [محمد الجراح]</p>
  <p>© 2026 - جميع الحقوق محفوظة</p>
</div>
""", unsafe_allow_html=True)
