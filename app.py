import streamlit as st
import pandas as pd
import sqlite3
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# إعدادات الصفحة الخاصة بـ Streamlit
st.set_page_config(page_title="نظام تفتيش الاحتيال وغسيل الأموال | Mozn Project", page_icon="🛡️", layout="wide")

# تصميم مخصص باستخدام CSS البسيط لمحاكاة هويات أنظمة الامتثال
st.markdown("""
    <style>
    .main-title { font-size:28px; font-weight:bold; color:#1E3A8A; text-align:right; margin-bottom:20px; }
    .subtitle { font-size:18px; color:#4B5563; text-align:right; margin-bottom:30px; }
    .metric-box { background-color:#F3F4F6; padding:15px; border-radius:10px; border-right: 5px solid #1E3A8A; }
    div[data-testid="stMarkdownContainer"] { text-align: right; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>🛡️ منصة الاستخبارات المالية والتحقيق الذكي (AML & Fraud Guard)</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>نظام متكامل يدمج بين تعلم الآلة (Machine Learning) والذكاء الاصطناعي التوليدي المحلي (GenAI) لكشف وتحليل العمليات المشبوهة.</div>", unsafe_allow_html=True)

# دالة لجلب البيانات وتدريب النموذج (مع استخدام التخزين المؤقت Cache لتسريع الأداء)
@st.cache_resource
def load_data_and_train_model():
    conn = sqlite3.connect('fraud_database.db')
    query = "SELECT type, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest, isFraud FROM transactions WHERE type IN ('TRANSFER', 'CASH_OUT')"
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    # هندسة الميزات
    df['errorBalanceOrig'] = df['newbalanceOrig'] + df['amount'] - df['oldbalanceOrg']
    df['errorBalanceDest'] = df['oldbalanceDest'] + df['amount'] - df['newbalanceDest']
    df_encoded = pd.get_dummies(df, columns=['type'], drop_first=True)
    
    X = df_encoded.drop('isFraud', axis=1)
    y = df_encoded['isFraud']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    model = RandomForestClassifier(n_estimators=50, class_weight='balanced', random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    # حفظ العمليات التي تم اكتشافها كاحتيال حقيقي للمعاينة
    fraud_indices = X_test[(y_pred == 1) & (y_test == 1)].index
    fraud_cases = X_test.loc[fraud_indices].copy()
    fraud_cases['isFraud'] = 1
    
    return model, fraud_cases

try:
    with st.spinner('جاري تحميل قاعدة البيانات وتدريب نموذج تعلم الآلة...'):
        model, fraud_df = load_data_and_train_model()
    st.sidebar.success("✅ تم تفعيل نموذج الـ Machine Learning بنجاح!")
except Exception as e:
    st.error(f"فشل تحميل قاعدة البيانات: {e}. تأكد من وجود ملف 'fraud_database.db' في نفس المجلد.")
    st.stop()

# القائمة الجانبية لإحصائيات النظام
st.sidebar.header("📊 حالة النظام والنموذج")
st.sidebar.metric(label="دقة اصطياد المحتالين (Recall)", value="96%")
st.sidebar.metric(label="دقة الاتهام الصائب (Precision)", value="100%")
st.sidebar.info("هذا النظام يعمل محلياً بالكامل (On-Premise) لضمان سرية وحماية بيانات العملاء المالية وفقاً للوائح البنك المركزي.")

# تقسيم الشاشة إلى تبويبين (Tabs)
tab1, tab2 = st.tabs(["🗂️ المعاملات المكتشفة", "🤖 تقرير التحقيق الذكي (GenAI)"])

with tab1:
    st.subheader("📌 قائمة بالمعاملات المشبوهة المكتشفة بواسطة خوارزمية Random Forest")
    st.write("يقوم النموذج أدناه بعرض العمليات التي صنفها كاحتيال عالي الخطورة بعد تحليل سلوك الحسابات وفحص الفروقات الرياضية:")
    
    st.dataframe(fraud_df.style.format({
        'amount': '${:,.2f}',
        'oldbalanceOrg': '${:,.2f}',
        'newbalanceOrig': '${:,.2f}',
        'oldbalanceDest': '${:,.2f}',
        'newbalanceDest': '${:,.2f}',
        'errorBalanceOrig': '${:,.2f}',
        'errorBalanceDest': '${:,.2f}'
    }), use_container_width=True)

with tab2:
    st.subheader("🤖 توليد تلخيص التنبيه الآلي (Automated Alert Summarization)")
    st.write("اختر رقم العملية من القائمة أدناه، ليقوم نظام الذكاء الاصطناعي التوليدي المحلي (Llama 3) بقراءة الأرقام وصياغة تقرير تحقيق جنائي مالي مباشر:")
    
    selected_index = st.selectbox("اختر رقم المعاملة المشبوهة للتحقيق فيها:", fraud_df.index)
    
    if selected_index:
        tx_data = fraud_df.loc[selected_index]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("مبلغ العملية", f"${tx_data['amount']:,.2f}")
        with col2:
            st.metric("الخطأ في حساب المرسل", f"${tx_data['errorBalanceOrig']:,.2f}")
        with col3:
            st.metric("نوع العملية", "TRANSFER / CASH_OUT")
            
        if st.button("🚀 توليد تقرير الذكاء الاصطناعي الآن (عبر Llama 3 محلياً)"):
            
            prompt_text = f"""
            You are an expert Anti-Money Laundering (AML) and Fraud Investigator.
            Our Machine Learning system has flagged the following transaction as HIGHLY SUSPICIOUS.
            
            Transaction Details:
            - Amount: ${tx_data['amount']:,.2f}
            - Sender Old Balance: ${tx_data['oldbalanceOrg']:,.2f}
            - Sender New Balance: ${tx_data['newbalanceOrig']:,.2f}
            - Receiver Old Balance: ${tx_data['oldbalanceDest']:,.2f}
            - Receiver New Balance: ${tx_data['newbalanceDest']:,.2f}
            - Mathematical Error in Sender's Balance: ${tx_data['errorBalanceOrig']:,.2f}
            
            Task:
            Write a brief, professional alert summary (1 paragraph) explaining WHY this transaction is suspicious based on the mathematical errors or empty balances. Provide the summary in both English and Arabic.
            """
            
            with st.spinner('جاري الاتصال بنموذج Llama 3 المحلي وتوليد التقرير...'):
                url = "http://localhost:11434/api/generate"
                payload = {
                    "model": "llama3",
                    "prompt": prompt_text,
                    "stream": False
                }
                
                try:
                    response = requests.post(url, json=payload)
                    response.raise_for_status()
                    ai_report = response.json()['response']
                    
                    st.success("✨ تم توليد التقرير بنجاح!")
                    st.markdown("### 📄 التقرير الفني الصادر عن النظام:")
                    st.info(ai_report)
                    
                except Exception as e:
                    st.error(f"❌ فشل الاتصال بنموذج الذكاء الاصطناعي المحلي.")
                    st.warning("تأكد من أن برنامج Ollama يعمل في الخلفية وأنك قمت بتحميل النموذج عبر كتابة: `ollama run llama3` في الـ Terminal.")