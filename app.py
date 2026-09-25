import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# --- 1. إعداد الصفحة لتكون عريضة واحترافية ---
st.set_page_config(layout="wide", page_title="نظام توقع تايتانيك", page_icon="🚢")

# --- 2. تحميل البيانات وتدريب الموديل (مع التخزين المؤقت لسرعة الأداء) ---
@st.cache_data
def load_data():
    # ملاحظة: تأكد أن اسم الملف عندك في الفولدر هو نفسه المكتوب هنا
    return pd.read_csv('Titanic-Dataset.csv')

@st.cache_resource
def train_model():
    data = load_data()
    X = data.drop('Survived', axis=1)
    y = data['Survived']
    
    numeric_features = ['Age', 'SibSp', 'Parch', 'Fare']
    categorical_features = ['Pclass', 'Sex', 'Embarked']
    
    numeric_transformer = SimpleImputer(strategy='median')
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('classifier', RandomForestClassifier(random_state=42))])
    model.fit(X, y)
    return model, data

model, data = train_model()

# --- 3. إعدادات القائمة الجانبية (Sidebar) لإدخال البيانات ---
with st.sidebar:
    st.header("📝 لوحة إدخال البيانات")
    st.write("قم بتعبئة بيانات الراكب للحصول على التوقع")
    st.divider()
    
    pclass = st.selectbox("درجة التذكرة (Pclass)", [1, 2, 3])
    sex = st.selectbox("الجنس (Sex)", ["male", "female"])
    age = st.number_input("العمر (Age)", min_value=0.0, max_value=100.0, value=25.0)
    sibsp = st.number_input("عدد الأزواج/الأخوات (SibSp)", min_value=0, value=0)
    parch = st.number_input("عدد الآباء/الأبناء (Parch)", min_value=0, value=0)
    fare = st.number_input("سعر التذكرة (Fare)", min_value=0.0, value=10.0)
    embarked = st.selectbox("مكان الركوب (Embarked)", ["S", "C", "Q"])
    
    st.divider()
    predict_btn = st.button("🔮 توقع النتيجة", use_container_width=True, type="primary")

# --- 4. الصفحة الرئيسية ---
st.title("🚢 نظام تحليل وتوقع نجاة ركاب التايتانيك")
st.markdown("---")

if predict_btn:
    # === قسم النتيجة والتوقع ===
    st.subheader("🎯 نتائج التوقع للراكب الحالي")
    
    # تجهيز بيانات الراكب
    new_passenger = pd.DataFrame({
        'Pclass': [pclass], 'Sex': [sex], 'Age': [age],
        'SibSp': [sibsp], 'Parch': [parch], 'Fare': [fare], 'Embarked': [embarked]
    })
    
    # التوقع
    prediction = model.predict(new_passenger)
    probability = model.predict_proba(new_passenger)
    
    # عرض النتيجة بشكل احترافي في 3 أعمدة
    col_res1, col_res2, col_res3 = st.columns([1, 1, 1])
    
    with col_res1:
        st.markdown("**القرار النهائي:**")
        if prediction[0] == 1:
            st.success("✅ الراكب نجا!")
            st.balloons()
        else:
            st.error("❌ الراكب لم ينجُ")
            
    with col_res2:
        st.markdown("**نسبة النجاة:**")
        st.metric(label="Survival Probability", value=f"{probability[0][1]*100:.2f}%")
        
    with col_res3:
        st.markdown("**نسبة الوفاة:**")
        st.metric(label="Death Probability", value=f"{probability[0][0]*100:.2f}%")

    st.divider()

    # === قسم المخططات البيانية ===
    st.subheader("📊 لوحة التحليلات والمخططات")
    
    # الصف الأول: المخطط الدائري للراكب + توزيع النجاة العام
    row1_col1, row1_col2 = st.columns(2)
    
    with row1_col1:
        st.markdown("**احتمالات النجاة لهذا الراكب**")
        fig_pie, ax_pie = plt.subplots(figsize=(6, 4))
        labels = ['Did Not Survive', 'Survived']
        sizes = [probability[0][0], probability[0][1]]
        colors = ['#ff9999', '#66b3ff']
        ax_pie.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax_pie.axis('equal')
        st.pyplot(fig_pie)

    with row1_col2:
        st.markdown("**توزيع النجاة في البيانات العامة**")
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        data['Survived'].value_counts().plot(kind='bar', color=['#66b3ff', '#ff9999'], ax=ax1)
        ax1.set_xticklabels(['Did Not Survive', 'Survived'], rotation=0)
        ax1.set_ylabel("Number of Passengers")
        st.pyplot(fig1)

    st.divider()

    # الصف الثاني: النجاة حسب الجنس + العمر والأجرة
    row2_col1, row2_col2 = st.columns(2)
    
    with row2_col1:
        st.markdown("**النجاة حسب الجنس (Survival by Gender)**")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        pd.crosstab(data['Sex'], data['Survived']).plot(kind='bar', color=['#66b3ff', '#ffcc99'], ax=ax2)
        ax2.set_xticklabels(['Female', 'Male'], rotation=0)
        ax2.set_ylabel("Number of Passengers")
        ax2.legend(['Did Not Survive', 'Survived'])
        st.pyplot(fig2)

    with row2_col2:
        st.markdown("**العمر والأجرة مقابل النجاة (Age & Fare vs Survival)**")
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        colors_scatter = {0: '#ff9999', 1: '#66b3ff'}
        ax3.scatter(data[data['Survived']==0]['Age'], data[data['Survived']==0]['Fare'], 
                    c=colors_scatter[0], label='Did Not Survive', alpha=0.6)
        ax3.scatter(data[data['Survived']==1]['Age'], data[data['Survived']==1]['Fare'], 
                    c=colors_scatter[1], label='Survived', alpha=0.6)
        ax3.set_xlabel("Age")
        ax3.set_ylabel("Fare")
        ax3.legend()
        st.pyplot(fig3)

else:
    # رسالة ترحيبية تظهر قبل الضغط على الزر
    st.info("👈 يرجى إدخال بيانات الراكب من القائمة الجانبية، ثم الضغط على زر **'توقع النتيجة'** لعرض النتيجة وكافة المخططات التحليلية هنا.")
