import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns

# Функция для логистической регрессии
def perform_logistic_regression(df, target_column):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Логистическая регрессия
    model = LogisticRegression()
    model.fit(X, y)
    
    # Возвращаем веса как словарь
    coef_dict = dict(zip(X.columns, model.coef_[0]))
    intercept = model.intercept_[0]
    return coef_dict, intercept

# Интерфейс для загрузки файла
st.title("Логистическая регрессия (используя CSV-данные)")

# Шаг 1: Загрузка CSV файла
uploaded_file = st.file_uploader("Загрузите CSV файл", type="csv")

if uploaded_file:
    # Чтение данных
    df = pd.read_csv(uploaded_file)
    st.write("Данные загружены:")
    st.write(df.head())

    # Выбор целевой переменной
    target_column = st.selectbox("Выберите целевую переменную (таргет)", df.columns)

    if target_column:
        # Выполнение логистической регрессии
        coef_dict, intercept = perform_logistic_regression(df, target_column)

        # Отображение результатов регрессии
        st.subheader("Результаты логистической регрессии:")
        st.write("Свободный член (intercept):", intercept)
        st.write("Коэффициенты для каждого признака:")
        st.write(coef_dict)

        # Шаг 2: Построение scatter plot
        st.subheader("Построение scatter plot")
        
        # Выбор двух признаков для scatter plot
        feature_x = st.selectbox("Выберите признак по оси X", df.columns)
        feature_y = st.selectbox("Выберите признак по оси Y", df.columns)

        if feature_x and feature_y:
            # Построение scatter plot с цветовой маркировкой по целевой переменной
            fig, ax = plt.subplots()
            sns.scatterplot(x=df[feature_x], y=df[feature_y], hue=df[target_column], ax=ax, palette="coolwarm")
            ax.set_title(f"Скатерплот: {feature_x} vs {feature_y}")
            st.pyplot(fig)