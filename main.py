import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# URL del archivo CSV en GitHub
DATA_URL = "https://raw.githubusercontent.com/JulianTorrest/Educacion/refs/heads/main/kc_house_data.csv"

# --- Parte 0: Carga de Datos ---
# Decorador para cachear los datos. Esto evita que la función se ejecute cada vez que Streamlit se refresca.
# Es ideal para datasets que no cambian a menudo y acelera la aplicación.
@st.cache_data
def load_data(url):
    """
    Carga los datos desde la URL y maneja posibles errores.
    """
    st.info("Cargando datos desde GitHub... Esto puede tardar un momento.")
    try:
        df = pd.read_csv(url)
        st.success("¡Datos cargados exitosamente!")
        return df
    except Exception as e:
        st.error(f"¡Error al cargar los datos! Por favor, verifica la URL o tu conexión a internet: {e}")
        return pd.DataFrame() # Retorna un DataFrame vacío en caso de error

# --- Parte 1: Preparación de Datos ---
def prepare_data(df):
    """
    Realiza las transformaciones necesarias en los datos para su análisis y modelado.
    Cada paso se explica con comentarios detallados.
    """
    st.header("1. Preparación de Datos")
    st.info("Procesando los datos: limpieza, tipado correcto y manejo de valores atípicos.")

    # 1.1 Poner los datos en sus tipos de datos correctos
    st.subheader("1.1 Conversión de tipos de datos")
    st.write("Aseguramos que cada columna tenga el tipo de dato apropiado para un mejor manejo y análisis.")
    df['date'] = pd.to_datetime(df['date']) # Convertir la columna 'date' a formato de fecha y hora.
    df['zipcode'] = df['zipcode'].astype(str) # 'zipcode' es categórico, no numérico para operaciones aritméticas.
    
    # Intentar downcast de columnas numéricas para optimizar el uso de memoria si los valores lo permiten.
    # Por ejemplo, un 'int64' puede convertirse en 'int32' o 'int16' si los números son pequeños.
    # IMPORTANTE: Asegúrate de que 'errors=coerce' se use para convertir valores no convertibles a NaN.
    for col in ['bedrooms', 'bathrooms', 'floors', 'waterfront', 'view', 'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'sqft_living15', 'sqft_lot15']:
        if col in df.columns:
            # Coercer errores a NaN, que luego serán manejados por el paso de imputación
            df[col] = pd.to_numeric(df[col], errors='coerce', downcast='integer')
    st.write("Tipos de datos después de la conversión:")
    st.dataframe(df.dtypes.apply(lambda x: str(x)).reset_index().rename(columns={'index': 'Columna', 0: 'Tipo de Dato'}))
    st.write("---")

    # 1.2 Remover los valores duplicados
    st.subheader("1.2 Remoción de valores duplicados")
    initial_rows = df.shape[0] # Contar las filas antes de la eliminación.
    df.drop_duplicates(inplace=True) # Eliminar filas que son exactamente iguales.
    duplicate_rows = initial_rows - df.shape[0] # Calcular cuántas filas fueron eliminadas.
    st.write(f"Filas iniciales: {initial_rows}")
    st.write(f"Filas después de remover duplicados: {df.shape[0]}")
    st.write(f"Número de filas duplicadas removidas: {duplicate_rows}")
    st.write("La eliminación de duplicados asegura que cada registro sea único, evitando sesgos en el análisis.")
    st.write("---")

    # 1.3 Procesar y reemplazar los datos que faltan (Missing N.A values)
    st.subheader("1.3 Manejo de valores faltantes (Missing N.A values)")
    missing_data = df.isnull().sum() # Contar los valores nulos por columna.
    missing_data = missing_data[missing_data > 0] # Filtrar solo las columnas con nulos.
    
    if not missing_data.empty:
        st.write("Valores faltantes detectados por columna ANTES de la imputación:")
        st.dataframe(missing_data.to_frame(name='Valores Faltantes'))
        
        # Imputación: rellenar valores nulos.
        # Para columnas numéricas, la mediana es robusta a outliers.
        numeric_cols_to_impute = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement', 'sqft_living15', 'sqft_lot15', 'waterfront', 'view', 'condition', 'grade', 'floors', 'yr_built']
        for col in numeric_cols_to_impute:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]) and df[col].isnull().any():
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                st.write(f"- Columna '{col}': Valores nulos imputados con la mediana ({median_val}).")
            elif col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                st.warning(f"La columna '{col}' no es numérica, omitiendo la imputación con mediana.")
        
        # 'yr_renovated': Si es 0 significa que no fue renovado. Si es NaN, imputamos con 0.
        if 'yr_renovated' in df.columns and df['yr_renovated'].isnull().any():
            df['yr_renovated'].fillna(0, inplace=True)
            st.write("- Columna 'yr_renovated': Valores nulos imputados con 0 (asumiendo que no fue renovado).")
            
        st.write("El manejo de valores faltantes es crucial para evitar errores en análisis y modelos posteriores.")
        # Verificación final de nulos después de la imputación en prepare_data
        final_missing_check = df.isnull().sum()
        final_missing_check = final_missing_check[final_missing_check > 0]
        if not final_missing_check.empty:
            st.warning("Advertencia: ¡Todavía hay valores faltantes después de la imputación! Detalles:")
            st.dataframe(final_missing_check.to_frame(name='Nulos Restantes'))
        else:
            st.success("Todos los valores faltantes han sido tratados.")

    else:
        st.write("No se detectaron valores faltantes en el dataset.")
    st.write("---")

    # 1.4 Manejo de Outliers (Valores Atípicos) - Usaremos el método IQR para algunas columnas numéricas clave
    st.subheader("1.4 Manejo de Outliers (Método IQR)")
    st.write("Los outliers pueden distorsionar los resultados del análisis y el rendimiento de los modelos. Los limitamos usando el método del Rango Intercuartílico (IQR).")
    outlier_cols = ['price', 'sqft_living', 'sqft_lot', 'bedrooms', 'bathrooms', 'sqft_above', 'sqft_basement']
    
    for col in outlier_cols:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Contar y aplicar capping (limitar valores a los límites IQR)
            outliers_before = df[(df[col] < lower_bound) | (df[col] > upper_bound)].shape[0]
            if outliers_before > 0:
                df[col] = np.clip(df[col], lower_bound, upper_bound)
                st.write(f"- Columna '{col}': Se detectaron y trataron {outliers_before} outliers (con capping).")
            else:
                st.write(f"- Columna '{col}': No se detectaron outliers significativos con el método IQR.")
    st.write("---")

    # 1.5 Re-escalar las variables / Normalizar (Para el modelo de ML, si es necesario)
    # Aunque no se visualiza directamente, es un paso fundamental para muchos modelos de ML.
    # Usaremos StandardScaler en la sección de ML para las características.

    # 1.6 Data Binning (Ejemplo: crear categorías de precios)
    st.subheader("1.5 Data Binning (Creación de categorías de precio)")
    st.write("Creamos una nueva columna categórica 'price_category' para agrupar las propiedades por rangos de precio, útil para análisis por segmentos.")
    if 'price' in df.columns:
        bins = [0, 200000, 400000, 600000, 800000, 1000000, np.inf]
        labels = ['Bajo', 'Medio-Bajo', 'Medio', 'Medio-Alto', 'Alto', 'Muy Alto']
        df['price_category'] = pd.cut(df['price'], bins=bins, labels=labels, right=False)
        st.write("Se ha creado la columna 'price_category' basada en el precio.")
        st.dataframe(df['price_category'].value_counts().to_frame().rename(columns={'count': 'Número de Propiedades'})) # Fix for Streamlit 1.35.0+
    st.write("---")

    # 1.7 Conversión Numérico a categórico / Categórico a Numérico
    # Ya se realizó en pasos anteriores (price_category, zipcode).
    # Para el modelo de ML, las columnas 'waterfront', 'view', 'condition', 'grade' ya son numéricas
    # y pueden ser tratadas como ordinales o categóricas directamente por el modelo lineal.

    return df

# --- Parte 2: Exploración Estadística y Visualización de Datos (EDA) ---
def perform_eda(df):
    """
    Realiza la exploración estadística y visualización de los datos procesados.
    """
    st.header("2. Exploración Estadística y Visualización de Datos (EDA)")
    st.info("Exploramos las distribuciones, relaciones y patrones en los datos a través de estadísticas y gráficos.")

    st.subheader("2.1 Estadísticas Descriptivas")
    st.write("Un resumen rápido de las principales estadísticas para las columnas numéricas, incluyendo media, desviación estándar, y cuartiles.")
    st.dataframe(df.describe())
    st.write("---")

    # Histogramas para variables numéricas clave
    st.subheader("2.2 Histogramas")
    st.write("Los histogramas muestran la distribución de una única variable numérica, ayudando a identificar sesgos o modos.")
    numeric_cols = ['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'grade', 'yr_built']
    for col in numeric_cols:
        if col in df.columns:
            fig = px.histogram(df, x=col, title=f'Distribución de {col}', marginal="box") # marginal="box" añade un box plot
            st.plotly_chart(fig)
    st.write("---")

    # Diagrámas de dispersión (Scatter Plots)
    st.subheader("2.3 Diagramas de Dispersión")
    st.write("Los diagramas de dispersión revelan la relación entre dos variables numéricas, permitiendo identificar correlaciones o clústeres.")
    
    st.write("#### Relación entre `sqft_living` y `price`:")
    fig_scatter_price_sqft = px.scatter(df, x='sqft_living', y='price',
                                        title='Precio vs. Pies Cuadrados Habitables',
                                        hover_data=['bedrooms', 'bathrooms', 'grade', 'price_category'],
                                        color='price_category' if 'price_category' in df.columns else None)
    st.plotly_chart(fig_scatter_price_sqft)

    st.write("#### Relación entre `grade` y `price`:")
    fig_scatter_grade_price = px.scatter(df, x='grade', y='price',
                                         title='Precio vs. Calidad de Construcción (Grade)',
                                         hover_data=['sqft_living', 'yr_built'])
    st.plotly_chart(fig_scatter_grade_price)
    
    st.write("#### Relación entre `yr_built` y `price`:")
    fig_scatter_yr_built_price = px.scatter(df, x='yr_built', y='price',
                                            title='Precio vs. Año de Construcción',
                                            hover_data=['sqft_living', 'grade'],
                                            color='grade', animation_frame='yr_built', animation_group='yr_built')
    st.plotly_chart(fig_scatter_yr_built_price)
    st.write("---")

    # Cajas y Bigotes (Box Plots)
    st.subheader("2.4 Diagramas de Cajas y Bigotes")
    st.write("Los diagramas de caja y bigotes muestran la distribución de una variable numérica para diferentes categorías, útiles para identificar diferencias de grupo y outliers.")
    
    st.write("#### Distribución del precio por número de dormitorios:")
    fig_box_bedrooms = px.box(df, x='bedrooms', y='price',
                              title='Distribución del Precio por Número de Dormitorios',
                              points="outliers") # Muestra los outliers
    st.plotly_chart(fig_box_bedrooms)

    st.write("#### Distribución del precio por número de baños:")
    fig_box_bathrooms = px.box(df, x='bathrooms', y='price',
                               title='Distribución del Precio por Número de Baños',
                               points="outliers")
    st.plotly_chart(fig_box_bathrooms)
    
    st.write("#### Distribución del precio por condición:")
    fig_box_condition = px.box(df, x='condition', y='price',
                                title='Distribución del Precio por Condición',
                                points="outliers")
    st.plotly_chart(fig_box_condition)
    st.write("---")

    # Mapa de calor de correlación
    st.subheader("2.5 Mapa de Calor de Correlación")
    st.write("Un mapa de calor visualiza la matriz de correlación entre variables numéricas, revelando qué tan fuertemente se relacionan entre sí (positiva o negativamente).")
    numeric_df = df.select_dtypes(include=np.number)
    
    # Excluir 'id' y 'zipcode' si son numéricos pero no relevantes para la correlación directa del precio
    if 'id' in numeric_df.columns:
        numeric_df = numeric_df.drop(columns=['id'])
    if 'zipcode' in numeric_df.columns and pd.api.types.is_numeric_dtype(numeric_df['zipcode']):
        numeric_df = numeric_df.drop(columns=['zipcode'])
    
    corr_matrix = numeric_df.corr()
    
    # Eliminar correlaciones con NaN (si alguna columna tiene varianza cero después del preprocesamiento)
    corr_matrix = corr_matrix.dropna(axis=0, how='all').dropna(axis=1, how='all')
    
    if not corr_matrix.empty and len(corr_matrix.columns) > 1: # Asegurarse de que haya al menos dos columnas para correlación
        plt.figure(figsize=(14, 12)) # Aumentar el tamaño del gráfico
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
        plt.title('Mapa de Calor de Correlación entre Variables Numéricas', fontsize=16)
        st.pyplot(plt)
        plt.clf() # Limpiar la figura de matplotlib para evitar superposiciones en Streamlit
    else:
        st.write("No hay suficientes variables numéricas para generar un mapa de calor de correlación.")
    st.write("---")

    # Visualización de la categoría de precio
    if 'price_category' in df.columns:
        st.subheader("2.6 Conteo de Propiedades por Categoría de Precio")
        st.write("Este gráfico de barras muestra la distribución de las propiedades a lo largo de las categorías de precio definidas.")
        price_category_counts = df['price_category'].value_counts().reset_index()
        price_category_counts.columns = ['Price Category', 'Count'] # Renombrar columnas para Plotly Express
        
        fig_price_cat = px.bar(price_category_counts,
                               x='Price Category', y='Count',
                               labels={'Price Category': 'Categoría de Precio', 'Count': 'Número de Propiedades'},
                               title='Conteo de Propiedades por Categoría de Precio',
                               color='Price Category', # Colorear por categoría
                               category_orders={"Price Category": ['Bajo', 'Medio-Bajo', 'Medio', 'Medio-Alto', 'Alto', 'Muy Alto']}) # Ordenar categorías
        st.plotly_chart(fig_price_cat)
    st.write("---")

# --- Parte 3: Modelado de Machine Learning ---
def machine_learning_model(df):
    """
    Entrena y evalúa un modelo de regresión lineal para predecir el precio de las viviendas.
    """
    st.header("3. Modelo de Machine Learning: Predicción de Precios de Viviendas")
    st.info("Construimos un modelo de Regresión Lineal para predecir el precio de las propiedades usando las características disponibles.")

    # 3.1 Selección de Características (Features) y Variable Objetivo
    st.subheader("3.1 Selección de Características y Variable Objetivo")
    st.write("Identificamos las variables de entrada (características) que usaremos para predecir la variable de salida (precio).")
    
    # Excluir 'id', 'date', 'zipcode' (ya es str), 'price_category' (creada para EDA), 'price' (variable objetivo)
    features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 
                'waterfront', 'view', 'condition', 'grade', 'sqft_above', 
                'sqft_basement', 'yr_built', 'yr_renovated', 'lat', 'long', 
                'sqft_living15', 'sqft_lot15']
    
    # Asegurarse de que todas las características existan en el DataFrame y sean numéricas
    selected_features = [f for f in features if f in df.columns and pd.api.types.is_numeric_dtype(df[f])]
    
    if not selected_features:
        st.warning("No hay suficientes características numéricas válidas para entrenar el modelo. Por favor, revisa la preparación de datos.")
        return

    X = df[selected_features] # Características
    y = df['price']           # Variable objetivo (lo que queremos predecir)

    st.write(f"Características seleccionadas para el modelo: {', '.join(selected_features)}")
    st.write(f"Variable objetivo: 'price'")
    st.write("---")

    # --- Verificación Final de Valores Finitos (CRUCIAL para Scikit-learn) ---
    st.subheader("3.1.1 Verificación Final de Valores Finitos")
    initial_ml_rows = X.shape[0]
    # Reemplazar infinitos con NaN antes de eliminar los NaN para un manejo consistente
    combined_data = pd.concat([X, y], axis=1)
    combined_data.replace([np.inf, -np.inf], np.nan, inplace=True) 
    
    # Eliminar filas donde haya algún valor NaN en las características o en la variable objetivo
    cleaned_combined_data = combined_data.dropna() 

    if cleaned_combined_data.shape[0] < initial_ml_rows:
        rows_dropped = initial_ml_rows - cleaned_combined_data.shape[0]
        st.warning(f"¡Atención! Se detectaron y eliminaron {rows_dropped} filas con valores NaN/Infinitos en las características o la variable objetivo antes del entrenamiento del modelo.")
    else:
        st.write("No se detectaron valores NaN/Infinitos en las características o la variable objetivo. ¡Datos listos para el modelado!")

    X = cleaned_combined_data[selected_features]
    y = cleaned_combined_data['price']

    if X.empty or y.empty:
        st.error("Después de limpiar valores NaN/Infinitos, el conjunto de datos para el modelo está vacío o es demasiado pequeño. No se puede entrenar el modelo.")
        return
    # --- FIN Verificación Final de Valores Finitos ---


    # 3.2 División del Dataset (Entrenamiento y Prueba)
    st.subheader("3.2 División del Dataset (Entrenamiento y Prueba)")
    st.write("Dividimos los datos en conjuntos de entrenamiento (70%) y prueba (30%) para evaluar la capacidad del modelo de generalizar en datos no vistos.")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    st.write(f"Tamaño del conjunto de entrenamiento: {len(X_train)} muestras")
    st.write(f"Tamaño del conjunto de prueba: {len(X_test)} muestras")
    st.write("---")

    # 3.3 Escalado de Características (Feature Scaling)
    st.subheader("3.3 Escalado de Características (StandardScaler)")
    st.write("El escalado es importante para modelos basados en gradientes y distancias. StandardScaler normaliza las características para que tengan media 0 y desviación estándar 1.")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train) # Ajustar y transformar en el conjunto de entrenamiento
    X_test_scaled = scaler.transform(X_test)       # Transformar solo en el conjunto de prueba

    st.write("Características escaladas exitosamente.")
    st.write("---")

    # 3.4 Entrenamiento del Modelo de Regresión Lineal
    st.subheader("3.4 Entrenamiento del Modelo de Regresión Lineal")
    st.write("Entrenamos un modelo de Regresión Lineal, que busca la mejor línea (o hiperplano) para ajustar los datos y predecir el precio.")
    model = LinearRegression()
    model.fit(X_train_scaled, y_train) # Entrenar el modelo con los datos escalados
    st.write("Modelo de Regresión Lineal entrenado.")
    st.write("---")

    # 3.5 Evaluación del Modelo
    st.subheader("3.5 Evaluación del Modelo")
    st.write("Evaluamos el rendimiento del modelo en el conjunto de prueba utilizando métricas clave.")
    y_pred = model.predict(X_test_scaled) # Realizar predicciones en el conjunto de prueba

    mse = mean_squared_error(y_test, y_pred) # Error Cuadrático Medio
    rmse = np.sqrt(mse)                     # Raíz del Error Cuadrático Medio
    r2 = r2_score(y_test, y_pred)           # Coeficiente de Determinación R²

    st.write(f"**Métricas de Evaluación:**")
    st.write(f"- **Error Cuadrático Medio (MSE):** ${mse:,.2f}")
    st.write(f"- **Raíz del Error Cuadrático Medio (RMSE):** ${rmse:,.2f}")
    st.write(f"- **Coeficiente de Determinación (R²):** {r2:.4f}")

    st.write("---")
    st.subheader("3.6 Visualización de Predicciones vs. Valores Reales")
    st.write("Un gráfico de dispersión de los valores reales vs. los predichos nos permite ver visualmente qué tan bien se ajusta el modelo.")
    
    # Crear un DataFrame para la visualización de resultados
    results_df = pd.DataFrame({'Actual Price': y_test, 'Predicted Price': y_pred})
    
    fig_predictions = px.scatter(results_df, x='Actual Price', y='Predicted Price',
                                 title='Precios Reales vs. Precios Predichos',
                                 labels={'Actual Price': 'Precio Real', 'Predicted Price': 'Precio Predicho'},
                                 trendline="ols", # Añadir una línea de regresión (Ordinary Least Squares)
                                 trendline_color_override="red") # Color de la línea de tendencia
    st.plotly_chart(fig_predictions)

    st.write("Idealmente, los puntos deberían agruparse cerca de la línea de 45 grados, indicando predicciones precisas.")
    st.write("---")

# --- Parte 4: Conclusiones ---
def display_conclusions():
    st.header("4. Conclusiones")
    st.info("Resumen de los hallazgos clave de la preparación de datos y la exploración, y la información obtenida del modelado.")

    st.subheader("(I) Conclusiones sobre los Datos Procesados:")
    st.write("""
    - **Calidad de Datos Mejorada:** La conversión de tipos, la eliminación de duplicados y la imputación de valores nulos han limpiado el dataset, haciéndolo más robusto para el análisis. La verificación final de valores nulos e infinitos asegura que los datos estén listos para el modelado.
    - **Manejo de Outliers:** Al limitar los valores atípicos, se reduce la influencia de puntos extremos en las distribuciones y modelos, lo que puede llevar a resultados más estables y representativos.
    - **Datos Listos para Modelado:** El preprocesamiento, incluyendo el escalado (para el modelo), prepara las características en un formato óptimo para algoritmos de Machine Learning, mejorando su rendimiento y estabilidad.
    - **Categorización para Análisis:** La creación de la categoría de precio (`price_category`) facilita el análisis segmentado, permitiendo entender mejor cómo se distribuyen las propiedades por rangos de valor.
    """)

    st.subheader("(II) Conclusiones sobre la Información Observada:")
    st.write("""
    - **Factores Clave del Precio:** La exploración visual y el mapa de calor de correlación confirman que **`sqft_living`**, **`grade`** y el número de **`bathrooms`** y **`bedrooms`** son los atributos más fuertemente correlacionados positivamente con el precio.
    - **Distribución de Precios:** Los histogramas muestran que la mayoría de las propiedades tienen precios en los rangos medio-bajo a medio, con una cola derecha que indica la presencia de propiedades de alto valor.
    - **Impacto de la Calidad (Grade):** Las propiedades con un **`grade`** más alto (`calidad de construcción`) tienen consistentemente precios promedio mucho mayores, como se ve en los diagramas de dispersión y cajas.
    - **Antigüedad vs. Precio:** El año de construcción (**`yr_built`**) muestra una correlación interesante; las casas más antiguas pueden tener un valor histórico o haber sido renovadas, mientras que las más nuevas pueden tener precios más altos debido a modernidad o mejores acabados.
    - **Rendimiento del Modelo:** El modelo de Regresión Lineal, aunque básico, proporciona una base para la predicción de precios. El **R²** (coeficiente de determinación) indica la proporción de la varianza en la variable dependiente que es predecible a partir de las variables independientes. Un RMSE (Raíz del Error Cuadrático Medio) nos da una idea del error promedio de predicción en la misma unidad que la variable objetivo (dólares). Esto indica que, en promedio, nuestras predicciones se desvían del precio real en aproximadamente la cantidad del RMSE.
    - **Áreas de Mejora:** Observamos que la relación entre algunas características y el precio no es puramente lineal, lo que sugiere que modelos más complejos (como Random Forest o Gradient Boosting) o ingeniería de características adicional podrían mejorar significativamente la precisión de las predicciones.
    """)
    st.write("---")


# --- Diseño de la Aplicación Streamlit ---
def main():
    st.set_page_config(layout="wide", page_title="Análisis de Datos de Viviendas", page_icon="🏠")
    st.title("🏠 Análisis y Predicción de Precios de Viviendas en King County")
    st.markdown("Bienvenido a esta aplicación interactiva para explorar y modelar los precios de las casas en King County, EE. UU.")

    # Cargar los datos
    df_original = load_data(DATA_URL)

    if not df_original.empty:
        st.subheader("Vista Previa de los Datos Originales")
        st.write("Aquí están las primeras 5 filas de los datos tal como se cargaron, antes de cualquier procesamiento.")
        st.dataframe(df_original.head())
        st.write("---")

        # Sidebar para navegación
        st.sidebar.title("Navegación")
        section = st.sidebar.radio("Ir a la Sección:",
                                   ["Preparación de Datos", "Exploración de Datos (EDA)", "Modelo de Machine Learning", "Conclusiones"])

        # Procesar los datos una vez y pasarlos a las funciones correspondientes
        # Aseguramos que `df_processed` esté siempre disponible y actualizado para cada sección
        df_processed = prepare_data(df_original.copy()) # Creamos una copia para no modificar el DataFrame original


        if section == "Preparación de Datos":
            # Si se está en la sección de preparación, mostrar una vista previa de los datos procesados
            st.subheader("Vista Previa de los Datos Procesados")
            st.write("Aquí están las primeras 5 filas de los datos después de aplicar todas las transformaciones de limpieza y preprocesamiento.")
            st.dataframe(df_processed.head())
            st.write("---")
            # Opcional: mostrar un resumen de las columnas faltantes después del procesamiento
            st.subheader("Resumen de Nulos Después del Procesamiento")
            st.dataframe(df_processed.isnull().sum().to_frame(name='Nulos Después de Procesar'))

        elif section == "Exploración de Datos (EDA)":
            perform_eda(df_processed)

        elif section == "Modelo de Machine Learning":
            machine_learning_model(df_processed)

        elif section == "Conclusiones":
            display_conclusions()

    else:
        st.warning("No se pudo cargar el dataset. La aplicación no puede continuar.")

if __name__ == "__main__":
    main()
