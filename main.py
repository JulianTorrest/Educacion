import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# URL del archivo CSV en GitHub
DATA_URL = "https://raw.githubusercontent.com/JulianTorrest/Educacion/refs/heads/main/kc_house_data.csv"

@st.cache_data
def load_data(url):
    """
    Carga los datos desde la URL y realiza la preparación inicial.
    """
    try:
        df = pd.read_csv(url)
        return df
    except Exception as e:
        st.error(f"Error al cargar los datos: {e}")
        return pd.DataFrame() # Retorna un DataFrame vacío en caso de error

def prepare_data(df):
    """
    Prepara los datos realizando las transformaciones necesarias.
    """
    st.subheader("1. Preparación de Datos")

    # 1. Poner los datos en sus tipos de datos correctos
    st.write("### 1.1 Conversión de tipos de datos")
    df['date'] = pd.to_datetime(df['date'])
    # Convertir 'zipcode' a tipo de dato object (categórico) si no se usará como número
    df['zipcode'] = df['zipcode'].astype(str)
    # Algunas columnas numéricas pueden ser optimizadas
    for col in ['bedrooms', 'bathrooms', 'floors', 'waterfront', 'view', 'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'sqft_living15', 'sqft_lot15']:
        if col in df.columns:
            # Intentar downcast para ahorrar memoria si es posible
            df[col] = pd.to_numeric(df[col], errors='coerce', downcast='integer')
    st.write("Tipos de datos después de la conversión:")
    st.dataframe(df.dtypes.apply(lambda x: str(x)).reset_index().rename(columns={'index': 'Columna', 0: 'Tipo de Dato'}))

    # 2. Remover los valores duplicados
    st.write("### 1.2 Remoción de valores duplicados")
    initial_rows = df.shape[0]
    df.drop_duplicates(inplace=True)
    duplicate_rows = initial_rows - df.shape[0]
    st.write(f"Filas iniciales: {initial_rows}")
    st.write(f"Filas después de remover duplicados: {df.shape[0]}")
    st.write(f"Número de filas duplicadas removidas: {duplicate_rows}")

    # 3. Procesar y reemplazar los datos que faltan (Missing N.A values)
    st.write("### 1.3 Manejo de valores faltantes (Missing N.A values)")
    missing_data = df.isnull().sum()
    missing_data = missing_data[missing_data > 0]
    if not missing_data.empty:
        st.write("Valores faltantes por columna:")
        st.dataframe(missing_data.to_frame(name='Valores Faltantes'))
        
        # Estrategias de imputación (ejemplos, ajustar según el contexto)
        # Para columnas numéricas, se puede usar la mediana o la media
        for col in ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement', 'sqft_living15', 'sqft_lot15']:
            if col in df.columns and df[col].isnull().any():
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                st.write(f"Columna '{col}': Valores nulos imputados con la mediana ({median_val})")
        
        # Para 'yr_renovated', si es 0 significa que no fue renovado. Si es NaN, podríamos imputar con 0.
        if 'yr_renovated' in df.columns and df['yr_renovated'].isnull().any():
            df['yr_renovated'].fillna(0, inplace=True)
            st.write("Columna 'yr_renovated': Valores nulos imputados con 0 (no renovado)")
            
    else:
        st.write("No hay valores faltantes en el dataset.")

    # 4. Manejo de outliers (Usaremos el método IQR para algunas columnas numéricas clave)
    st.write("### 1.4 Manejo de Outliers (IQR Method)")
    outlier_cols = ['price', 'sqft_living', 'sqft_lot', 'bedrooms', 'bathrooms', 'sqft_above', 'sqft_basement']
    for col in outlier_cols:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers_count = df[(df[col] < lower_bound) | (df[col] > upper_bound)].shape[0]
            if outliers_count > 0:
                st.write(f"Columna '{col}': Se detectaron {outliers_count} outliers.")
                # Opción 1: Capping (limitar los valores a los límites IQR)
                df[col] = np.clip(df[col], lower_bound, upper_bound)
                st.write(f"Outliers en '{col}' fueron tratados con capping (limitados a los límites IQR).")
            else:
                st.write(f"Columna '{col}': No se detectaron outliers significativos con el método IQR.")

    # 5. Re-escalar las variables / Normalizar (Opcional, dependiendo del modelo a usar)
    # Para visualización, no siempre es necesario, pero si se fuera a usar ML podría ser útil.
    # No lo implementaremos aquí a menos que sea explícitamente necesario para ML.

    # 6. Data Binning (Ejemplo: crear categorías de precios)
    st.write("### 1.5 Data Binning (Creación de categorías de precio)")
    if 'price' in df.columns:
        bins = [0, 200000, 400000, 600000, 800000, 1000000, np.inf]
        labels = ['Bajo', 'Medio-Bajo', 'Medio', 'Medio-Alto', 'Alto', 'Muy Alto']
        df['price_category'] = pd.cut(df['price'], bins=bins, labels=labels, right=False)
        st.write("Se ha creado la columna 'price_category' basada en el precio.")
        st.dataframe(df['price_category'].value_counts().to_frame())

    # 7. Conversión Numérico a categórico - Ya hicimos un ejemplo con 'price_category'
    # 8. Conversión Categórico a Numérico (Ejemplo: 'waterfront', 'view', 'condition', 'grade')
    st.write("### 1.6 Conversión Categórico a Numérico")
    # Estas columnas ya son numéricas pero representan categorías ordinales.
    # Podemos asegurarnos de que sean de tipo 'category' si queremos optimizar memoria,
    # o convertirlas a int si ya no lo son y representan categorías.
    
    # Para 'zipcode' si fuera necesario para un modelo y no se hiciera one-hot encoding
    # df['zipcode_encoded'] = df['zipcode'].astype('category').cat.codes

    return df

def perform_eda(df):
    """
    Realiza la exploración estadística y visualización de los datos.
    """
    st.subheader("2. Exploración Estadística y Visualización de Datos")

    st.write("### 2.1 Estadísticas Descriptivas")
    st.dataframe(df.describe())

    # Histogramas para variables numéricas clave
    st.write("### 2.2 Histogramas")
    numeric_cols = ['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'grade', 'yr_built']
    for col in numeric_cols:
        if col in df.columns:
            fig = px.histogram(df, x=col, title=f'Distribución de {col}')
            st.plotly_chart(fig)

    # Diagrámas de dispersión (Scatter Plots)
    st.write("### 2.3 Diagramas de Dispersión")
    st.write("Relación entre `sqft_living` y `price`:")
    fig_scatter_price_sqft = px.scatter(df, x='sqft_living', y='price',
                                        title='Precio vs. Pies Cuadrados Habitables',
                                        hover_data=['bedrooms', 'bathrooms', 'grade'])
    st.plotly_chart(fig_scatter_price_sqft)

    st.write("Relación entre `grade` y `price`:")
    fig_scatter_grade_price = px.scatter(df, x='grade', y='price',
                                         title='Precio vs. Calidad de Construcción (Grade)',
                                         hover_data=['sqft_living'])
    st.plotly_chart(fig_scatter_grade_price)
    
    st.write("Relación entre `yr_built` y `price`:")
    fig_scatter_yr_built_price = px.scatter(df, x='yr_built', y='price',
                                            title='Precio vs. Año de Construcción',
                                            hover_data=['sqft_living'])
    st.plotly_chart(fig_scatter_yr_built_price)

    # Cajas y Bigotes (Box Plots)
    st.write("### 2.4 Diagramas de Cajas y Bigotes")
    st.write("Distribución del precio por número de dormitorios:")
    fig_box_bedrooms = px.box(df, x='bedrooms', y='price',
                              title='Distribución del Precio por Número de Dormitorios')
    st.plotly_chart(fig_box_bedrooms)

    st.write("Distribución del precio por número de baños:")
    fig_box_bathrooms = px.box(df, x='bathrooms', y='price',
                               title='Distribución del Precio por Número de Baños')
    st.plotly_chart(fig_box_bathrooms)
    
    st.write("Distribución del precio por condición:")
    fig_box_condition = px.box(df, x='condition', y='price',
                                title='Distribución del Precio por Condición')
    st.plotly_chart(fig_box_condition)

    # Mapa de calor de correlación
    st.write("### 2.5 Mapa de Calor de Correlación")
    numeric_df = df.select_dtypes(include=np.number)
    
    # Filtrar columnas con baja varianza o pocas correlaciones para evitar errores o gráficos no informativos
    corr_matrix = numeric_df.corr()
    
    # Eliminar correlaciones con NaN (por ejemplo, si una columna tiene solo un valor)
    corr_matrix = corr_matrix.dropna(axis=0, how='all').dropna(axis=1, how='all')
    
    if not corr_matrix.empty:
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Mapa de Calor de Correlación entre Variables Numéricas')
        st.pyplot(plt)
    else:
        st.write("No hay suficientes variables numéricas para generar un mapa de calor de correlación.")

    # Visualización de la categoría de precio (si se creó)
    if 'price_category' in df.columns:
        st.write("### 2.6 Conteo de Propiedades por Categoría de Precio")
        fig_price_cat = px.bar(df['price_category'].value_counts().reset_index(), 
                               x='index', y='price_category',
                               labels={'index': 'Categoría de Precio', 'price_category': 'Número de Propiedades'},
                               title='Conteo de Propiedades por Categoría de Precio')
        st.plotly_chart(fig_price_cat)


# Título de la aplicación Streamlit
st.title("Análisis de Datos de Precios de Viviendas en King County")

# Cargar los datos
df_original = load_data(DATA_URL)

if not df_original.empty:
    st.write("Datos originales cargados (primeras 5 filas):")
    st.dataframe(df_original.head())

    # Preparar los datos
    df_processed = prepare_data(df_original.copy()) # Usamos una copia para no modificar el original

    # Realizar la exploración de datos
    perform_eda(df_processed)
