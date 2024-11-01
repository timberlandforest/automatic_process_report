import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import os

# Definir las columnas por área de proceso
areas_de_proceso = {
    'Combustion': ['ssq_ton_d', 'pct_ssq', 'Prim', 'Sec', 'Sec Alt', 'Terc', 'Cuat', 'Ratio_aircomb_liq', 'Out_gas_temp'],
    'Vapor': ['Ratio_Steam_Stream', 'temp_lp_vapor_post_vv', 'Atem'],
    'Ensuciamiento': ['T15', 'Soiling_rate_point', 'Diff_Press_SC', 'Diff_Press_BG', 'Diff_Press_ECO1', 'Diff_Press_ECO2'],
    'Licor Verde': ['red_TK_disolv_pct', 'sulfidez_LV_crudo_pct'],
    'Emisiones': ['cems1_nox', 'cems1_mp10', 'cems1_so2', 'cems1_trs', 'cems1_co']
}

# Función para calcular límites basados en percentiles
def calcular_limites(df, columnas, percentil_inferior=5, percentil_superior=95):
    limites = {}
    for columna in columnas:
        if pd.api.types.is_numeric_dtype(df[columna]):
            limite_inferior = df[columna].quantile(percentil_inferior / 100)
            limite_superior = df[columna].quantile(percentil_superior / 100)
            limites[columna] = {'limite_inferior': limite_inferior, 'limite_superior': limite_superior}
        else:
            limites[columna] = {'limite_inferior': None, 'limite_superior': None}
    return limites

# Función para graficar y guardar las imágenes
def graficar_media_por_hora_con_limites(df, columnas, limites, area="General"):
    image_paths = []
    df['ts'] = pd.to_datetime(df['ts'])
    df_hora = df.set_index('ts').resample('H').mean().reset_index()

    if not os.path.exists('report_images'):
        os.makedirs('report_images')

    for columna in columnas:
        df_hora[f'{columna}_movil_8h'] = df_hora[columna].rolling(window=8).mean()
        df_hora[f'{columna}_movil_30h'] = df_hora[columna].rolling(window=30).mean()

        plt.figure(figsize=(10, 6))
        sns.lineplot(x='ts', y=columna, data=df_hora, label=columna)
        sns.lineplot(x='ts', y=f'{columna}_movil_8h', data=df_hora, label=f'{columna} (Media móvil 8h)')
        sns.lineplot(x='ts', y=f'{columna}_movil_30h', data=df_hora, label=f'{columna} (Media móvil 30h)')

        limite_inferior = limites.get(columna, {}).get('limite_inferior', None)
        limite_superior = limites.get(columna, {}).get('limite_superior', None)

        if limite_inferior is not None:
            plt.axhline(y=limite_inferior, color='red', linestyle='--', label='Límite Inferior')
        if limite_superior is not None:
            plt.axhline(y=limite_superior, color='green', linestyle='--', label='Límite Superior')

        plt.title(f'{columna} ({area}) - Medias móviles y límites')
        plt.xlabel('Fecha')
        plt.ylabel('Valor')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()

        image_path = f'report_images/{columna}_{area}.png'
        plt.savefig(image_path)
        image_paths.append(image_path)
        plt.close()

    return image_paths

# Función para generar el reporte HTML
def generar_reporte_html(imagenes_por_area):
    html_content = "<html><head><title>Reporte de Proceso</title></head><body>"
    html_content += "<h1>Reporte de Visualización de Procesos</h1>"

    for area, imagenes in imagenes_por_area.items():
        html_content += f"<h2>Área: {area}</h2>"
        for imagen in imagenes:
            html_content += f'<img src="{imagen}" width="800"><br>'

    html_content += "</body></html>"

    with open("reporte_proceso.html", "w") as file:
        file.write(html_content)

    st.success("Reporte HTML generado: reporte_proceso.html")

# Aplicación en Streamlit
st.title("Reporte Procesos Automatizado")

# Cargar los datos directamente desde un archivo en la carpeta
archivo_csv = "data_caldera.csv"  # Reemplaza con el nombre de tu archivo CSV
if os.path.exists(archivo_csv):
    df = pd.read_csv(archivo_csv)
    df['ts'] = pd.to_datetime(df['ts'])  # Asegurarse de que la columna de tiempo sea datetime

    # Seleccionar el rango de fechas para el reporte
    fecha_inicio, fecha_fin = st.date_input(
        "Selecciona el rango de fechas",
        value=(df['ts'].min().date(), df['ts'].max().date())
    )

    # Convertir fechas seleccionadas a datetime para hacer coincidir tipos
    fecha_inicio = pd.to_datetime(fecha_inicio)
    fecha_fin = pd.to_datetime(fecha_fin)

    # Filtrar el DataFrame según el rango de fechas seleccionado
    df_filtrado = df[(df['ts'] >= fecha_inicio) & (df['ts'] <= fecha_fin)]

    # Opción para generar un informe general o por subsistema
    tipo_reporte = st.radio("¿Deseas generar un reporte general o por subsistema?", ('Por subsistema', 'General'))

    if tipo_reporte == 'Por subsistema':
        # Selección del área de proceso
        area_seleccionada = st.selectbox("Seleccionar un área de proceso", list(areas_de_proceso.keys()))

        # Selección de columnas
        columnas_seleccionadas = st.multiselect("Selecciona las columnas a graficar", areas_de_proceso[area_seleccionada])

        # Botón para generar las gráficas
        if st.button("Generar informe"):
            if len(columnas_seleccionadas) > 0:
                limites_calculados = calcular_limites(df_filtrado, columnas_seleccionadas)
                imagenes = graficar_media_por_hora_con_limites(df_filtrado, columnas_seleccionadas, limites_calculados, area_seleccionada)
                imagenes_por_area = {area_seleccionada: imagenes}
                generar_reporte_html(imagenes_por_area)

                for img in imagenes:
                    st.image(img, caption=f'Gráfica de {img}', use_column_width=True)
            else:
                st.warning("Seleccionar al menos una columna para graficar.")
    
    elif tipo_reporte == 'General':
        # Botón para generar el informe general
        if st.button("Generar Informe General"):
            imagenes_por_area = {}

            for area, columnas in areas_de_proceso.items():
                limites_calculados = calcular_limites(df_filtrado, columnas)
                imagenes = graficar_media_por_hora_con_limites(df_filtrado, columnas, limites_calculados, area=area)
                imagenes_por_area[area] = imagenes

                for img in imagenes:
                    st.image(img, caption=f'Gráfica de {img}', use_column_width=True)

            generar_reporte_html(imagenes_por_area)
else:
    st.error(f"El archivo {archivo_csv} no se encuentra en la carpeta.")
