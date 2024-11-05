import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import streamlit as st
import os
import tempfile
from xhtml2pdf import pisa
import re
from io import BytesIO

# Define process columns by area
areas_de_proceso = {
    'Combustion': ['ssq [ton/d]', 'pct_ssq [%]', 'liq_temp [°C]', 'Prim', 'Sec', 'Sec Alt', 'Terc', 'Cuat', 'Ratio_aircomb_liq', 'Out_gas_temp'],
    'Vapor': ['Ratio_Steam_Stream', 'temp_lp_vapor_post_vv [°C]', 'Atem'],
    'Ensuciamiento': ['T15', 'Soiling_rate_point', 'Diff_Press_SC', 'Diff_Press_BG', 'Diff_Press_ECO1', 'Diff_Press_ECO2', 'heat_coef_SH1 [kJ/m2C]', 'heat_coef_SH2 [kJ/m2C]', 'heat_coef_SH3 [kJ/m2C]', 'heat_coef_SH4 [kJ/m2C]'],
    'Licor Verde': ['reduction_lab [%]', 'alcali_lv_lab [g/L]', 'sulfidez_lab [%]', 'reduction_i [%]', 'alcali_lv_i [g/L]', 'sulfidez_i [%]'],
    'Emisiones': ['cems1_nox', 'cems1_mp10', 'cems1_so2', 'cems1_trs', 'cems1_co', 'O2_left_cont [%]', 'O2_mid_cont [%]', 'O2_right_content [%]']
}

# Helper function to sanitize file names


def sanitize_filename(name):
    return re.sub(r'[\\/*?:"<>|\[\]\/]', '_', name)

# Function to calculate limits based on percentiles


def calcular_limites(df, columnas, percentil_inferior=5, percentil_superior=95):
    limites = {}
    for columna in columnas:
        if pd.api.types.is_numeric_dtype(df[columna]):
            limite_inferior = df[columna].quantile(percentil_inferior / 100)
            limite_superior = df[columna].quantile(percentil_superior / 100)
            limites[columna] = {
                'limite_inferior': limite_inferior, 'limite_superior': limite_superior}
        else:
            limites[columna] = {
                'limite_inferior': None, 'limite_superior': None}
    return limites

# Function to plot and save images with Matplotlib/Seaborn


def graficar_con_seaborn(df, columnas, limites, area="General"):
    image_paths = []
    df['ts'] = pd.to_datetime(df['ts'])
    df_hora = df.set_index('ts').resample('H').mean().reset_index()

    if not os.path.exists('report_images'):
        os.makedirs('report_images')

    for columna in columnas:
        df_hora[f'{columna}_movil_10h'] = df_hora[columna].rolling(
            window=10).mean()

        plt.figure(figsize=(10, 6))
        sns.lineplot(x='ts', y=columna, data=df_hora, label=columna)
        sns.lineplot(x='ts', y=f'{columna}_movil_10h',
                     data=df_hora, label=f'{columna} (Media móvil 10h)')

        limite_inferior = limites.get(columna, {}).get('limite_inferior', None)
        limite_superior = limites.get(columna, {}).get('limite_superior', None)

        if limite_inferior is not None:
            plt.axhline(y=limite_inferior, color='red',
                        linestyle='--', label='Límite Inferior')
        if limite_superior is not None:
            plt.axhline(y=limite_superior, color='green',
                        linestyle='--', label='Límite Superior')

        plt.title(f'{columna} ({area})')
        plt.xlabel('Fecha')
        plt.ylabel('Valor')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()

        image_path = f'report_images/{sanitize_filename(columna)}_{area}.png'
        plt.savefig(image_path)
        image_paths.append(image_path)
        plt.close()

    return image_paths

# Function to plot and save images with Plotly Express


def graficar_con_plotly(df, columnas, limites, area="General"):
    image_paths = []
    df['ts'] = pd.to_datetime(df['ts'])
    df_hora = df.set_index('ts').resample('H').mean().reset_index()

    if not os.path.exists('report_images'):
        os.makedirs('report_images')

    for columna in columnas:
        df_hora[f'{columna}_movil_10h'] = df_hora[columna].rolling(
            window=10).mean()

        fig = px.line(df_hora, x='ts', y=[columna, f'{columna}_movil_10h'],
                      labels={
                          columna: f'{columna}', f'{columna}_movil_10h': f'{columna} (Media Móvil 10h)'},
                      title=f'{columna} ({area})')

        fig.update_layout(
            legend_title_text='',
            legend=dict(
                yanchor="bottom", y=0.01, xanchor="left", x=0.01,
                font=dict(size=10)
            )
        )

        limite_inferior = limites.get(columna, {}).get('limite_inferior', None)
        limite_superior = limites.get(columna, {}).get('limite_superior', None)
        if limite_inferior is not None:
            fig.add_hline(y=limite_inferior, line_dash="dash",
                          line_color="red", annotation_text="Límite Inferior")
        if limite_superior is not None:
            fig.add_hline(y=limite_superior, line_dash="dash",
                          line_color="green", annotation_text="Límite Superior")

        image_path = f'report_images/{sanitize_filename(columna)}_{area}.png'
        fig.write_image(image_path)
        image_paths.append(image_path)

        st.plotly_chart(fig, use_container_width=True)

    return image_paths

# Function to generate the HTML report and convert to PDF


def generar_reporte_html_y_pdf(imagenes_por_area):
    html_content = "<html><head><title>Reporte de Proceso</title></head><body>"
    html_content += "<h1>Reporte de Visualización de Procesos</h1>"

    for area, imagenes in imagenes_por_area.items():
        html_content += f"<h2>Área: {area}</h2>"
        for imagen in imagenes:
            html_content += f'<img src="{imagen}" width="800"><br>'

    html_content += "</body></html>"

    pdf_buffer = BytesIO()
    pisa_status = pisa.CreatePDF(html_content, dest=pdf_buffer)
    pdf_buffer.seek(0)

    if pisa_status.err:
        st.error("Hubo un error generando el PDF.")
    else:
        st.success("Reporte PDF generado.")

        st.download_button(
            label="Descargar Informe",
            data=pdf_buffer,
            file_name="reporte_proceso.pdf",
            mime="application/pdf"
        )


# Streamlit App
st.title("Reporte Procesos Automatizado")

# Load data directly from a file in the folder
archivo_csv = "data_caldera_opt.csv"
if os.path.exists(archivo_csv):
    df = pd.read_csv(archivo_csv)
    df['ts'] = pd.to_datetime(df['ts'])

    fecha_inicio, fecha_fin = st.date_input(
        "Selecciona un rango de fechas",
        value=(df['ts'].min().date(), df['ts'].max().date())
    )

    fecha_inicio = pd.to_datetime(fecha_inicio)
    fecha_fin = pd.to_datetime(fecha_fin)

    df_filtrado = df[(df['ts'] >= fecha_inicio) & (df['ts'] <= fecha_fin)]

    tipo_reporte = st.radio(
        "¿Deseas generar un reporte general o por subsistema?", ('Por subsistema', 'General'))
    tipo_grafico = st.radio(
        "¿Con qué librería deseas generar las gráficas?", ('Matplotlib/Seaborn', 'Plotly Express'))

    if tipo_reporte == 'Por subsistema':
        area_seleccionada = st.selectbox(
            "Seleccionar un área de proceso", list(areas_de_proceso.keys()))
        columnas_seleccionadas = areas_de_proceso[area_seleccionada]

        if st.button("Generar informe"):
            limites_calculados = calcular_limites(
                df_filtrado, columnas_seleccionadas)

            if tipo_grafico == 'Matplotlib/Seaborn':
                imagenes = graficar_con_seaborn(
                    df_filtrado, columnas_seleccionadas, limites_calculados, area_seleccionada)
            else:
                imagenes = graficar_con_plotly(
                    df_filtrado, columnas_seleccionadas, limites_calculados, area_seleccionada)

            imagenes_por_area = {area_seleccionada: imagenes}
            generar_reporte_html_y_pdf(imagenes_por_area)

    elif tipo_reporte == 'General':
        if st.button("Generar Informe"):
            imagenes_por_area = {}

            for area, columnas in areas_de_proceso.items():
                limites_calculados = calcular_limites(df_filtrado, columnas)

                if tipo_grafico == 'Matplotlib/Seaborn':
                    imagenes = graficar_con_seaborn(
                        df_filtrado, columnas, limites_calculados, area=area)
                else:
                    imagenes = graficar_con_plotly(
                        df_filtrado, columnas, limites_calculados, area=area)

                imagenes_por_area[area] = imagenes

            generar_reporte_html_y_pdf(imagenes_por_area)
else:
    st.error(f"El archivo {archivo_csv} no se encuentra en la carpeta.")
