import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import os
import tempfile
from xhtml2pdf import pisa

# Define process columns by area
areas_de_proceso = {
    'Combustion': ['ssq [ton/d]', 'pct_ssq [%]', 'liq_temp [°C]', 'Prim', 'Sec', 'Sec Alt', 'Terc', 'Cuat', 'Ratio_aircomb_liq', 'Out_gas_temp'],
    'Vapor': ['Ratio_Steam_Stream', 'temp_lp_vapor_post_vv', 'Atem'],
    'Ensuciamiento': ['T15', 'Soiling_rate_point', 'Diff_Press_SC', 'Diff_Press_BG', 'Diff_Press_ECO1', 'Diff_Press_ECO2', 'heat_coef_SH1 [kJ/m2C]', 'heat_coef_SH2 [kJ/m2C]', 'heat_coef_SH3 [kJ/m2C]', 'heat_coef_SH4 [kJ/m2C]'],
    'Licor Verde': ['reduction_lab [%]', 'alcali_lv_lab [g/L]', 'sulfidez_lab [%]', 'reduction_i [%]', 'alcali_lv_i [g/L]', 'sulfidez_i [%]'],
    'Emisiones': ['cems1_nox', 'cems1_mp10', 'cems1_so2', 'cems1_trs', 'cems1_co', 'O2_left_cont [%]', 'O2_mid_cont [%]', 'O2_right_content [%]']
}

# Function to calculate limits based on percentiles
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

# Function to plot and save images
def graficar_media_por_hora_con_limites(df, columnas, limites, area="General"):
    image_paths = []
    df['ts'] = pd.to_datetime(df['ts'])
    df_hora = df.set_index('ts').resample('H').mean().reset_index()

    if not os.path.exists('report_images'):
        os.makedirs('report_images')

    for columna in columnas:
        df_hora[f'{columna}_movil_10h'] = df_hora[columna].rolling(window=10).mean()

        plt.figure(figsize=(10, 6))
        sns.lineplot(x='ts', y=columna, data=df_hora, label=columna)
        sns.lineplot(x='ts', y=f'{columna}_movil_10h', data=df_hora, label=f'{columna} (Media móvil 10h)')

        limite_inferior = limites.get(columna, {}).get('limite_inferior', None)
        limite_superior = limites.get(columna, {}).get('limite_superior', None)

        if limite_inferior is not None:
            plt.axhline(y=limite_inferior, color='red', linestyle='--', label='Límite Inferior')
        if limite_superior is not None:
            plt.axhline(y=limite_superior, color='green', linestyle='--', label='Límite Superior')

        plt.title(f'{columna} ({area})')
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

# Function to generate the HTML report and convert to PDF
def generar_reporte_html_y_pdf(imagenes_por_area):
    html_content = "<html><head><title>Reporte de Proceso</title></head><body>"
    html_content += "<h1>Reporte de Visualización de Procesos</h1>"

    for area, imagenes in imagenes_por_area.items():
        html_content += f"<h2>Área: {area}</h2>"
        for imagen in imagenes:
            html_content += f'<img src="{imagen}" width="800"><br>'

    html_content += "</body></html>"

    # Save HTML content to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_html:
        tmp_html.write(html_content.encode('utf-8'))
        tmp_html_path = tmp_html.name

    # Convert HTML to PDF using xhtml2pdf
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
        pdf_path = tmp_pdf.name
        with open(tmp_pdf.name, "wb") as pdf_file:
            pisa_status = pisa.CreatePDF(html_content, dest=pdf_file)
            if pisa_status.err:
                st.error("Hubo un error generando el PDF.")
            else:
                st.success("Reporte PDF generado.")

    # Streamlit download button for PDF
    with open(pdf_path, "rb") as pdf_file:
        st.download_button(
            label="Descargar Informe",
            data=pdf_file,
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

    # Date range selection for the report
    fecha_inicio, fecha_fin = st.date_input(
        "Selecciona el rango de fechas",
        value=(df['ts'].min().date(), df['ts'].max().date())
    )

    fecha_inicio = pd.to_datetime(fecha_inicio)
    fecha_fin = pd.to_datetime(fecha_fin)

    df_filtrado = df[(df['ts'] >= fecha_inicio) & (df['ts'] <= fecha_fin)]

    tipo_reporte = st.radio("¿Deseas generar un reporte general o por subsistema?", ('Por subsistema', 'General'))

    if tipo_reporte == 'Por subsistema':
        area_seleccionada = st.selectbox("Seleccionar un área de proceso", list(areas_de_proceso.keys()))
        columnas_seleccionadas = st.multiselect("Selecciona las columnas a graficar", areas_de_proceso[area_seleccionada])

        if st.button("Generar informe"):
            if columnas_seleccionadas:
                limites_calculados = calcular_limites(df_filtrado, columnas_seleccionadas)
                imagenes = graficar_media_por_hora_con_limites(df_filtrado, columnas_seleccionadas, limites_calculados, area_seleccionada)
                imagenes_por_area = {area_seleccionada: imagenes}
                generar_reporte_html_y_pdf(imagenes_por_area)

                for img in imagenes:
                    st.image(img, caption=f'Gráfica de {img}', use_column_width=True)
            else:
                st.warning("Seleccionar al menos una columna para graficar.")
    
    elif tipo_reporte == 'General':
        if st.button("Generar Informe"):
            imagenes_por_area = {}

            for area, columnas in areas_de_proceso.items():
                limites_calculados = calcular_limites(df_filtrado, columnas)
                imagenes = graficar_media_por_hora_con_limites(df_filtrado, columnas, limites_calculados, area=area)
                imagenes_por_area[area] = imagenes

                for img in imagenes:
                    st.image(img, caption=f'Gráfica de {img}', use_column_width=True)

            generar_reporte_html_y_pdf(imagenes_por_area)
else:
    st.error(f"El archivo {archivo_csv} no se encuentra en la carpeta.")
    
