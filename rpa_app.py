import pandas as pd
import numpy as np
import PIconnect as PI
from datetime import datetime
from PIconnect.PIConsts import SummaryType
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from xhtml2pdf import pisa
import os

# Conexión al servidor PI
conectar = "CLMAPPIDATAPRD"  # Ajustar según el entorno: "CLARAPIDATAPRD" para Arauco, "CLSCLPIDATAPRD" para Pi CENTRAL
servidor = PI.PIServer(server=conectar)
PI.PIConfig.DEFAULT_TIMEZONE = 'Etc/GMT+3'
print(f"Conectado al servidor {servidor}")

# Función para obtener información y metadatos de los tags
def obtener_taginfo(tags, startTime, endTime):
    dataset = servidor.search(tags)
    taginfo = {point.name: f"{point.description} [{point.units_of_measurement}]" for point in dataset}
    dias_analisis = (datetime.strptime(endTime, "%d-%m-%Y %H:%M") - datetime.strptime(startTime, "%d-%m-%Y %H:%M")).days
    print(f"Analizando {dias_analisis} días.")
    return dataset, taginfo, dias_analisis

# Definición de tags y periodo de análisis
tags = [
    '752FI4816_M1.MV', '752DI1363.MV', '752TIC1354.MV', '752FIC0741.MV', 
    '752FIC0780.MV', '752FIC0809.MV', '752FIC0850.MV', '752FIC0843.MV', 
    '752TI1033.MV', '752TI1059.MV', '752TI1084.MV', '752TI1109.MV', 
    '752FIC0433.MV', '752TI0610.MV', '752TI0578.MV', '752TIC0579.MV',
    '752TI0577.MV', '752TIC0580.MV', '752TIC0581.MV', '752TIC0588.MV', 
    '752TIC0582.MV', '752TIC0589.MV', '752TIC0590.MV', '752TIC0592.MV', 
    '752TIC0591.MV', '752TIC0593.MV', '752LAB065.QC', '752LAB064.QC',
    '752FI0611.MV', '752PIC0993.MV', '752PI1027.MV', '752PI1053.MV',
    '752PI1078.MV', '752PI1103.MV', '752PDI0994.MV', '752PDI0995.MV',
    '752PDI1000.MV', '752PDI1001.MV', '752PDI1006.MV', '752PDI1007.MV',
    '752PDI1012.MV', '752PDI1013.MV', '752LAB087.QC', '753LAB143.QC',
    '752AI1142.MV', '752AI1147.MV', '752AI1143.MV', '752AI1144.MV',
    '752AI1146.MV'
]

startTime, endTime, interval = "01-01-2024 00:00", "01-10-2024 00:00", '5m'
dataset, taginfo, dias_analisis = obtener_taginfo(tags, startTime, endTime)

# Crear DataFrame con valores interpolados y nombrar columnas
df = pd.concat([tag.interpolated_values(startTime, endTime, interval) for tag in dataset], axis=1)
df.columns = [taginfo.get(tag, "Tag no disponible") for tag in df.columns]
df.index = pd.to_datetime(df.index, format="%Y-%m-%d %H:%M")
df.reset_index(inplace=True)
df.rename(columns={'index': 'timestamp'}, inplace=True)
df = df.apply(pd.to_numeric, errors='coerce')

# Renombrar las columnas
new_column_names = [
    'ts',  # timestamp
    'ssq_ton_d',  # Sólidos secos quemados [ton/d]
    'pct_ssq',  # % Solidos secos a quemar CR 3 [%]
    'liq_temp', # Temperatura Licor Negro a Boquillas [°C]
    'fa_vp',  # Flujo aire a ventilador primario CR3 [Nm³/s]
    'fa_as',  # Flujo aire a anillo secundario CR3 [Nm³/s]
    'fa_asa',  # Flujo aire a anillo secundario alto CR3 [Nm³/s]
    'fa_at',  # Flujo aire a anillo terciario CR3 [Nm³/s]
    'fa_ac',  # Flujo aire a anillo cuaternario CR3 [Nm³/s]
    'temp_ac_VTI1',  # Temperatura aire combustion antes VTI #1 CR3 [°C]
    'temp_ac_VTI2',  # Temperatura aire combustion antes VTI #2 CR3 [°C]
    'temp_ac_VTI3',  # Temperatura aire combustion antes VTI #3 CR3 [°C]
    'temp_ac_VTI4',  # Temperatura aire combustion antes VTI #4 CR3 [°C]
    'fa_agua',  # Flujo agua alimentacion a CR3 [t/h]
    'temp_lp_vapor_post_vv',  # Temperatura linea principal de vapor post v/v de vapor CR3 [°C]
    'temp_vapor_pre_atemp2',  # Temperatura vapor pre atemperador 2 CR3 [°C]
    'temp_vapor_sobrec_post_atemp1',  # Temperatura Vapor sobrecalentado post atemperador 1 CR3 [°C]
    'temp_vapor_pre_atemp1',  # Temperatura vapor pre atemperador 1 CR3 [°C]
    'temp_vapor_sobrec_post_atemp2',  # Temperatura Vapor sobrecalentado post atemperador 2 CR3 [°C]
    'temp_vapor_sob_desc_sob_sec_1',  # Temperatura Vapor sobre Desc Sobrec Secund CR3 [°C] (primera aparición)
    'temp_vapor_sobrec_post_atemp3',  # Temperatura Vapor sobrecalentado post atemperador 3 CR3 [°C]
    'temp_vapor_sob_desc_sob_sec_2',  # Temperatura Vapor sobre Desc Sobrec Secund CR3 [°C] (segunda aparición)
    'temp_vapor_sobrec_post_atemp4',  # Temperatura Vapor sobrecalentado post atemperador 4 CR3 [°C]
    'temp_vapor_sobrec_post_sobrec3',  # Temperatura Vapor sobrecalentado post sobrecalentador 3 CR3 [°C]
    'temp_vapor_sobrec_post_atemp5',  # Temperatura Vapor sobrecalentado post atemperador 5 CR3 [°C]
    'temp_vapor_sobrec_post_sobrec3_2',  # Temperatura Vapor sobrecalentado post Sobrecalentador 3 CR3 [°C] (segundo uso)
    'temp_vapor_sobrec_post_atemp6',  # Temperatura Vapor sobrecalentado post atemperador 6 CR3 [°C]
    'K_cenizas_pct', # Análisis Potasio (%) Cenizas. CR3 [%]
    'Cl_cenizas_pct', # Análisis Cloruro (%) Cenizas. CR3 [%]
    'gen_vapor_ton_h', # Generación vapor CR [ton/h]
    'press_hogar', # Presión hogar de CR3 [kPa]
    'press_aire_comb_PPT1', # Presión aire combustión antes de PPT #1 CR3 [kPa]
    'press_aire_comb_PPT2', # Presión aire combustión antes de PPT #2 CR3 [kPa]
    'press_aire_comb_PPT3', # Presión aire combustión antes de PPT #3 CR3 [kPa]
    'press_aire_comb_PPT4', # Presión aire combustión antes de PPT #4 CR3 [kPa]
    'dif_press_comb_sobrec_izq', # Diferencial presión aire combustión sobrecalentado pared izq CR3 [kPa]
    'dif_press_comb_sobrec_der', # Diferencial presión aire combustión sobrecalentado pared Der CR3 [kPa]
    'dif_press_comb_sobrebgp_izq', # Diferencial presion aire combs sobre banco generador pared izq  CR3
    'dif_press_comb_sobrebgp_der', # Diferencial presion aire combs sobre banco generador pared der CR3 
    'dif_press_comb_eco2_izq', # Diferencial presión aire combustión sobre Eco 2 pared izq CR3 [kPa]
    'dif_press_comb_eco2_der', # Diferencial presión aire combustión sobre Eco 2 pared der CR3 [kPa] 
    'dif_press_comb_eco1_izq', # Diferencial presión aire combustión sobre Eco 1 pared izq CR3 [kPa]
    'dif_press_comb_eco1_der', # Diferencial presión aire combustión sobre Eco 1 pared der CR3 [kPa]
    'red_TK_disolv_pct', # Análisis Reducción TK Disolvedor [%]
    'sulfidez_LV_crudo_pct', # Análisis Sulfidez (%) LV Crudo. Estanque LV Crudo [%]
    'cems1_nox',  # CEMS 1 chimenea comun calderas Nox [mg/Nm³]
    'cems1_mp10',  # CEMS 1 chimenea comun calderas MP 10 [mg/Nm³]
    'cems1_so2',  # CEMS 1 chimenea comun calderas SO2 [mg/Nm³]
    'cems1_trs',  # CEMS 1 chimenea comun calderas TRS [mg/Nm³]
    'cems1_co'  # CEMS 1 chimenea comun calderas CO [mg/Nm³]
]

df.columns = new_column_names

# Interpolación de valores faltantes
def interpolar_valores(df):
    for columna in df.columns[1:]:
        df[columna] = df[columna].interpolate(method='linear', limit_direction='both')
    return df

df = interpolar_valores(df)

# Cálculo de variables específicas
total_flujo_aire = df[['fa_vp', 'fa_as', 'fa_asa', 'fa_at', 'fa_ac']].sum(axis=1)
df['Prim'] = df['fa_vp'] / total_flujo_aire
df['Sec'] = df['fa_as'] / total_flujo_aire
df['Sec Alt'] = df['fa_asa'] / total_flujo_aire
df['Terc'] = df['fa_at'] / total_flujo_aire
df['Cuat'] = df['fa_ac'] / total_flujo_aire
df['Ratio_aircomb_liq'] = total_flujo_aire / df['ssq_ton_d']
df['Out_gas_temp'] = df[['temp_ac_VTI1', 'temp_ac_VTI2', 'temp_ac_VTI3', 'temp_ac_VTI4']].mean(axis=1)
df['Ratio_Steam_Stream'] = df['fa_agua'] / df['ssq_ton_d']

# Cálculo de columna Atemperación corregida
df['Atem'] = (
    (df['temp_vapor_pre_atemp2'] - df['temp_vapor_sobrec_post_atemp1']) + 
    (df['temp_vapor_pre_atemp1'] - df['temp_vapor_sobrec_post_atemp2']) +
    (df['temp_vapor_sob_desc_sob_sec_1'] - df['temp_vapor_sobrec_post_atemp3']) +
    (df['temp_vapor_sob_desc_sob_sec_2'] - df['temp_vapor_sobrec_post_atemp4'])
)

# Cálculo de la columna T15
def calcular_T15(df):
    condiciones = [
        (df['Cl_cenizas_pct'] <= 2.88) & (df['K_cenizas_pct'] < 5.5),
        (df['Cl_cenizas_pct'] > 2.88) & (df['K_cenizas_pct'] < 5.5),
        (df['Cl_cenizas_pct'] <= 2.88) & (df['K_cenizas_pct'] >= 5.5)
    ]
    resultados = [
        6.0934 * df['Cl_cenizas_pct'] ** 2 - 93.985 * df['Cl_cenizas_pct'] + 817.59 + (1.891 - df['K_cenizas_pct']) * 9,
        -0.2729 * df['Cl_cenizas_pct'] + 598.33 + (1.891 - df['K_cenizas_pct']) * 9,
        11.689 * df['Cl_cenizas_pct'] ** 2 - 117.56 * df['Cl_cenizas_pct'] + 783.36 + (10.03 - df['K_cenizas_pct']) * 5.5
    ]
    defecto = -0.6933 * df['Cl_cenizas_pct'] + 543.41 + (10.03 - df['K_cenizas_pct']) * 5.5
    df['T15'] = np.select(condiciones, resultados, default=defecto)
    return df

df = calcular_T15(df)

# Cálculo de columnas adicionales
df['A'] = df['gen_vapor_ton_h'] * (1000/3600)
df['B'] = df['press_hogar'] * 1000
df['C'] = (df[['press_aire_comb_PPT1', 'press_aire_comb_PPT2', 'press_aire_comb_PPT3', 'press_aire_comb_PPT4']].sum(axis=1)) * 250
df['Soiling_rate_point'] = (df['B'] - df['C']) / (df['A'] ** 2)

df['Diff_Press_SC'] = (df['dif_press_comb_sobrec_izq'] + df['dif_press_comb_sobrec_der']) / 2
df['Diff_Press_BG'] = df['dif_press_comb_sobrebgp_izq'] + df['dif_press_comb_sobrebgp_der']
df['Diff_Press_ECO1'] = df['dif_press_comb_eco1_izq'] + df['dif_press_comb_eco1_der']
df['Diff_Press_ECO2'] = df['dif_press_comb_eco2_izq'] + df['dif_press_comb_eco2_der']


# Definir las columnas por área de proceso
areas_de_proceso = {
    'Combustion': ['ssq_ton_d', 'pct_ssq', 'liq_temp', 'Prim', 'Sec', 'Sec Alt', 'Terc', 'Cuat', 'Ratio_aircomb_liq', 'Out_gas_temp'],
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
            limites[columna] = {
                'limite_inferior': df[columna].quantile(percentil_inferior / 100),
                'limite_superior': df[columna].quantile(percentil_superior / 100)
            }
        else:
            limites[columna] = {'limite_inferior': None, 'limite_superior': None}
    return limites

# Función para graficar y guardar las imágenes con límites
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

# Función para generar el reporte HTML y convertirlo a PDF
def generar_reporte_html_y_pdf(imagenes_por_area):
    html_content = "<html><head><title>Reporte de Proceso</title></head><body>"
    html_content += "<h1>Reporte de Visualización de Procesos</h1>"

    for area, imagenes in imagenes_por_area.items():
        html_content += f"<h2>Área: {area}</h2>"
        for imagen in imagenes:
            html_content += f'<img src="{imagen}" width="800"><br>'

    html_content += "</body></html>"

    # Guardar el archivo HTML
    with open("reporte_proceso.html", "w") as file:
        file.write(html_content)

    # Convertir el HTML a PDF usando xhtml2pdf
    with open("reporte_proceso.pdf", "wb") as pdf_file:
        pisa_status = pisa.CreatePDF(html_content, dest=pdf_file)
        if pisa_status.err:
            st.error("Hubo un error generando el PDF.")
        else:
            st.success("Reporte PDF generado: reporte_proceso.pdf")

    # Botón de descarga en Streamlit
    with open("reporte_proceso.pdf", "rb") as pdf_file:
        st.download_button(
            label="Descargar Informe",
            data=pdf_file,
            file_name="reporte_proceso.pdf",
            mime="application/pdf"
        )

# Aplicación en Streamlit
st.title("Reporte Procesos Automatizado")

# Selección de rango de fechas en Streamlit
df['ts'] = pd.to_datetime(df['ts'])
fecha_inicio, fecha_fin = st.date_input(
    "Selecciona el rango de fechas",
    value=(df['ts'].min().date(), df['ts'].max().date())
)
fecha_inicio, fecha_fin = pd.to_datetime(fecha_inicio), pd.to_datetime(fecha_fin)
df_filtrado = df[(df['ts'] >= fecha_inicio) & (df['ts'] <= fecha_fin)]

# Selección de tipo de reporte en Streamlit
tipo_reporte = st.radio("¿Deseas generar un reporte general o por subsistema?", ('Por subsistema', 'General'))

# Generación de reporte por subsistema
if tipo_reporte == 'Por subsistema':
    area_seleccionada = st.selectbox("Seleccionar un área de proceso", list(areas_de_proceso.keys()))
    columnas_seleccionadas = st.multiselect("Selecciona las columnas a graficar", areas_de_proceso[area_seleccionada])

    if st.button("Generar informe") and columnas_seleccionadas:
        limites_calculados = calcular_limites(df_filtrado, columnas_seleccionadas)
        imagenes = graficar_media_por_hora_con_limites(df_filtrado, columnas_seleccionadas, limites_calculados, area_seleccionada)
        imagenes_por_area = {area_seleccionada: imagenes}
        generar_reporte_html_y_pdf(imagenes_por_area)

        for img in imagenes:
            st.image(img, caption=f'Gráfica de {img}', use_column_width=True)

# Generación de reporte general
elif tipo_reporte == 'General' and st.button("Generar Informe"):
    imagenes_por_area = {}
    for area, columnas in areas_de_proceso.items():
        limites_calculados = calcular_limites(df_filtrado, columnas)
        imagenes = graficar_media_por_hora_con_limites(df_filtrado, columnas, limites_calculados, area=area)
        imagenes_por_area[area] = imagenes
        for img in imagenes:
            st.image(img, caption=f'Gráfica de {img}', use_column_width=True)

    generar_reporte_html_y_pdf(imagenes_por_area)

    