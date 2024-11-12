import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import os
import tempfile
from xhtml2pdf import pisa
import re
from io import BytesIO

# Define process columns by area
areas_de_proceso = {
    'Combustion': [
        'Carga [TSS/d]', 'Solidos a quemado [%]', 'Temperatura LN a boquillas [°C]', 
        'Primario', 'Secundario', 'Secundario Alto', 'Terciario', 'Cuaternario', 
        'Aire de combustión/ carga de licor [Nm3/kg DS]', 'Temperatura de gases de salida [°C]'
    ],
    'Vapor': [
        'Ratio flujo de vapor/ [Ton vap/kg DS]', 'Temperatura de salida vapor [°C]', 'Atemperacion [°C]'
    ],
    'Ensuciamiento': [
        'Soiling_rate_point', 'Diff_Press_SC [kPa]', 'Diff_Press_BG [kPa]', 
        'Diff_Press_ECO1 [kPa]', 'Diff_Press_ECO2 [kPa]', 
        'heat_coef_SH1 [kJ/m2C]', 'heat_coef_SH2 [kJ/m2C]', 
        'heat_coef_SH3 [kJ/m2C]', 'heat_coef_SH4 [kJ/m2C]'
    ],
    'Licor Verde': [
        'reduction_ins [%]', 'alcali_lv_ins [g/L]', 'sulfidez_ins [%]', 
        'reduction_lab [%]', 'alcali_lv_lab [g/L]', 'sulfidez_lab [%]'
    ],
    'Emisiones': [
        'NOx [mg/Nm³]', 'Material particulado [mg/Nm³]', 'SO2 [mg/Nm³]', 
        'TRS [mg/Nm³]', 'CO [mg/Nm³]', 'O2_cont_left [%]', 
        'O2_cont_center [%]', 'O2_cont_right [%]', 'CO_cont_left_wall [%]', 
        'CO_cont_center [%]', 'CO_cont_right_wall [%]'
    ]
}

# Variables that should be grouped and not plotted individually
omit_individual_plots = {
    'Combustion': ['Primario', 'Secundario', 'Secundario Alto', 'Terciario', 'Cuaternario'],
    'Ensuciamiento': [
        'Diff_Press_SC [kPa]', 'Diff_Press_BG [kPa]', 'Diff_Press_ECO1 [kPa]', 
        'Diff_Press_ECO2 [kPa]', 'heat_coef_SH1 [kJ/m2C]', 
        'heat_coef_SH2 [kJ/m2C]', 'heat_coef_SH3 [kJ/m2C]'
    ],
    'Licor Verde': [
        'reduction_ins [%]', 'alcali_lv_ins [g/L]', 'sulfidez_ins [%]', 
        'reduction_lab [%]', 'alcali_lv_lab [g/L]', 'sulfidez_lab [%]'
    ],
    'Emisiones': [
        'O2_cont_left [%]', 'O2_cont_center [%]', 'O2_cont_right [%]', 
        'CO_cont_left_wall [%]', 'CO_cont_center [%]', 'CO_cont_right_wall [%]'
    ]
}

# Helper function to sanitize file names
def sanitize_filename(name):
    return re.sub(r'[\\/*?:"<>|\[\]\/]', '_', name)

# Helper function to check if a column exists in the DataFrame
def column_exists(df, column_name):
    return column_name in df.columns

# Function to calculate limits based on percentiles with column existence check
def calcular_limites(df, columnas, percentil_inferior=5, percentil_superior=95):
    limites = {}
    for columna in columnas:
        if column_exists(df, columna) and pd.api.types.is_numeric_dtype(df[columna]):
            limite_inferior = df[columna].quantile(percentil_inferior / 100)
            limite_superior = df[columna].quantile(percentil_superior / 100)
            limites[columna] = {
                'limite_inferior': limite_inferior, 'limite_superior': limite_superior
            }
        else:
            limites[columna] = {
                'limite_inferior': None, 'limite_superior': None
            }
    return limites

# Function to plot with Seaborn and Matplotlib
def graficar_con_seaborn(df, columnas, limites, area="General"):
    image_paths = []
    df['datetime'] = pd.to_datetime(df['datetime'])
    df_hora = df.set_index('datetime').resample('H').mean().reset_index()

    if not os.path.exists('report_images'):
        os.makedirs('report_images')

    # Skip variables that are in the omit_individual_plots for the current area
    columnas = [col for col in columnas if col not in omit_individual_plots.get(area, [])]

    for i in range(0, len(columnas), 2):
        plt.figure(figsize=(15, 5))
        for j, columna in enumerate(columnas[i:i+2]):
            if not column_exists(df_hora, columna):
                print(f"Warning: Column '{columna}' does not exist in the DataFrame and will be skipped.")
                continue

            plt.subplot(1, 2, j + 1)
            sns.lineplot(x='datetime', y=columna, data=df_hora, label=columna, linewidth = 2)

            limite_inferior = limites.get(columna, {}).get('limite_inferior', None)
            limite_superior = limites.get(columna, {}).get('limite_superior', None)

            if limite_inferior is not None:
                plt.axhline(y=limite_inferior, color='red', linestyle='--', label='Límite Inferior')
            if limite_superior is not None:
                plt.axhline(y=limite_superior, color='green', linestyle='--', label='Límite Superior')

            plt.title(f'{columna}', fontweight='bold')
            plt.grid(True)
            plt.xlabel('Fecha')
            plt.ylabel('Valor')
            plt.legend(loc='upper left')
            plt.xticks(rotation=45)
            plt.tight_layout()

        image_path = f'report_images/{sanitize_filename("_".join(columnas[i:i+2]))}_{area}.png'
        plt.savefig(image_path)
        image_paths.append(image_path)
        plt.close()

        st.image(image_path)

    return image_paths
    
# Functions for extra visualizations
def graficar_distribucion_aire(df, tipo_grafico):
    image_path = "report_images/Air_Distribution_Combustion.png"
    if tipo_grafico == 'Plotly Express':
        fig = go.Figure()
        variables = ['Primario', 'Secundario', 'Secundario Alto', 'Terciario', 'Cuaternario']
        for var in variables:
            fig.add_trace(go.Scatter(x=df['datetime'], y=df[var], mode='lines', name=var))
        fig.update_layout(
            title="Air Distribution [%]",
            xaxis_title="Fecha",
            yaxis_title="Valor",
            legend=dict(yanchor="bottom", y=0.01, xanchor="left", x=0.01, font=dict(size=10))
        )
        fig.write_image(image_path)
        st.plotly_chart(fig, use_container_width=True)
    else:
        plt.figure(figsize=(14, 10))
        for var in ['Primario', 'Secundario', 'Secundario Alto', 'Terciario', 'Cuaternario']:
            plt.plot(df['datetime'], df[var], label=var, linewidth = 0.5)
        plt.title("Air Distribution [%]")
        plt.xlabel("Fecha")
        plt.ylabel("Valor")
        plt.legend(loc='upper left')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(image_path)
        st.pyplot()
    return image_path

def graficar_diferencia_presion(df, tipo_grafico):
    image_path = "report_images/Pressure_Diff_Ensuciamiento.png"
    if tipo_grafico == 'Plotly Express':
        fig = go.Figure()
        variables = ['Diff_Press_SC [kPa]', 'Diff_Press_BG [kPa]', 'Diff_Press_ECO1 [kPa]', 'Diff_Press_ECO2 [kPa]']
        for var in variables:
            fig.add_trace(go.Scatter(x=df['datetime'], y=df[var], mode='lines', name=var))
        fig.update_layout(
            title="Pressure_Diff [kPa]",
            xaxis_title="Fecha",
            yaxis_title="Valor",
            legend=dict(yanchor="bottom", y=0.01, xanchor="left", x=0.01, font=dict(size=10))
        )
        fig.write_image(image_path)
        st.plotly_chart(fig, use_container_width=True)
    else:
        plt.figure(figsize=(14, 10))
        for var in ['Diff_Press_SC [kPa]', 'Diff_Press_BG [kPa]', 'Diff_Press_ECO1 [kPa]', 'Diff_Press_ECO2 [kPa]']:
            plt.plot(df['datetime'], df[var], label=var, linewidth = 0.5)
        plt.title("Pressure_Diff [kPa]")
        plt.xlabel("Fecha")
        plt.ylabel("Valor")
        plt.legend(loc='upper left')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(image_path)
        st.pyplot()
    return image_path

def graficar_distribucion_heat_coef(df, tipo_grafico):
    image_path = "report_images/Heat_Coefficient_Distribution_Ensuciamiento.png"
    variables = ['heat_coef_SH1 [kJ/m2C]', 'heat_coef_SH2 [kJ/m2C]', 'heat_coef_SH3 [kJ/m2C]']

    if tipo_grafico == 'Plotly Express':
        fig = go.Figure()
        for var in variables:
            fig.add_trace(go.Scatter(x=df['datetime'], y=df[var], mode='lines', name=var))
        fig.update_layout(
            title="Heat Coefficient Distribution [kJ/m2C]",
            xaxis_title="Fecha",
            yaxis_title="Valor",
            legend=dict(yanchor="bottom", y=0.01, xanchor="left", x=0.01, font=dict(size=10))
        )
        fig.write_image(image_path)
        st.plotly_chart(fig, use_container_width=True)
    else:
        plt.figure(figsize=(14, 10))
        for var in variables:
            plt.plot(df['datetime'], df[var], label=var, linewidth = 0.5)
        plt.title("Heat Coefficient Distribution [kJ/m2C]")
        plt.xlabel("Fecha")
        plt.ylabel("Valor")
        plt.legend(loc='upper left')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(image_path)
        st.pyplot()

    return image_path

def graficar_comparacion_licor_verde(df, tipo_grafico):
    figs_paths = []
    comparisons = [
        ('reduction_lab [%]', 'reduction_ins [%]', 'Reduction Comparison [%]'),
        ('alcali_lv_lab [g/L]', 'alcali_lv_ins [g/L]', 'Alcali Comparison [g/L]'),
        ('sulfidez_lab [%]', 'sulfidez_ins [%]', 'Sulfidez Comparison [%]')
    ]

    for lab_var, inst_var, title in comparisons:
        # Check if both columns exist
        if not (column_exists(df, lab_var) and column_exists(df, inst_var)):
            print(f"Warning: One or both columns '{lab_var}' or '{inst_var}' do not exist in the DataFrame. Skipping plot '{title}'.")
            continue

        image_path = f"report_images/{sanitize_filename(title)}_Licor_Verde.png"
        if tipo_grafico == 'Plotly Express':
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['datetime'], y=df[lab_var], mode='lines', name=lab_var))
            fig.add_trace(go.Scatter(x=df['datetime'], y=df[inst_var], mode='lines', name=inst_var))
            fig.update_layout(
                title=title,
                xaxis_title="Fecha",
                yaxis_title="Valor",
                legend=dict(yanchor="bottom", y=0.01, xanchor="left", x=0.01, font=dict(size=10))
            )
            fig.write_image(image_path)
            st.plotly_chart(fig, use_container_width=True)
        else:
            plt.figure(figsize=(14, 10))
            plt.plot(df['datetime'], df[lab_var], label=lab_var, linewidth = 0.5)
            plt.plot(df['datetime'], df[inst_var], label=inst_var, linewidth = 0.5)
            plt.title(title)
            plt.xlabel("Fecha")
            plt.ylabel("Valor")
            plt.legend(loc='upper left')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(image_path)
            st.pyplot()

        figs_paths.append(image_path)
    return figs_paths

def graficar_contenido_oxigeno(df, tipo_grafico):
    image_path = "report_images/O2_Content_Emisiones.png"
    if tipo_grafico == 'Plotly Express':
        fig = go.Figure()
        variables = ['O2_cont_left [%]', 'O2_cont_center [%]', 'O2_cont_right [%]']
        for var in variables:
            fig.add_trace(go.Scatter(x=df['datetime'], y=df[var], mode='lines', name=var))
        fig.update_layout(
            title="O2 Content [%]",
            xaxis_title="Fecha",
            yaxis_title="Valor",
            legend=dict(yanchor="bottom", y=0.01, xanchor="left", x=0.01, font=dict(size=10))
        )
        fig.write_image(image_path)
        st.plotly_chart(fig, use_container_width=True)
    else:
        plt.figure(figsize=(14, 10))
        for var in ['O2_cont_left [%]', 'O2_cont_center [%]', 'O2_cont_right [%]']:
            plt.plot(df['datetime'], df[var], label=var, linewidth = 0.5)
        plt.title("O2 Content [%]")
        plt.xlabel("Fecha")
        plt.ylabel("Valor")
        plt.legend(loc='upper left')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(image_path)
        st.pyplot()
    return image_path

def graficar_contenido_monoxido(df, tipo_grafico):
    image_path = "report_images/CO_Content_Emisiones.png"
    if tipo_grafico == 'Plotly Express':
        fig = go.Figure()
        variables = ['CO_cont_left_wall [%]', 'CO_cont_center [%]', 'CO_cont_right_wall [%]']
        for var in variables:
            fig.add_trace(go.Scatter(x=df['datetime'], y=df[var], mode='lines', name=var))
        fig.update_layout(
            title="CO Content [%]",
            xaxis_title="Fecha",
            yaxis_title="Valor",
            legend=dict(yanchor="bottom", y=0.01, xanchor="left", x=0.01, font=dict(size=10))
        )
        fig.write_image(image_path)
        st.plotly_chart(fig, use_container_width=True)
    else:
        plt.figure(figsize=(14, 10))
        for var in ['CO_cont_left_wall [%]', 'CO_cont_center [%]', 'CO_cont_right_wall [%]']:
            plt.plot(df['datetime'], df[var], label=var, linewidth = 0.5)
        plt.title("CO Content [%]")
        plt.xlabel("Fecha")
        plt.ylabel("Valor")
        plt.legend(loc='upper left')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(image_path)
        st.pyplot()
    return image_path

# Function to plot with Plotly (including moving averages and limits)
def graficar_con_plotly(df, columnas, limites, area="General"):
    image_paths = []
    df['datetime'] = pd.to_datetime(df['datetime'])
    df_hora = df.set_index('datetime').resample('H').mean().reset_index()

    if not os.path.exists('report_images'):
        os.makedirs('report_images')

    # Skip variables that are in the omit_individual_plots for the current area
    columnas = [col for col in columnas if col not in omit_individual_plots.get(area, [])]

    for i in range(0, len(columnas), 2):
        figs = []
        for columna in columnas[i:i+2]:
            # Check if the column exists in df_hora before plotting
            if columna not in df_hora.columns:
                st.warning(f"Column '{columna}' is missing in the data and will be skipped.")
                continue
            
            fig = px.line(df_hora, x='datetime', y=columna, labels={columna: columna}, title=f'{columna} ({area})')
            fig.update_layout(
                legend_title_text='',
                legend=dict(yanchor="bottom", y=0.01, xanchor="left", x=0.01, font=dict(size=10))
            )

            limite_inferior = limites.get(columna, {}).get('limite_inferior', None)
            limite_superior = limites.get(columna, {}).get('limite_superior', None)
            if limite_inferior is not None:
                fig.add_hline(y=limite_inferior, line_dash="dash", line_color="red", annotation_text="Límite Inferior")
            if limite_superior is not None:
                fig.add_hline(y=limite_superior, line_dash="dash", line_color="green", annotation_text="Límite Superior")

            image_path = f'report_images/{sanitize_filename(columna)}_{area}.png'
            fig.write_image(image_path)
            image_paths.append(image_path)
            figs.append(fig)

        # Display plots in Streamlit if figures were created
        if figs:
            st.plotly_chart(figs[0], use_container_width=True)
            if len(figs) > 1:
                st.plotly_chart(figs[1], use_container_width=True)

    return image_paths

# Function to generate the HTML and PDF report
def generar_reporte_html_y_pdf(imagenes_por_area):
    html_content = "<html><head><title>Reporte de Proceso</title></head><body>"
    html_content += "<h1>Reporte de Visualización de Procesos</h1>"

    for area, imagenes in imagenes_por_area.items():
        html_content += f"<h2>Área: {area}</h2>"
        for i in range(0, len(imagenes), 2):
            html_content += '<div style="display: flex; justify-content: space-between;">'
            for img in imagenes[i:i+2]:
                html_content += f'<img src="{img}" width="49%"><br>'
            html_content += '</div>'

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

archivo_csv = "data_caldera_opt.csv"
if os.path.exists(archivo_csv):
    df = pd.read_csv(archivo_csv)
    df['datetime'] = pd.to_datetime(df['datetime'])

    fecha_inicio, fecha_fin = st.date_input("Selecciona un rango de fechas",
                                            value=(df['datetime'].min().date(), df['datetime'].max().date()))
    fecha_inicio, fecha_fin = pd.to_datetime(fecha_inicio), pd.to_datetime(fecha_fin)
    df_filtrado = df[(df['datetime'] >= fecha_inicio) & (df['datetime'] <= fecha_fin)]

    tipo_reporte = st.radio("¿Deseas generar un reporte general o por subsistema?", ('Por subsistema', 'General'))
    tipo_grafico = st.radio("¿Con qué librería deseas generar las gráficas?", ('Matplotlib/Seaborn', 'Plotly Express'))

    if tipo_reporte == 'Por subsistema':
        area_seleccionada = st.selectbox("Seleccionar un área de proceso", list(areas_de_proceso.keys()))
        columnas_seleccionadas = areas_de_proceso[area_seleccionada]

        if st.button("Generar informe"):
            limites_calculados = calcular_limites(df_filtrado, columnas_seleccionadas)

            if tipo_grafico == 'Matplotlib/Seaborn':
                imagenes = graficar_con_seaborn(df_filtrado, columnas_seleccionadas, limites_calculados, area_seleccionada)
            else:
                imagenes = graficar_con_plotly(df_filtrado, columnas_seleccionadas, limites_calculados, area_seleccionada)

            # Agregar imágenes adicionales según el área seleccionada
            if area_seleccionada == 'Combustion':
                image_path = graficar_distribucion_aire(df_filtrado, tipo_grafico)
                imagenes.append(image_path)
            elif area_seleccionada == 'Ensuciamiento':
                image_path_press_diff = graficar_diferencia_presion(df_filtrado, tipo_grafico)
                imagenes.append(image_path_press_diff)
                image_path_heat_coef = graficar_distribucion_heat_coef(df_filtrado, tipo_grafico)
                imagenes.append(image_path_heat_coef)
            elif area_seleccionada == 'Licor Verde':
                figs_licor = graficar_comparacion_licor_verde(df_filtrado, tipo_grafico)
                imagenes.extend(figs_licor)
            elif area_seleccionada == 'Emisiones':
                image_path_oxigeno = graficar_contenido_oxigeno(df_filtrado, tipo_grafico)
                imagenes.append(image_path_oxigeno)
                image_path_monoxido = graficar_contenido_monoxido(df_filtrado, tipo_grafico)
                imagenes.append(image_path_monoxido)

            imagenes_por_area = {area_seleccionada: imagenes}
            generar_reporte_html_y_pdf(imagenes_por_area)

    elif tipo_reporte == 'General':
        columnas_seleccionadas = [col for cols in areas_de_proceso.values() for col in cols]

        if st.button("Generar informe"):
            limites_calculados = calcular_limites(df_filtrado, columnas_seleccionadas)

            if tipo_grafico == 'Matplotlib/Seaborn':
                imagenes = graficar_con_seaborn(df_filtrado, columnas_seleccionadas, limites_calculados, "General")
            else:
                imagenes = graficar_con_plotly(df_filtrado, columnas_seleccionadas, limites_calculados, "General")

            # Agregar imágenes adicionales para cada área en el informe general
            for area, columnas in areas_de_proceso.items():
                if area == 'Combustion':
                    image_path = graficar_distribucion_aire(df_filtrado, tipo_grafico)
                    imagenes.append(image_path)
                elif area == 'Ensuciamiento':
                    image_path_press_diff = graficar_diferencia_presion(df_filtrado, tipo_grafico)
                    imagenes.append(image_path_press_diff)
                    image_path_heat_coef = graficar_distribucion_heat_coef(df_filtrado, tipo_grafico)
                    imagenes.append(image_path_heat_coef)
                elif area == 'Licor Verde':
                    figs_licor = graficar_comparacion_licor_verde(df_filtrado, tipo_grafico)
                    imagenes.extend(figs_licor)
                elif area == 'Emisiones':
                    image_path_oxigeno = graficar_contenido_oxigeno(df_filtrado, tipo_grafico)
                    imagenes.append(image_path_oxigeno)
                    image_path_monoxido = graficar_contenido_monoxido(df_filtrado, tipo_grafico)
                    imagenes.append(image_path_monoxido)

            imagenes_por_area = {"General": imagenes}
            generar_reporte_html_y_pdf(imagenes_por_area)
else:
    st.error(f"El archivo {archivo_csv} no se encuentra en la carpeta.")
