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
import base64

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

# Límites para las variables por áreas de proceso

limites_proceso = {
    'Combustion': {
        'Carga [TSS/d]': {'inferior': 5000, 'superior': 7000},
        'Solidos a quemado [%]': {'inferior': 77, 'superior': 81},
        'Temperatura LN a boquillas [°C]': {'inferior': 130, 'superior': 150},
        'Aire de combustión/ carga de licor [Nm3/kg DS]': {'inferior': 3, 'superior': 4.5},
        'Temperatura de gases de salida [°C]': {'superior': 213},
    },
    'Vapor': {
        'Ratio flujo de vapor/ [Ton vap/kg DS]': {'inferior': 3.6, 'superior': 4.2},
        'Temperatura de salida vapor [°C]': {'superior': 495},
    },
    'Ensuciamiento': {
        'Soiling_rate_point': {'superior': 0.065},
    },
    'Licor Verde': {
        'reduction_ins [%]': {'inferior': 88, 'superior': 98},
        'alcali_lv_ins [g/L]': {'inferior': 152, 'superior': 178},
        'sulfidez_ins [%]': {'inferior': 25, 'superior': 35},
    },
    'Emisiones': {
        'NOx [mg/Nm³]': {'superior': 190},
        'Material particulado [mg/Nm³]': {'superior': 20},
        'SO2 [mg/Nm³]': {'superior': 35},
        'TRS [mg/Nm³]': {'superior': 2},
        'CO [mg/Nm³]': {'superior': 375},
    }
}

# Helper function to sanitize file names

def sanitize_filename(name):
    return re.sub(r'[\\/*?:"<>|\[\]\/]', '_', name)

# Helper function to check if a column exists in the DataFrame

def column_exists(df, column_name):
    return column_name in df.columns

# Function to calculate limits based on percentiles with column existence check

def calcular_limites(df, columnas, area=None, percentil_inferior=5, percentil_superior=95):
    """
    Calcula los límites utilizando los límites reales definidos en `limites_proceso`.
    Si no existen límites definidos, calcula automáticamente los límites basados en percentiles.

    Args:
        df (pd.DataFrame): DataFrame con los datos.
        columnas (list): Lista de columnas para calcular los límites.
        area (str): Área del proceso para buscar límites reales en `limites_proceso`.
        percentil_inferior (float): Percentil inferior para el cálculo automático (default=5).
        percentil_superior (float): Percentil superior para el cálculo automático (default=95).

    Returns:
        dict: Diccionario con los límites para cada columna.
    """
    limites = {}
    limites_reales = limites_proceso.get(area, {}) if area else {}

    for columna in columnas:
        # Verifica si la columna existe y tiene datos numéricos
        if column_exists(df, columna) and pd.api.types.is_numeric_dtype(df[columna]):
            # Limpiar valores nulos
            df_cleaned = df[columna].dropna()

            # Obtener límites reales definidos
            limite_real_inferior = limites_reales.get(columna, {}).get('inferior', None)
            limite_real_superior = limites_reales.get(columna, {}).get('superior', None)

            # Calcular límites automáticos si los reales no están definidos
            if limite_real_inferior is None and limite_real_superior is None:
                limite_inferior = df_cleaned.quantile(percentil_inferior / 100)
                limite_superior = df_cleaned.quantile(percentil_superior / 100)
            else:
                limite_inferior = limite_real_inferior
                limite_superior = limite_real_superior

            # Guardar los límites en el diccionario
            limites[columna] = {
                'limite_inferior': limite_inferior,
                'limite_superior': limite_superior
            }
        else:
            # Si la columna no existe o no es numérica, dejar los límites como None
            limites[columna] = {
                'limite_inferior': None,
                'limite_superior': None
            }

    return limites

# Function to plot with Seaborn and Matplotlib


def graficar_con_seaborn(df, columnas, limites, area="General"):
    image_paths = []
    df['datetime'] = pd.to_datetime(df['datetime'])
    df_hora = df.set_index('datetime').resample('h').mean().reset_index()

    if not os.path.exists('report_images'):
        os.makedirs('report_images')

    # Excluir columnas agrupadas para evitar duplicados
    columnas = [col for col in columnas if col not in omit_individual_plots.get(area, [])]

    for i in range(0, len(columnas), 2):
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))

        for j, columna in enumerate(columnas[i:i+2]):
            if not column_exists(df_hora, columna):
                print(f"Warning: Column '{columna}' does not exist in the DataFrame and will be skipped.")
                continue

            sns.lineplot(x='datetime', y=columna, data=df_hora, label=columna, linewidth=2, ax=axs[j])

            # Obtener límites calculados o reales
            limite_inferior = limites.get(columna, {}).get('limite_inferior', None)
            limite_superior = limites.get(columna, {}).get('limite_superior', None)

            print(f"Columna: {columna}, Límite inferior: {limite_inferior}, Límite superior: {limite_superior}")

            # Ajustar rango del eje para incluir límites
            min_y = min(df_hora[columna].min(), limite_inferior) if limite_inferior is not None else df_hora[columna].min()
            max_y = max(df_hora[columna].max(), limite_superior) if limite_superior is not None else df_hora[columna].max()

            # Expansión de márgenes para asegurar que los límites sean visibles
            margin = 0.25 * abs(max_y - min_y)  # Margen del 10% del rango
            axs[j].set_ylim(min_y - margin, max_y + margin)

            # Añadir líneas horizontales para los límites si existen
            if limite_inferior is not None:
                axs[j].axhline(y=limite_inferior, color='red', linestyle='--', label='Límite Inferior')
            if limite_superior is not None:
                axs[j].axhline(y=limite_superior, color='green', linestyle='--', label='Límite Superior')

            axs[j].set_title(f'{columna}', fontsize=12, fontweight='bold')
            axs[j].set_xlabel('Fecha', fontsize=12, fontweight='bold')
            axs[j].set_ylabel('Valor', fontsize=12, fontweight='bold')
            axs[j].legend(loc='upper left')
            plt.setp(axs[j].get_xticklabels(), rotation=45)

        # Remover subplots vacíos si hay un número impar de columnas
        if len(columnas[i:i+2]) < 2:
            fig.delaxes(axs[1])

        plt.tight_layout(pad=2)
        image_path = f'report_images/{sanitize_filename("_".join(columnas[i:i+2]))}_{area}.png'
        plt.savefig(image_path)
        image_paths.append(image_path)
        st.pyplot(fig)  # Mostrar figura explícitamente
        plt.close(fig)

    return image_paths

# Functions for extra visualizations

def graficar_distribucion_aire(df, tipo_grafico):
    image_path = "report_images/Air_Distribution_Combustion.png"
    limite_inferior = 0.14
    limite_superior = 0.28

    # Colores individuales para cada curva
    colores_individuales = {
        'Primario': 'yellow',
        'Secundario': 'green',
        'Secundario Alto': 'salmon',
        'Terciario': 'lightskyblue',
        'Cuaternario': 'black',
    }

    # Obtener el valor máximo de la variable de control
    max_val = df['Control APC Flujo aire a anillo cuaternario'].max()

    if tipo_grafico == 'Plotly Express':
        fig = go.Figure()
        variables = ['Primario', 'Secundario', 'Secundario Alto', 'Terciario', 'Cuaternario']
        for var in variables:
            # Color de la curva basado en si alcanza el máximo
            curve_color = [
                'blue' if value == max_val else colores_individuales[var]
                for value in df['Control APC Flujo aire a anillo cuaternario']
            ]
            fig.add_trace(go.Scatter(
                x=df['datetime'],
                y=df[var],
                mode='lines',
                name=var,
                line=dict(color=colores_individuales[var])  # Default color
            ))

        # Agregar límites con líneas horizontales
        fig.add_hline(y=limite_inferior, line_dash="dash", line_color="red", annotation_text="Límite Inferior")
        fig.add_hline(y=limite_superior, line_dash="dash", line_color="green", annotation_text="Límite Superior")

        fig.update_layout(
            title="Air Distribution [%]",
            xaxis_title="Fecha",
            yaxis_title="Valor",
            legend=dict(yanchor="bottom", y=0.01, xanchor="left", x=0.01, font=dict(size=10))
        )
        fig.write_image(image_path)
        return fig  # Return Plotly figure for further use

    else:  # Matplotlib
        fig, ax = plt.subplots(figsize=(14, 8))
        variables = ['Primario', 'Secundario', 'Secundario Alto', 'Terciario', 'Cuaternario']

        leyendas_agregadas = set()

        # Graficar las variables
        for var in variables:
            for i in range(len(df) - 1):
                x_segment = df['datetime'].iloc[i:i+2]
                y_segment = df[var].iloc[i:i+2]
                control_segment = df['Control APC Flujo aire a anillo cuaternario'].iloc[i:i+2]

                # Cambiar el color si la variable de control alcanza su valor máximo
                color = 'blue' if control_segment.max() == max_val else colores_individuales[var]

                # Etiqueta para la leyenda
                label = None
                if color == 'blue' and 'APC ON' not in leyendas_agregadas:
                    label = 'APC ON'
                    leyendas_agregadas.add('APC ON')
                elif color != 'blue' and var not in leyendas_agregadas:
                    label = var
                    leyendas_agregadas.add(var)

                ax.plot(x_segment, y_segment, color=color, linewidth=0.7, label=label)

        # Agregar límites con líneas horizontales
        ax.axhline(y=limite_inferior, color='red', linestyle='--', label='Límite Inferior')
        ax.axhline(y=limite_superior, color='green', linestyle='--', label='Límite Superior')

        ax.set_title("Air Distribution [%]", fontsize=15, fontweight='bold')
        ax.set_xlabel("Fecha", fontsize=15, fontweight='bold')
        ax.set_ylabel("Valor", fontsize=15, fontweight='bold')
        ax.legend(loc='upper left', fontsize=10)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(image_path)
        plt.close(fig)  # Close the figure to release memory
        return image_path


def graficar_diferencia_presion(df, tipo_grafico):
    image_path = "report_images/Pressure_Diff_Ensuciamiento.png"
    limite_inferior = 0.3
    limite_superior = 1.2

    if tipo_grafico == 'Plotly Express':
        fig = go.Figure()
        variables = ['Diff_Press_SC [kPa]', 'Diff_Press_BG [kPa]',
                     'Diff_Press_ECO1 [kPa]', 'Diff_Press_ECO2 [kPa]']
        for var in variables:
            fig.add_trace(go.Scatter(
                x=df['datetime'], y=df[var], mode='lines', name=var))
        
        # Agregar límites con líneas horizontales
        fig.add_hline(y=limite_inferior, line_dash="dash", line_color="red", annotation_text="Límite Inferior")
        fig.add_hline(y=limite_superior, line_dash="dash", line_color="green", annotation_text="Límite Superior")
        
        fig.update_layout(
            title="Pressure_Diff [kPa]",
            xaxis_title="Fecha",
            yaxis_title="Valor",
            legend=dict(yanchor="bottom", y=0.01, xanchor="left",
                        x=0.01, font=dict(size=10))
        )
        fig.write_image(image_path)
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig, ax = plt.subplots(figsize=(14, 8))
        variables = ['Diff_Press_SC [kPa]', 'Diff_Press_BG [kPa]',
                     'Diff_Press_ECO1 [kPa]', 'Diff_Press_ECO2 [kPa]']
        for var in variables:
            ax.plot(df['datetime'], df[var], label=var, linewidth=0.7)
        
        # Agregar límites con líneas horizontales
        ax.axhline(y=limite_inferior, color='red', linestyle='--', label='Límite Inferior')
        ax.axhline(y=limite_superior, color='green', linestyle='--', label='Límite Superior')
        
        ax.set_title("Pressure_Diff [kPa]", fontsize=15, fontweight='bold')
        ax.set_xlabel("Fecha", fontsize=15, fontweight='bold')
        ax.set_ylabel("Valor", fontsize=15, fontweight='bold')
        ax.legend(loc='upper left')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(image_path)
        st.pyplot(fig)  # Mostrar explícitamente la figura
        plt.close(fig)
    return image_path


# Continue similarly for the other functions like `graficar_distribucion_heat_coef`, `graficar_comparacion_licor_verde`, etc.

def graficar_distribucion_heat_coef(df, tipo_grafico):
    image_path = "report_images/Heat_Coefficient_Distribution_Ensuciamiento.png"
    variables = ['heat_coef_SH1 [kJ/m2C]', 'heat_coef_SH2 [kJ/m2C]', 'heat_coef_SH3 [kJ/m2C]']

    if tipo_grafico == 'Plotly Express':
        fig = go.Figure()
        for var in variables:
            if var in df.columns:
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
        # Verificar si las columnas existen antes de graficar
        valid_columns = [var for var in variables if var in df.columns]
        if not valid_columns:
            st.warning("No se encontraron columnas válidas para graficar.")
            return None

        fig, ax = plt.subplots(figsize=(14, 8))
        for var in valid_columns:
            ax.plot(df['datetime'], df[var], label=var, linewidth=0.7)
        ax.set_title("Heat Coefficient Distribution [kJ/m2C]", fontsize=15, fontweight='bold')
        ax.set_xlabel("Fecha", fontsize=15, fontweight='bold')
        ax.set_ylabel("Valor", fontsize=15, fontweight='bold')
        ax.legend(loc='upper left')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(image_path)
        st.pyplot(fig)  # Mostrar el gráfico
        plt.close(fig)  # Cerrar la figura
    return image_path


def graficar_comparacion_licor_verde(df, tipo_grafico):
    figs_paths = []
    comparisons = [
        ('reduction_lab [%]', 'reduction_ins [%]', 'Reduction Comparison [%]', 88, 98),
        ('alcali_lv_lab [g/L]', 'alcali_lv_ins [g/L]', 'Alcali Comparison [g/L]', 152, 178),
        ('sulfidez_lab [%]', 'sulfidez_ins [%]', 'Sulfidez Comparison [%]', 25, 35)
    ]

    for lab_var, inst_var, title, limite_inferior, limite_superior in comparisons:
        # Verificar si ambas columnas existen
        if not (column_exists(df, lab_var) and column_exists(df, inst_var)):
            print(f"Warning: One or both columns '{lab_var}' or '{inst_var}' do not exist in the DataFrame. Skipping plot '{title}'.")
            continue

        image_path = f"report_images/{sanitize_filename(title)}_Licor_Verde.png"
        if tipo_grafico == 'Plotly Express':
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df['datetime'], y=df[lab_var], mode='lines', name=lab_var))
            fig.add_trace(go.Scatter(
                x=df['datetime'], y=df[inst_var], mode='lines', name=inst_var))
            fig.add_hline(y=limite_inferior, line_dash="dash", line_color="red", annotation_text="Límite Inferior")
            fig.add_hline(y=limite_superior, line_dash="dash", line_color="green", annotation_text="Límite Superior")
            fig.update_layout(
                title=title,
                xaxis_title="Fecha",
                yaxis_title="Valor",
                legend=dict(yanchor="bottom", y=0.01, xanchor="left", x=0.01, font=dict(size=10))
            )
            fig.write_image(image_path)
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig, ax = plt.subplots(figsize=(14, 8))
            ax.plot(df['datetime'], df[lab_var], label=lab_var, linewidth=0.7)
            ax.plot(df['datetime'], df[inst_var], label=inst_var, linewidth=0.7)
            # Añadir líneas horizontales para los límites
            ax.axhline(y=limite_inferior, color='red', linestyle='--', label='Límite Inferior')
            ax.axhline(y=limite_superior, color='green', linestyle='--', label='Límite Superior')
            ax.set_title(title, fontsize=15, fontweight='bold')
            ax.set_xlabel("Fecha", fontsize=15, fontweight='bold')
            ax.set_ylabel("Valor", fontsize=15, fontweight='bold')
            ax.legend(loc='upper left')
            plt.xticks(rotation=45)
            plt.tight_layout()
            fig.savefig(image_path)
            st.pyplot(fig)  # Pasar la figura explícitamente
            plt.close(fig)

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
        fig, ax = plt.subplots(figsize=(14, 8))  # Asignar a `fig` y `ax`
        variables = ['O2_cont_left [%]', 'O2_cont_center [%]', 'O2_cont_right [%]']
        for var in variables:
            ax.plot(df['datetime'], df[var], label=var, linewidth=0.7)
        ax.set_title("O2 Content [%]", fontsize=15, fontweight='bold')
        ax.set_xlabel("Fecha", fontsize=15, fontweight='bold')
        ax.set_ylabel("Valor", fontsize=15, fontweight='bold')
        ax.legend(loc='upper left')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(image_path)
        st.pyplot(fig)  # Pasar `fig` explícitamente
        plt.close(fig)  # Cerrar correctamente el objeto `fig`
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
        fig, ax = plt.subplots(figsize=(14, 8))  # Asignar a `fig` y `ax`
        variables = ['CO_cont_left_wall [%]', 'CO_cont_center [%]', 'CO_cont_right_wall [%]']
        for var in variables:
            ax.plot(df['datetime'], df[var], label=var, linewidth=0.7)
        ax.set_title("CO Content [%]", fontsize=15, fontweight='bold')
        ax.set_xlabel("Fecha", fontsize=15, fontweight='bold')
        ax.set_ylabel("Valor", fontsize=15, fontweight='bold')
        ax.legend(loc='upper left')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(image_path)
        st.pyplot(fig)  # Pasar `fig` explícitamente
        plt.close(fig)  # Cerrar correctamente el objeto `fig`
    return image_path


# Function to plot with Plotly (including moving averages and limits)


def graficar_con_plotly(df, columnas, limites, area="General"):
    image_paths = []
    df['datetime'] = pd.to_datetime(df['datetime'])
    df_hora = df.set_index('datetime').resample('h').mean().reset_index()

    if not os.path.exists('report_images'):
        os.makedirs('report_images')

    columnas = [col for col in columnas if col not in omit_individual_plots.get(area, [])]

    for columna in columnas:
        if columna not in df_hora.columns:
            st.warning(f"Column '{columna}' is missing in the data.")
            continue

        fig = px.line(df_hora, x='datetime', y=columna, title=columna)

        limite_inferior = limites.get(columna, {}).get('limite_inferior')
        limite_superior = limites.get(columna, {}).get('limite_superior')

        if limite_inferior is not None:
            fig.add_hline(y=limite_inferior, line_dash="dash", line_color="red", annotation_text="Límite Inferior")
        if limite_superior is not None:
            fig.add_hline(y=limite_superior, line_dash="dash", line_color="green", annotation_text="Límite Superior")

        image_path = f'report_images/{sanitize_filename(columna)}_{area}.png'
        fig.write_image(image_path)
        image_paths.append(image_path)
        st.plotly_chart(fig, use_container_width=True)

    return image_paths

# Function to generate the HTML and PDF report


def generar_reporte_html_y_pdf(imagenes_por_area):
    logo_path = "docs/LOGO_ARAUCO.jpg"
    html_content = f"""
    <html>
        <head>
            <title>Reporte de Proceso</title>
            <style>
                /* Style for the header container */
                .header-container {{
                    display: flex;
                    align-items: center;
                    padding: 10px;
                }}
                /* Logo on the top left */
                .logo-container {{
                    background-color: white;
                    padding: 5px;
                    margin-right: auto;
                }}
                /* Center the title */
                h1 {{
                    text-align: center;
                    font-size: 21px;
                    font-weight: bold;
                    margin: 0;
                    flex-grow: 1;
                }}
                /* Style for area titles */
                h2 {{
                    text-align: left;
                    font-size: 18px;
                    font-weight: normal;
                    margin-left: 10px;
                    margin-top: 20px;
                }}
            </style>
        </head>
        <body>
            <div class="header-container">
                <div class="logo-container">
                    <img src="{logo_path}" alt="Company Logo" width="150">
                </div>
                <h1>Reporte APC CR3</h1>
            </div>
    """

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

# Function to add a background image

def add_bg_from_local(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background: linear-gradient(rgba(255, 255, 255, 0.5), rgba(255, 255, 255, 0.5)), 
                        url(data:image/png;base64,{encoded_string});
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


# Set the background image
add_bg_from_local("docs/MAPA_L3.jpg")


# Streamlit App


# Logo de la Empresa
logo_url = "https://gestal.usm.cl/wp-content/uploads/2024/09/LOGO-ARAUCO.png"

# Mostrar logo
st.markdown(
    f"""
    <div style="text-align: center;">
        <img src="{logo_url}" width="300">
    </div>
    """,
    unsafe_allow_html=True
)

# Título de la aplicación
st.title("Reporte Automatizado de Procesos")

archivo_csv = "data_caldera_2.csv"
if os.path.exists(archivo_csv):
    df = pd.read_csv(archivo_csv)
    df['datetime'] = pd.to_datetime(df['datetime'])

    # Selección de fechas
    fecha_inicio, fecha_fin = st.date_input(
        "Seleccionar fechas de inicio y fin",
        value=(df['datetime'].min().date(), df['datetime'].max().date())
    )
    fecha_inicio, fecha_fin = pd.to_datetime(fecha_inicio), pd.to_datetime(fecha_fin)
    df_filtrado = df[(df['datetime'] >= fecha_inicio) & (df['datetime'] <= fecha_fin)]

    # Selección del tipo de informe y gráficos
    tipo_reporte = st.radio("Seleccionar clase de informe", ('Por Area de Proceso', 'General'))
    tipo_grafico = st.radio("Seleccionar tipo de visualización", ('Matplotlib/Seaborn', 'Plotly Express'))

    if tipo_reporte == 'Por Area de Proceso':
        area_seleccionada = st.selectbox("Seleccionar un área de proceso", list(areas_de_proceso.keys()))
        columnas_seleccionadas = areas_de_proceso[area_seleccionada]

        if st.button("Generar informe"):
            limites_calculados = calcular_limites(df_filtrado, columnas_seleccionadas, area=area_seleccionada)

            if tipo_grafico == 'Matplotlib/Seaborn':
                imagenes = graficar_con_seaborn(df_filtrado, columnas_seleccionadas, limites_calculados, area_seleccionada)
            else:
                imagenes = graficar_con_plotly(df_filtrado, columnas_seleccionadas, limites_calculados, area_seleccionada)

            # Agregar gráficos específicos según el área seleccionada
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

            # Generar informe
            imagenes_por_area = {area_seleccionada: imagenes}
            generar_reporte_html_y_pdf(imagenes_por_area)

    elif tipo_reporte == 'General':
        columnas_seleccionadas = [col for cols in areas_de_proceso.values() for col in cols]

        if st.button("Generar informe"):
            imagenes = []
            for area, columnas in areas_de_proceso.items():
                limites_calculados = calcular_limites(df_filtrado, columnas, area=area)

                columnas_no_agrupadas = [
                    col for col in columnas if col not in omit_individual_plots.get(area, [])
                ]
                if tipo_grafico == 'Matplotlib/Seaborn':
                    imagenes.extend(
                        graficar_con_seaborn(df_filtrado, columnas_no_agrupadas, limites_calculados, area)
                    )
                else:
                    imagenes.extend(
                        graficar_con_plotly(df_filtrado, columnas_no_agrupadas, limites_calculados, area)
                    )

                # Agregar gráficos específicos por área
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

            # Crear informe general con gráficos por área
            imagenes_por_area = {}
            for area, columnas in areas_de_proceso.items():
                imagenes_area = [
                    img for img in imagenes if any(col in img for col in columnas)
                ]
                imagenes_por_area[area] = imagenes_area

            generar_reporte_html_y_pdf(imagenes_por_area)
else:
    st.error(f"El archivo {archivo_csv} no se encuentra en la carpeta.")
    
