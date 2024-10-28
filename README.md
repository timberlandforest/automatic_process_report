
# APP para Reporte Automático de Procesos Productivos

Esta aplicación ha sido desarrollada con Streamlit y permite generar reportes automatizados de análisis de procesos productivos, visualizando gráficos y exportando el informe en formato PDF. Puedes acceder a la última versión en [https://automaticprocessreportv2.streamlit.app/](https://automaticprocessreportv2.streamlit.app/).

## Funcionalidades Principales

1. **Selección de Áreas de Proceso**: La aplicación permite elegir entre diferentes áreas de proceso (Combustión, Vapor, Ensuciamiento, Licor Verde, y Emisiones) y muestra las variables relevantes para cada una.

2. **Límites de Visualización**: Calcula y aplica límites de percentil para cada variable, con líneas horizontales de límites inferior y superior en los gráficos.

3. **Filtrado por Fecha**: Los usuarios pueden seleccionar un rango de fechas específico para analizar el comportamiento de las variables dentro de ese período.

4. **Exportación de Reportes**: Genera un archivo HTML con las visualizaciones, que luego se convierte a PDF, permitiendo la descarga del reporte.

## Estructura del Código

- **Función `calcular_limites`**: Calcula límites basados en percentiles para cada columna numérica.
- **Función `graficar_media_por_hora_con_limites`**: Genera gráficos de línea por hora con medias móviles, aplicando los límites calculados.
- **Función `generar_reporte_html_y_pdf`**: Genera un reporte HTML con todas las visualizaciones y convierte el archivo a PDF.

## Requisitos

Asegúrate de tener instaladas las siguientes bibliotecas antes de ejecutar la aplicación:

- pandas
- seaborn
- matplotlib
- streamlit
- xhtml2pdf
- wkhtmltopdf (especificar la ruta si es necesario)

Instálalas con:
```bash
pip install pandas seaborn matplotlib streamlit xhtml2pdf
```

## Ejecución

1. Coloca el archivo `data_caldera.csv` en el mismo directorio que el script.
2. Ejecuta la aplicación en tu entorno local:
   ```bash
   streamlit run app.py
   ```
3. En la aplicación, selecciona el área de proceso, variables y rango de fechas, y luego genera el reporte.

## Solución de Problemas

- Si encuentras errores al exportar a PDF, asegúrate de que `wkhtmltopdf` esté instalado correctamente y accesible desde la ruta especificada.
