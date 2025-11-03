import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os

# ---  DEFINICIÓN DE RUTAS ---
output_dir_graficos = '../resultados/'
output_dir_datos = '../data/02_limpios/'

os.makedirs(output_dir_graficos, exist_ok=True)
os.makedirs(output_dir_datos, exist_ok=True)
print(f"Los gráficos se guardarán en: '{output_dir_graficos}'")

# ---  CARGA DE DATOS LIMPIOS ---
path_bienestar = '../data/02_limpios/estilo_vida_limpio.csv'
path_economia = '../data/02_limpios/economia_limpio.csv'

try:
    df_bienestar = pd.read_csv(path_bienestar)
    df_economia = pd.read_csv(path_economia)
    print("Archivos CSV limpios cargados exitosamente.")
except FileNotFoundError:
    print(f"Error: No se encontraron los archivos limpios en '{output_dir_datos}'.")
    print("Asegúrate de ejecutar primero los scripts de análisis individuales.")
    exit()

# --- UNIÓN Y CÁLCULO DE KPIS ---
df_completo = pd.merge(df_bienestar, df_economia, on='Numero_Cuenta', how='inner')
print(f"Unión exitosa. Se encontraron {len(df_completo)} estudiantes en ambas encuestas.")

# Guardamos el CSV limpio en la carpeta data/02_limpios/
ruta_limpia = os.path.join(output_dir_datos, 'combinado_limpio.csv')
df_completo.to_csv(ruta_limpia, index=False)
print(f"\nDataFrame limpio guardado en: '{ruta_limpia}'")

# --- CONFIGURACIÓN DE GRÁFICOS ---
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'sans-serif'

# --- GRÁFICO 1: Heatmap ---
contingency_table = pd.crosstab(df_completo['Situacion_Economica'], df_completo['Nivel_Ansiedad'])
ansiedad_orden = ['Ninguno', 'Leve', 'Moderada', 'Grave']
situacion_orden = ['Buena', 'Estable', 'Regular', 'Complicada/Mala']
contingency_table = contingency_table.reindex(columns=ansiedad_orden, index=situacion_orden, fill_value=0)
plt.figure(figsize=(12, 8))
sns.heatmap(contingency_table, annot=True, fmt='d', cmap='YlGnBu')
plt.title('Relación entre Situación Económica y Nivel de Ansiedad', fontsize=16, fontweight='bold')
plt.ylabel('Situación Económica', fontsize=12)
plt.xlabel('Nivel de Ansiedad', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(output_dir_graficos, 'combinado_heatmap_economia_vs_ansiedad.png'), dpi=150)
plt.close() 

# --- GRÁFICO 2: Promedio vs. Estrés Financiero ---
promedio_por_estres = df_completo.groupby('Sentimiento_Financiero')['Promedio_Escolar'].mean().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(12, 7))
sns.barplot(x=promedio_por_estres.index, 
            y=promedio_por_estres.values, 
            palette='viridis', 
            ax=ax,
            hue=promedio_por_estres.index, 
            legend=False)
ax.set_title('El Estrés Financiero se Correlaciona con un Menor Promedio', fontsize=16, fontweight='bold', pad=25)
fig.suptitle('Promedio escolar según el sentimiento generado por las finanzas personales', fontsize=12, color='gray')
ax.set_ylabel('Promedio Escolar (GPA)', fontsize=12, fontweight='bold')
ax.set_xlabel('')
ax.bar_label(ax.containers[0], fmt='%.2f', size=12, fontweight='bold')
sns.despine(left=True, bottom=True)
if not promedio_por_estres.empty:
    plt.ylim(promedio_por_estres.min() * 0.95, promedio_por_estres.max() * 1.02)
plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.figtext(0.99, 0.01, f'Fuente: Encuesta Estudiantil (N={len(df_completo)})', horizontalalignment='right', size=8, color='grey')
plt.savefig(os.path.join(output_dir_graficos, 'combinado_promedio_vs_estres.png'), dpi=150)
plt.close()

# --- GRÁFICO 3: Déficit de Energía por Gasto ---
df_sin_energia = df_completo[df_completo['Siente_Energia'] == 'No']
energia_por_gasto = df_sin_energia['Gasto_Principal'].value_counts().sort_values()
fig, ax = plt.subplots(figsize=(12, 7))
ax.barh(energia_por_gasto.index, energia_por_gasto.values, color=sns.color_palette('viridis', len(energia_por_gasto)))
ax.set_title('El Transporte es el Gasto que Más Drena la Energía', fontsize=16, fontweight='bold', pad=25)
fig.suptitle('Principal gasto de los estudiantes que reportan sentirse sin energía', fontsize=12, color='gray')
ax.set_xlabel('Cantidad de Estudiantes sin Energía', fontsize=12, fontweight='bold')
ax.set_ylabel('')
ax.bar_label(ax.containers[0], size=12, padding=5)
sns.despine(left=True, bottom=True)
plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.figtext(0.99, 0.01, f'Fuente: Encuesta Estudiantil (N={len(df_sin_energia)})', horizontalalignment='right', size=8, color='grey')
plt.savefig(os.path.join(output_dir_graficos, 'combinado_energia_vs_gasto.png'), dpi=150)
plt.close()

# --- GRÁFICO 4: Desglose de Ansiedad Exacerbada ---
df_ansiedad_financiera = df_completo[df_completo['Sentimiento_Financiero'] == 'Ansiedad/Preocupación']
ansiedad_counts = df_ansiedad_financiera['Nivel_Ansiedad'].value_counts().reindex(ansiedad_orden).fillna(0) 
fig, ax = plt.subplots(figsize=(12, 7))
ax.barh(ansiedad_counts.index, ansiedad_counts.values, color=sns.color_palette('viridis_r', len(ansiedad_counts)))
ax.set_title('Ansiedad Moderada a Grave Domina en Estudiantes con Estrés Financiero', fontsize=16, fontweight='bold', pad=25)
fig.suptitle('Desglose del nivel de ansiedad para el grupo con preocupación financiera', fontsize=12, color='gray')
ax.set_xlabel('Cantidad de Estudiantes', fontsize=12, fontweight='bold')
ax.set_ylabel('')
ax.bar_label(ax.containers[0], size=12, padding=5)
sns.despine(left=True, bottom=True)
plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.figtext(0.99, 0.01, f'Fuente: Encuesta Estudiantil (N={len(df_ansiedad_financiera)})', horizontalalignment='right', size=8, color='grey')
plt.savefig(os.path.join(output_dir_graficos, 'combinado_ansiedad_exacerbada.png'), dpi=150)
plt.close()

print("\nAnálisis completo y generación de imágenes finalizados.")
