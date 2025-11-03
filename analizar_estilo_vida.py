import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os

print("\nIniciando el script de Análisis de Estilo de Vida...")

# Directorio para guardar los gráficos
output_dir_graficos = '../resultados/'
# Directorio para guardar el CSV limpio
output_dir_datos = '../data/02_limpios/'

os.makedirs(output_dir_graficos, exist_ok=True)
os.makedirs(output_dir_datos, exist_ok=True)
print(f"Los gráficos se guardarán en: '{output_dir_graficos}'")
print(f"Los datos limpios se guardarán en: '{output_dir_datos}'")

# --- LIMPIEZA Y PREPARACIÓN DE DATOS ---

file_path = '../data/01_crudos/encuesta_estilo_vida.csv'
try:
    df_raw = pd.read_csv(file_path, encoding='utf-8-sig') 
    print("Archivo CSV original cargado exitosamente.")
except FileNotFoundError:
    print(f"Error: No se encontró el archivo '{file_path}'.")
    print("Asegúrate de tener la estructura de carpetas: data/01_crudos/encuesta_estilo_vida.csv")
    exit()

# Renombramos las columnas
column_names = [
    'Timestamp', 'Numero_Cuenta', 'Dias_Ejercicio', 'Actividad_Recreativa',
    'Comidas_Dia', 'Toma_Alcohol', 'Horas_Sueño', 'Area_Trabajo', 'Edad',
    'Sexo', 'Nivel_Ansiedad', 'Vivienda', 'Tiene_Beca', 'Horas_Pantalla',
    'Ingresos_Mensuales', 'Siente_Energia', 'Promedio_Escolar', 'Es_Regular',
    'Ayuda_Psicologica', 'Tiene_Pareja', 'Alguien_Depende_Economicamente'
]
df_raw.columns = column_names

# --- Funciones de Limpieza de Datos ---

def limpiar_texto_general(texto):
    if isinstance(texto, str):
        texto = texto.replace('\xa0', ' ')  
        texto = texto.replace('\t', ' ') 
        return texto.strip().lower()
    if pd.isna(texto): 
        return None
    return texto

def limpiar_ansiedad(texto):
    texto = limpiar_texto_general(texto)
    if not texto or pd.isna(texto):
        return "No especificado"
    kw_grave = ['grave']
    if any(palabra in texto for palabra in kw_grave):
        return 'Grave'
    kw_moderada = ['moderada', 'social', 'moderado']
    if any(palabra in texto for palabra in kw_moderada):
        return 'Moderada'
    kw_leve = ['leve']
    if any(palabra in texto for palabra in kw_leve):
        return 'Leve'
    kw_ninguno = ['ninguno', 'ninguna']
    if any(palabra in texto for palabra in kw_ninguno):
        return 'Ninguno'
    return 'No especificado'

def limpiar_sexo(texto):
    texto = limpiar_texto_general(texto)
    if not texto: return "No especificado"
    if texto.startswith('f'): return 'Femenino'
    if texto.startswith('m') or texto.startswith('h'): return 'Masculino'
    return 'No especificado'

def limpiar_si_no_robusto(texto):
    texto = limpiar_texto_general(texto)
    if not texto or pd.isna(texto):
        return "No especificado"
    kw_si = ['si', 'sí', 'sip']
    if any(palabra in texto for palabra in kw_si):
        return 'Sí'
    kw_no = ['no', 'nop', 'ninguna']
    if any(palabra in texto for palabra in kw_no):
        return 'No'
    if 'mas o menos' in texto:
        return 'A veces' 
    return 'No especificado'

# --- Aplicamos Limpieza y Guardar CSV ---

datos_limpios = {
    'Numero_Cuenta': pd.to_numeric(df_raw['Numero_Cuenta'], errors='coerce'),
    'Sexo': df_raw['Sexo'].apply(limpiar_sexo),
    'Nivel_Ansiedad': df_raw['Nivel_Ansiedad'].apply(limpiar_ansiedad),
    'Tiene_Beca': df_raw['Tiene_Beca'].apply(limpiar_si_no_robusto),
    'Siente_Energia': df_raw['Siente_Energia'].apply(limpiar_si_no_robusto),
    'Horas_Sueño': pd.to_numeric(df_raw['Horas_Sueño'], errors='coerce'),
    'Promedio_Escolar': pd.to_numeric(df_raw['Promedio_Escolar'], errors='coerce').round(1)
}
df = pd.DataFrame(datos_limpios)
print("Datos limpiados y estandarizados.")

print("\n--- Visualización del DataFrame Limpio (primeras filas) ---")
print(df.head())

ruta_limpia = os.path.join(output_dir_datos, 'estilo_vida_limpio.csv')
df.to_csv(ruta_limpia, index=False)
print(f"\nDataFrame limpio guardado en: '{ruta_limpia}'")

# --- Generación de Gráficos ---

sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'sans-serif'

# --- Gráfico Nivel de Ansiedad ---
plt.figure(figsize=(10, 7))
ansiedad_orden = ['Ninguno', 'Leve', 'Moderada', 'Grave']
ax1 = sns.countplot(y=df['Nivel_Ansiedad'], 
                    order=ansiedad_orden, 
                    palette='coolwarm',
                    hue=df['Nivel_Ansiedad'], 
                    legend=False)             
plt.title('Distribución del Nivel de Ansiedad en Estudiantes', fontsize=12, fontweight='bold')
plt.xlabel('Cantidad de Estudiantes', fontsize=12)
plt.ylabel('Nivel de Ansiedad Reportado', fontsize=12)
for p in ax1.patches:
    if p.get_width() > 0:
        ax1.annotate(f'{int(p.get_width())}', (p.get_width(), p.get_y() + p.get_height() / 2.),
                     ha='left', va='center', xytext=(5, 0), textcoords='offset points')
plt.tight_layout()
plt.savefig(os.path.join(output_dir_graficos, 'estilo_vida_ansiedad.png')) # <-- CORREGIDO
plt.close()

# --- Gráfico Distribución por Sexo ---
plt.figure(figsize=(10, 10))
sexo_counts = df['Sexo'].value_counts()
def mostrar_porcentaje(pct):
    total = sexo_counts.sum()
    valor = int(round(pct/100 * total))
    return f'{pct:.1f}%\n({valor:d})'
plt.pie(sexo_counts, 
    labels=sexo_counts.index, 
    autopct=mostrar_porcentaje, 
    startangle=140, 
    colors=['#66b3ff', '#ff9999'])
plt.title('Distribución de Estudiantes por Sexo', fontsize=12, fontweight='bold')
plt.ylabel('')
plt.savefig(os.path.join(output_dir_graficos, 'estilo_vida_sexo.png')) 
plt.close()

# --- Gráfico Horas de Sueño ---
plt.figure(figsize=(10, 7))
ax3 = sns.countplot(x=df['Horas_Sueño'].round(), 
                    palette='coolwarm',
                    hue=df['Horas_Sueño'].round(), 
                    legend=False)                 
plt.title('Horas de Sueño Promedio por Noche', fontsize=12, fontweight='bold')
plt.xlabel('Horas de Sueño', fontsize=12)
plt.ylabel('Cantidad de Estudiantes', fontsize=12)
for p in ax3.patches:
    if p.get_height() > 0:
        ax3.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='center', xytext=(0, 10), textcoords='offset points')
plt.tight_layout()
plt.savefig(os.path.join(output_dir_graficos, 'estilo_vida_sueno.png')) # <-- CORREGIDO
plt.close()

# --- Gráfico Estudiantes con Beca ---
plt.figure(figsize=(10, 10))
beca_counts = df['Tiene_Beca'].value_counts()
total_beca = beca_counts.sum() 

def mostrar_porcentaje_y_valor(pct):
    valor = int(round(pct/100 * total_beca))
    return f'{pct:.1f}%\n({valor:d})'

plt.pie(beca_counts, 
        labels=beca_counts.index, 
        autopct=mostrar_porcentaje_y_valor, 
        startangle=90, 
        colors=['#c2c2f0','#ffb3e6'])

plt.title('Proporción de Estudiantes con Beca', fontsize=12, fontweight='bold')
plt.ylabel('')
plt.savefig(os.path.join(output_dir_graficos, 'estilo_vida_beca.png'))
plt.close()

# --- Gráfico Relación entre Ansiedad y Promedio Escolar ---
plt.figure(figsize=(12, 8))
sns.boxplot(x='Nivel_Ansiedad', 
            y='Promedio_Escolar', 
            data=df, 
            order=ansiedad_orden, 
            palette='coolwarm',
            hue='Nivel_Ansiedad', 
            dodge=False,        
            legend=False)    
plt.title('Promedio Escolar vs. Nivel de Ansiedad', fontsize=12, fontweight='bold')
plt.xlabel('Nivel de Ansiedad Reportado', fontsize=12)
plt.ylabel('Promedio Escolar', fontsize=12)
plt.ylim(5, 10)
plt.tight_layout()
plt.savefig(os.path.join(output_dir_graficos, 'estilo_vida_ansiedad_vs_promedio.png')) 
plt.close()

print("\nAnálisis finalizado con éxito.")
print(f"Revisa la carpeta '{output_dir_graficos}' para ver gráficos generados.")
