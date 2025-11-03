import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os

print("\nIniciando el script de Análisis de Economía Estudiantil")

# Directorio para guardar los gráficos
output_dir_graficos = '../resultados/'
# Directorio para guardar el CSV limpio
output_dir_datos = '../data/02_limpios/'

os.makedirs(output_dir_graficos, exist_ok=True)
os.makedirs(output_dir_datos, exist_ok=True)
print(f"Los gráficos se guardarán en: '{output_dir_graficos}'")
print(f"Los datos limpios se guardarán en: '{output_dir_datos}'")

# --- LIMPIEZA Y PREPARACIÓN DE DATOS ---

file_path = '../data/01_crudos/encuesta_economia.csv'
try:
    df_raw = pd.read_csv(file_path, encoding='utf-8-sig')
    print("Archivo CSV original cargado exitosamente.")
except FileNotFoundError:
    print(f"Error: No se encontró el archivo '{file_path}'.")
    print("Asegúrate de tener la estructura de carpetas: data/01_crudos/encuesta_economia.csv")
    exit()

# Renombramos las columnas para que sean más fáciles de manejar
column_names = [
    'Timestamp', 'Numero_Cuenta', 'Semestre', 'Carrera',
    'Situacion_Economica', 'Fuente_Ingresos', 'Gasto_Dificil',
    'Impacto_Economico', 'Equilibrio_Trabajo_Estudio', 'Renuncia_Oportunidad',
    'Sentimiento_Finanzas', 'Estrategias_Dinero', 'Apoyo_Economico_Deseado',
    'Otro_Tipo_Ayuda', 'Utilidad_Educacion_Financiera',
    'Situacion_Economica_Ideal_5_Anios', 'Momento_Mayor_Presion',
    'Accion_Gasto_Inesperado', 'Recibio_Orientacion_Financiera', 'Consejo_Estudiantes'
]
df_raw.columns = column_names

# --- Funciones de Limpieza de Datos ---

def limpiar_situacion_economica(texto):
    if not isinstance(texto, str): return "No especificado"
    texto = texto.lower()
    if 'buena' in texto: return 'Buena'
    if 'estable' in texto: return 'Estable'
    if 'regular' in texto: return 'Regular'
    if 'complicada' in texto: return 'Complicada'
    if 'mala' in texto: return 'Mala'
    return 'Otra'

def limpiar_sentimientos(texto):
    if not isinstance(texto, str): return "No especificado"
    texto = texto.lower()
    if any(palabra in texto for palabra in ['ansiedad', 'preocupación', 'preocupacion']):
        return 'Ansiedad/Preocupación'
    if 'tranquilidad' in texto: return 'Tranquilidad'
    if 'indiferencia' in texto or 'indiferente' in texto: 
        return 'Indiferencia'
    if 'un poco de todo' in texto:
        return 'Mixto'
    if 'yo no tengo finanzas' in texto:
        return 'Indiferencia'
    return 'Otro'

def limpiar_gasto_dificil(texto):
    if not isinstance(texto, str): return "No especificado"
    texto = texto.lower()
    if 'transporte' in texto or 'pasajes' in texto: return 'Transporte'
    if 'comida' in texto or 'lunch' in texto or 'despensa' in texto: return 'Comida/Alimentos'
    if 'renta' in texto: return 'Renta'
    if 'ninguno' in texto: return 'Ninguno'
    return 'Otros'

def limpiar_renuncia_oportunidad(texto):
    if not isinstance(texto, str): return "No especificado"
    texto = texto.lower().strip()
    if texto.startswith('si') or texto.startswith('sí'): return 'Sí'
    if texto.startswith('no') or texto.startswith('ninguna') or texto.startswith('ninguno') or 'creo que no' in texto:
        return 'No'
    # Si no es 'Sí' o 'No', lo dejamos como No especificado para revisión
    return 'No especificado'

def limpiar_impacto_academico(texto):
    if not isinstance(texto, str): return "No especificado"
    texto = texto.lower()
    palabras_alto = [
        'alto', 'mucho', 'bastante', 'muy', 'buen', 
        'afecta', 'importante', 'todos los aspectos', 
        'hambre', 'desconcentro', 'cansado'
    ]
    if any(palabra in texto for palabra in palabras_alto): 
        return 'Alto'
    if 'medio' in texto or 'media' in texto: 
        return 'Medio'
    if 'poco' in texto:
        return 'Bajo'
    palabras_ninguno = ['ninguno', 'no tiene', 'ninguna', 'bien,', 'no tendria que ver']
    if any(palabra in texto for palabra in palabras_ninguno):
        return 'Ninguno'
    return 'No especificado'

# --- Aplicamos Limpieza y Guardar CSV ---

datos_limpios = {
    'Numero_Cuenta': pd.to_numeric(df_raw['Numero_Cuenta'], errors='coerce'),
    'Situacion_Economica': df_raw['Situacion_Economica'].apply(limpiar_situacion_economica),
    'Sentimiento_Financiero': df_raw['Sentimiento_Finanzas'].apply(limpiar_sentimientos),
    'Gasto_Principal': df_raw['Gasto_Dificil'].apply(limpiar_gasto_dificil),
    'Renuncia_Oportunidad': df_raw['Renuncia_Oportunidad'].apply(limpiar_renuncia_oportunidad),
    'Impacto_Academico': df_raw['Impacto_Economico'].apply(limpiar_impacto_academico)
}
df = pd.DataFrame(datos_limpios)
print("Datos limpiados y estandarizados.")

print("\n--- Visualización del DataFrame Limpio (primeras filas) ---")
print(df.head())

# Guardamos el CSV limpio en la carpeta data/02_limpios/
ruta_limpia = os.path.join(output_dir_datos, 'economia_limpio.csv')
df.to_csv(ruta_limpia, index=False)
print(f"\nDataFrame limpio guardado en: '{ruta_limpia}'")

# --- Generación de Gráficos ---

sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'sans-serif'

# --- Gráfico Situación Económica ---
plt.figure(figsize=(10, 7))
ax1 = sns.countplot(y=df['Situacion_Economica'], 
                    order=df['Situacion_Economica'].value_counts().index, 
                    palette='coolwarm', 
                    hue=df['Situacion_Economica'], 
                    legend=False)
plt.title('Distribución de la Situación Económica', fontsize=16, fontweight='bold')
plt.xlabel('Cantidad de Estudiantes', fontsize=12)
plt.ylabel('Situación Percibida', fontsize=12)
for p in ax1.patches:
    if p.get_width() > 0:
        ax1.annotate(f'{int(p.get_width())}', 
                     (p.get_width(), p.get_y() + p.get_height() / 2.), 
                     ha='left', va='center', 
                     xytext=(5, 0), textcoords='offset points')
plt.tight_layout()
plt.savefig(os.path.join(output_dir_graficos, 'economia_situacion.png'))
plt.close()

# --- Gráfico Sentimiento Financiero ---
plt.figure(figsize=(10, 10))
sentimiento_counts = df['Sentimiento_Financiero'].value_counts()
total_counts = sentimiento_counts.sum()

def mostrar_porcentaje_y_valor(pct):
    valor = int(round(pct/100 * total_counts))
    return f'{pct:.1f}%\n({valor:d})'

plt.pie(sentimiento_counts, 
        labels=sentimiento_counts.index, 
        autopct=mostrar_porcentaje_y_valor,
        startangle=140, 
        colors=sns.color_palette('coolwarm'))
plt.title('Sentimientos Generados por las Finanzas', fontsize=16, fontweight='bold')
plt.ylabel('')
plt.savefig(os.path.join(output_dir_graficos, 'economia_sentimiento.png'))
plt.close()

# --- Gráfico Gasto Principal ---
plt.figure(figsize=(10, 7))
ax3 = sns.countplot(x=df['Gasto_Principal'], 
                    order=df['Gasto_Principal'].value_counts().index, 
                    palette='coolwarm', 
                    hue=df['Gasto_Principal'], 
                    legend=False)
plt.title('Gastos Mensuales Más Difíciles de Cubrir', fontsize=16, fontweight='bold')
plt.xlabel('Tipo de Gasto', fontsize=12)
plt.ylabel('Cantidad de Estudiantes', fontsize=12)
for p in ax3.patches:
    if p.get_height() > 0:
        ax3.annotate(f'{int(p.get_height())}', 
                     (p.get_x() + p.get_width() / 2., p.get_height()), 
                     ha='center', va='center', 
                     xytext=(0, 10), textcoords='offset points')
plt.tight_layout()
plt.savefig(os.path.join(output_dir_graficos, 'economia_gasto.png'))
plt.close()

# --- Gráfico Renuncia a Oportunidades ---
plt.figure(figsize=(10, 10))
renuncia_counts = df['Renuncia_Oportunidad'].value_counts()
total_renuncia = renuncia_counts.sum()

def mostrar_porcentaje_y_valor_renuncia(pct):
    valor = int(round(pct/100 * total_renuncia))
    return f'{pct:.1f}%\n({valor:d})'

plt.pie(renuncia_counts, 
    labels=renuncia_counts.index, 
    autopct=mostrar_porcentaje_y_valor_renuncia, 
    startangle=90, 
    colors=['#ff9999','#66b3ff']) # Colores personalizados
plt.title('Renuncia a Oportunidades por Motivos Económicos', fontsize=16, fontweight='bold')
plt.ylabel('')
plt.savefig(os.path.join(output_dir_graficos, 'economia_renuncia.png'))
plt.close()

# --- Gráfico Impacto Académico ---
plt.figure(figsize=(10, 7))
ax5 = sns.countplot(x=df['Impacto_Academico'], 
                    order=df['Impacto_Academico'].value_counts().index, 
                    palette='coolwarm', 
                    hue=df['Impacto_Academico'], 
                    legend=False)
plt.title('Impacto Económico en el Desempeño Académico', fontsize=16, fontweight='bold')
plt.xlabel('Nivel de Impacto Percibido', fontsize=12)
plt.ylabel('Cantidad de Estudiantes', fontsize=12)
for p in ax5.patches:
    if p.get_height() > 0:
        ax5.annotate(f'{int(p.get_height())}', 
                     (p.get_x() + p.get_width() / 2., p.get_height()), 
                     ha='center', va='center', 
                     xytext=(0, 10), textcoords='offset points')
plt.tight_layout()
plt.savefig(os.path.join(output_dir_graficos, 'economia_impacto.png'))
plt.close()

print("\nAnálisis finalizado con éxito.")
print(f"Revisa la carpeta '{output_dir_graficos}' para ver gráficos generados.")