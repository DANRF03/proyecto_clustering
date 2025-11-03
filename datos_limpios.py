import pandas as pd
import numpy as np
import re
from typing import Dict, List

class LimpiadorEncuestasMejorado:
    def __init__(self):
        self.economia_df = None
        self.estilo_vida_df = None
        
    def cargar_datos(self):
        """Cargar datos con manejo robusto de encoding"""
        try:
            # Intentar diferentes encodings comúnmente usados en Windows
            encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
            
            for encoding in encodings:
                try:
                    self.economia_df = pd.read_csv('encuesta_economia.csv', encoding=encoding)
                    self.estilo_vida_df = pd.read_csv('encuesta_estilo_vida.csv', encoding=encoding)
                    print(f"✅ Datos cargados con encoding: {encoding}")
                    break
                except UnicodeDecodeError:
                    continue
            else:
                print("❌ No se pudo cargar con ningún encoding común")
                return False
                
            print(f"📊 Economía: {len(self.economia_df)} registros")
            print(f"🏃 Estilo vida: {len(self.estilo_vida_df)} registros")
            return True
            
        except Exception as e:
            print(f"❌ Error al cargar datos: {e}")
            return False

    def limpiar_economia(self):
        """Limpieza robusta del dataset económico"""
        print("\n🧹 Limpiando datos económicos...")
        
        # 1. Limpiar nombres de columnas (más robusto)
        self.economia_df.columns = [str(col).strip().replace('"', '').replace('﻿', '') for col in self.economia_df.columns]
        
        # Mostrar columnas disponibles para debugging
        print("📝 Columnas disponibles:", list(self.economia_df.columns))
        
        # 2. Estandarizar carrera (todas son MAC)
        self.economia_df['Carrera_limpia'] = 'MAC'
        
        # 3. Limpiar semestres (más tolerante)
        def limpiar_semestre(semestre):
            if pd.isna(semestre):
                return None
            try:
                # Buscar cualquier número en el texto
                numeros = re.findall(r'\d+', str(semestre))
                if numeros:
                    semestre_num = int(numeros[0])
                    return min(max(semestre_num, 1), 9)  # Permitir hasta 9
                return None
            except:
                return None
        
        # Usar índice de columna si el nombre no funciona
        col_semestre = self.economia_df.columns[2]  # Tercera columna
        self.economia_df['Semestre_limpio'] = self.economia_df[col_semestre].apply(limpiar_semestre)
        
        # 4. Categorizar situación económica (mejorado)
        def categorizar_situacion(texto):
            if pd.isna(texto):
                return 'No especificado'
            
            texto = str(texto).lower().strip()
            
            # Mapeo más comprehensivo
            if any(palabra in texto for palabra in ['buena', 'bien', 'bein', 'excelente', 'suficiente']):
                return 'Buena'
            elif any(palabra in texto for palabra in ['regular', 'normal', 'media', 'aceptable', 'estable']):
                return 'Regular'
            elif any(palabra in texto for palabra in ['mala', 'complicada', 'deficiente', 'dependiendo']):
                return 'Mala'
            else:
                return 'Regular'  # Valor por defecto más seguro
        
        col_situacion = self.economia_df.columns[4]  # Quinta columna
        self.economia_df['Situacion_economica'] = self.economia_df[col_situacion].apply(categorizar_situacion)
        
        # 5. Fuentes de ingresos (mejorado)
        def estandarizar_ingresos(texto):
            if pd.isna(texto):
                return 'No especificado'
            
            texto = str(texto).lower()
            
            if 'familia' in texto and 'trabajo' in texto:
                return 'Familia y Trabajo'
            elif 'familia' in texto and 'beca' in texto:
                return 'Familia y Beca'
            elif 'trabajo' in texto:
                return 'Trabajo'
            elif 'familia' in texto:
                return 'Familia'
            elif 'beca' in texto:
                return 'Beca'
            else:
                return 'Otros'
        
        col_ingresos = self.economia_df.columns[5]  # Sexta columna
        self.economia_df['Fuente_ingresos'] = self.economia_df[col_ingresos].apply(estandarizar_ingresos)
        
        # 6. Sentimientos financieros (más comprehensivo)
        def categorizar_sentimientos(texto):
            if pd.isna(texto):
                return 'No especificado'
            
            texto = str(texto).lower()
            
            if 'ansiedad' in texto:
                return 'Ansiedad'
            elif 'preocupación' in texto or 'preocupacion' in texto:
                return 'Preocupación'
            elif 'tranquilidad' in texto or 'tranquilo' in texto:
                return 'Tranquilidad'
            elif 'indiferencia' in texto or 'indiferente' in texto:
                return 'Indiferencia'
            elif 'mixto' in texto or 'mezclado' in texto or 'un poco de todo' in texto:
                return 'Mixto'
            else:
                return 'Otros'
        
        col_sentimientos = self.economia_df.columns[10]  # Undécima columna
        self.economia_df['Sentimiento_financiero'] = self.economia_df[col_sentimientos].apply(categorizar_sentimientos)
        
        print("✅ Datos económicos limpiados correctamente")

    def limpiar_estilo_vida(self):
        """Limpieza robusta del dataset de estilo de vida"""
        print("\n🏃 Limpiando datos de estilo de vida...")
        
        # 1. Limpiar nombres de columnas
        self.estilo_vida_df.columns = [str(col).strip().replace('"', '').replace('﻿', '') for col in self.estilo_vida_df.columns]
        
        # 2. Días de ejercicio (más tolerante)
        def limpiar_ejercicio(texto):
            if pd.isna(texto):
                return 0
            numeros = re.findall(r'\d+', str(texto))
            return int(numeros[0]) if numeros else 0
        
        col_ejercicio = self.estilo_vida_df.columns[2]  # Tercera columna
        self.estilo_vida_df['Dias_ejercicio'] = self.estilo_vida_df[col_ejercicio].apply(limpiar_ejercicio)
        
        # 3. Actividad recreativa
        def categorizar_actividad(texto):
            if pd.isna(texto) or str(texto).lower() in ['no', 'ninguna', 'no realizo']:
                return 'No'
            else:
                return 'Sí'
        
        col_actividad = self.estilo_vida_df.columns[3]  # Cuarta columna
        self.estilo_vida_df['Actividad_recreativa'] = self.estilo_vida_df[col_actividad].apply(categorizar_actividad)
        
        # 4. Horas de sueño
        def limpiar_sueno(texto):
            if pd.isna(texto):
                return 6
            numeros = re.findall(r'\d+', str(texto))
            if numeros:
                horas = int(numeros[0])
                return min(max(horas, 2), 12)
            return 6
        
        col_sueno = self.estilo_vida_df.columns[6]  # Séptima columna
        self.estilo_vida_df['Horas_sueno'] = self.estilo_vida_df[col_sueno].apply(limpiar_sueno)
        
        # 5. Edad
        def limpiar_edad(texto):
            if pd.isna(texto):
                return None
            numeros = re.findall(r'\d+', str(texto))
            if numeros:
                edad = int(numeros[0])
                return min(max(edad, 17), 35)
            return None
        
        col_edad = self.estilo_vida_df.columns[8]  # Novena columna
        self.estilo_vida_df['Edad'] = self.estilo_vida_df[col_edad].apply(limpiar_edad)
        
        print("✅ Datos de estilo de vida limpiados correctamente")

    def crear_dataset_final(self):
        """Crear dataset final unificado"""
        print("\n🔗 Creando dataset final...")
        
        # Columnas limpias de economía
        economia_limpio = self.economia_df[[
            'Número de cuenta', 'Semestre_limpio', 'Carrera_limpia',
            'Situacion_economica', 'Fuente_ingresos', 'Sentimiento_financiero'
        ]].copy()
        
        # Columnas limpias de estilo de vida
        estilo_limpio = self.estilo_vida_df[[
            'Número de cuenta', 'Dias_ejercicio', 'Actividad_recreativa',
            'Horas_sueno', 'Edad'
        ]].copy()
        
        # Unir datasets
        dataset_final = pd.merge(economia_limpio, estilo_limpio, on='Número de cuenta', how='inner')
        
        # Renombrar columnas
        dataset_final.columns = [
            'Numero_cuenta', 'Semestre', 'Carrera', 'Situacion_economica',
            'Fuente_ingresos', 'Sentimiento_financiero', 'Dias_ejercicio',
            'Actividad_recreativa', 'Horas_sueno', 'Edad'
        ]
        
        print(f"✅ Dataset final creado: {len(dataset_final)} registros")
        return dataset_final

    def generar_reportes(self, dataset_final):
        """Generar reportes detallados"""
        print("\n📊 GENERANDO REPORTES...")
        
        print("\n1. RESUMEN GENERAL:")
        print(f"   - Total de registros: {len(dataset_final)}")
        print(f"   - Variables: {len(dataset_final.columns)}")
        
        print("\n2. DISTRIBUCIÓN POR SITUACIÓN ECONÓMICA:")
        print(dataset_final['Situacion_economica'].value_counts())
        
        print("\n3. ESTADÍSTICAS DE ESTILO DE VIDA:")
        print(f"   - Edad promedio: {dataset_final['Edad'].mean():.1f} años")
        print(f"   - Horas de sueño promedio: {dataset_final['Horas_sueno'].mean():.1f} horas")
        print(f"   - Días de ejercicio promedio: {dataset_final['Dias_ejercicio'].mean():.1f} días")
        
        print("\n4. DATOS FALTANTES:")
        print(dataset_final.isnull().sum())

    def ejecutar_limpieza_completa(self):
        """Ejecutar todo el pipeline de limpieza"""
        print("🚀 INICIANDO LIMPIEZA COMPLETA MEJORADA...")
        
        # 1. Cargar datos
        if not self.cargar_datos():
            return None
        
        # 2. Limpiar datasets
        self.limpiar_economia()
        self.limpiar_estilo_vida()
        
        # 3. Crear dataset final
        dataset_final = self.crear_dataset_final()
        
        # 4. Generar reportes
        self.generar_reportes(dataset_final)
        
        # 5. Guardar resultados
        dataset_final.to_csv('dataset_final_mejorado.csv', index=False, encoding='utf-8')
        self.economia_df.to_csv('economia_limpio_mejorado.csv', index=False, encoding='utf-8')
        self.estilo_vida_df.to_csv('estilo_vida_limpio_mejorado.csv', index=False, encoding='utf-8')
        
        print("\n💾 ARCHIVOS GUARDADOS:")
        print("   - dataset_final_mejorado.csv")
        print("   - economia_limpio_mejorado.csv")
        print("   - estilo_vida_limpio_mejorado.csv")
        
        print("\n🎉 ¡LIMPIEZA COMPLETADA EXITOSAMENTE!")
        return dataset_final

# Ejecutar la limpieza
if __name__ == "__main__":
    limpiador = LimpiadorEncuestasMejorado()
    datos_limpios = limpiador.ejecutar_limpieza_completa()
    
    if datos_limpios is not None:
        print("\n📋 MUESTRA DEL DATASET FINAL:")
        print(datos_limpios.head())