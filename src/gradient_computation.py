# -*- coding: utf-8 -*-
"""
Cálculo de Gradientes y Análisis de Funciones
Ejercicio con NumPy y visualización de derivadas
"""

import numpy as np
import matplotlib.pyplot as plt

def problema_1_funcion_lineal():
    """
    Problema 1: Función lineal y = 0.5x + 1
    """
    print("=" * 70)
    print("PROBLEMA 1: FUNCIÓN LINEAL y = 0.5x + 1")
    print("=" * 70)
    
    # Crear array de x desde -50 hasta 50 con paso 0.1
    x = np.arange(-50, 50.1, 0.1)
    y = 0.5 * x + 1
    
    print(f"✓ Array x creado: {x.shape} elementos")
    print(f"✓ Rango x: [{x.min():.1f}, {x.max():.1f}]")
    print(f"✓ Paso: {x[1] - x[0]:.1f}")
    print(f"✓ Primeros 5 valores de x: {x[:5]}")
    print(f"✓ Primeros 5 valores de y: {y[:5]}")
    
    return x, y

def problema_2_combinar_arrays(x, y):
    """
    Problema 2: Combinar arrays x e y en uno solo (1001, 2)
    """
    print("\n" + "=" * 70)
    print("PROBLEMA 2: COMBINACIÓN DE ARRAYS")
    print("=" * 70)
    
    # Combinar usando np.column_stack
    array_xy = np.column_stack([x, y])
    
    # Alternativa: np.concatenate con reshape
    # array_xy = np.concatenate([x.reshape(-1,1), y.reshape(-1,1)], axis=1)
    
    print(f"✓ Array x shape: {x.shape}")
    print(f"✓ Array y shape: {y.shape}")
    print(f"✓ Array combinado shape: {array_xy.shape}")
    print(f"✓ Primeras 5 filas combinadas:")
    for i in range(5):
        print(f"  [{array_xy[i, 0]:.1f}, {array_xy[i, 1]:.1f}]")
    
    return array_xy

def problema_3_calcular_gradiente(x, y):
    """
    Problema 3: Calcular gradiente usando diferencias finitas
    """
    print("\n" + "=" * 70)
    print("PROBLEMA 3: CÁLCULO DEL GRADIENTE")
    print("=" * 70)
    
    # Calcular gradiente: Δy/Δx
    gradient = np.diff(y) / np.diff(x)
    
    # Los gradientes corresponden a los puntos medios entre x[i] y x[i+1]
    x_gradient = (x[:-1] + x[1:]) / 2
    
    print(f"✓ Array x original: {x.shape} elementos")
    print(f"✓ Array gradiente: {gradient.shape} elementos")
    print(f"✓ Primeros 5 valores del gradiente: {gradient[:5]}")
    print(f"✓ Últimos 5 valores del gradiente: {gradient[-5:]}")
    print(f"✓ Valor teórico del gradiente (derivada exacta): 0.5")
    print(f"✓ Valor promedio del gradiente calculado: {np.mean(gradient):.6f}")
    
    return x_gradient, gradient

def problema_4_visualizar_funcion_gradiente(x, y, x_gradient, gradient):
    """
    Problema 4: Visualizar función y su gradiente
    """
    print("\n" + "=" * 70)
    print("PROBLEMA 4: VISUALIZACIÓN")
    print("=" * 70)
    
    # Configuración para CodeSpaces
    plt.switch_backend('Agg')
    
    plt.figure(figsize=(15, 10))
    
    # Gráfico 1: Función original
    plt.subplot(2, 2, 1)
    plt.plot(x, y, 'b-', linewidth=2, label='y = 0.5x + 1')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Función Lineal', fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Gráfico 2: Gradiente
    plt.subplot(2, 2, 2)
    plt.plot(x_gradient, gradient, 'r-', linewidth=2, label='Gradiente (dy/dx)')
    plt.axhline(y=0.5, color='black', linestyle='--', alpha=0.7, label='Valor teórico: 0.5')
    plt.xlabel('x')
    plt.ylabel('Gradiente')
    plt.title('Gradiente de la Función', fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Gráfico 3: Función y gradiente juntos
    plt.subplot(2, 2, 3)
    # Eje izquierdo para la función
    ax1 = plt.gca()
    ax1.plot(x, y, 'b-', linewidth=2, label='Función')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    
    # Eje derecho para el gradiente
    ax2 = ax1.twinx()
    ax2.plot(x_gradient, gradient, 'r-', linewidth=2, label='Gradiente')
    ax2.set_ylabel('Gradiente', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    plt.title('Función y su Gradiente', fontweight='bold')
    
    # Gráfico 4: Error del gradiente
    plt.subplot(2, 2, 4)
    error = np.abs(gradient - 0.5)  # Error absoluto respecto al valor teórico
    plt.plot(x_gradient, error, 'g-', linewidth=1, label='Error absoluto')
    plt.xlabel('x')
    plt.ylabel('Error')
    plt.title('Error del Gradiente Calculado', fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('funcion_lineal_gradiente.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✓ Gráficos guardados como 'funcion_lineal_gradiente.png'")

def compute_gradient(function, x_range=(-50, 50.1, 0.1)):
    """
    Función general para calcular gradiente de cualquier función
    
    Parameters:
    function : función que toma array_x y retorna array_y
    x_range : tupla con parámetros para np.arange (start, stop, step)
    
    Returns:
    array_xy : array combinado de x e y
    gradient : array con los gradientes calculados
    """
    # Crear array de x
    x = np.arange(*x_range)
    
    # Calcular y usando la función proporcionada
    y = function(x)
    
    # Combinar x e y
    array_xy = np.column_stack([x, y])
    
    # Calcular gradiente
    gradient = np.diff(y) / np.diff(x)
    
    return array_xy, gradient, x

def function1(array_x):
    """y = x²"""
    return array_x ** 2

def function2(array_x):
    """y = 2x² + 2x"""
    return 2 * array_x ** 2 + 2 * array_x

def function3(array_x):
    """y = sin(x/2) - para x en [0, 50]"""
    return np.sin(array_x / 2)

def problema_5_funciones_variadas():
    """
    Problema 5: Aplicar a tres funciones diferentes
    """
    print("\n" + "=" * 70)
    print("PROBLEMA 5: TRES FUNCIONES DIFERENTES")
    print("=" * 70)
    
    funciones = [
        ("y = x²", function1, (-50, 50.1, 0.1)),
        ("y = 2x² + 2x", function2, (-50, 50.1, 0.1)),
        ("y = sin(x/2)", function3, (0, 50.1, 0.1))
    ]
    
    resultados = []
    
    for nombre, funcion, rango in funciones:
        print(f"\n--- {nombre} ---")
        array_xy, gradient, x = compute_gradient(funcion, rango)
        x_gradient = (x[:-1] + x[1:]) / 2
        
        print(f"✓ Rango x: [{x.min():.1f}, {x.max():.1f}]")
        print(f"✓ Shape array_xy: {array_xy.shape}")
        print(f"✓ Shape gradient: {gradient.shape}")
        print(f"✓ Gradiente promedio: {np.mean(gradient):.4f}")
        print(f"✓ Gradiente mínimo: {np.min(gradient):.4f}")
        print(f"✓ Gradiente máximo: {np.max(gradient):.4f}")
        
        resultados.append((nombre, funcion, x, array_xy, gradient, x_gradient))
        
        # Visualizar cada función
        visualizar_funcion_completa(nombre, funcion, x, array_xy, gradient, x_gradient)
    
    return resultados

def visualizar_funcion_completa(nombre, funcion, x, array_xy, gradient, x_gradient):
    """
    Visualiza una función y su gradiente
    """
    plt.figure(figsize=(12, 8))
    
    # Gráfico 1: Función original
    plt.subplot(2, 1, 1)
    y = funcion(x)
    plt.plot(x, y, 'b-', linewidth=2, label=nombre)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Función: {nombre}', fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Gráfico 2: Gradiente
    plt.subplot(2, 1, 2)
    plt.plot(x_gradient, gradient, 'r-', linewidth=2, label=f'Gradiente de {nombre}')
    plt.xlabel('x')
    plt.ylabel('Gradiente (dy/dx)')
    plt.title(f'Gradiente: {nombre}', fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Guardar con nombre específico
    nombre_archivo = nombre.replace('=', '').replace(' ', '_').replace('²', '2').replace('/', 'div') + '.png'
    plt.tight_layout()
    plt.savefig(nombre_archivo, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✓ Gráfico guardado como '{nombre_archivo}'")

def problema_6_encontrar_minimos(resultados):
    """
    Problema 6: Encontrar mínimos de las funciones
    """
    print("\n" + "=" * 70)
    print("PROBLEMA 6: ENCONTRAR MÍNIMOS")
    print("=" * 70)
    
    for nombre, funcion, x, array_xy, gradient, x_gradient in resultados:
        print(f"\n--- {nombre} ---")
        
        # Encontrar mínimo de y
        y = array_xy[:, 1]
        min_y = np.min(y)
        idx_min_y = np.argmin(y)
        x_min = array_xy[idx_min_y, 0]
        
        print(f"✓ Mínimo valor de y: {min_y:.4f}")
        print(f"✓ En x = {x_min:.4f}")
        print(f"✓ Índice del mínimo: {idx_min_y}")
        
        # Encontrar gradientes alrededor del mínimo
        if idx_min_y > 0 and idx_min_y < len(gradient):
            # El gradiente en el punto anterior al mínimo
            if idx_min_y - 1 >= 0:
                grad_antes = gradient[idx_min_y - 1]
                print(f"✓ Gradiente antes del mínimo: {grad_antes:.4f}")
            
            # El gradiente en el punto del mínimo (aproximado)
            if idx_min_y < len(gradient):
                grad_despues = gradient[idx_min_y]
                print(f"✓ Gradiente después del mínimo: {grad_despues:.4f}")
            
            # Análisis del comportamiento del gradiente
            print(f"✓ ¿Gradiente cambia de signo? ", end="")
            if idx_min_y > 0 and idx_min_y < len(gradient):
                if gradient[idx_min_y - 1] * gradient[idx_min_y] <= 0:
                    print("SÍ (indica punto crítico)")
                else:
                    print("NO")
        
        # Información adicional sobre la función
        print(f"✓ Rango de y: [{np.min(y):.4f}, {np.max(y):.4f}]")
        
        # Para funciones cuadráticas, verificar el vértice teórico
        if "x²" in nombre:
            if "2x² + 2x" in nombre:
                # Vértice de y = 2x² + 2x es x = -b/(2a) = -2/(4) = -0.5
                vertice_teorico = -0.5
                print(f"✓ Vértice teórico: x = {vertice_teorico:.4f}")
                print(f"✓ Diferencia con mínimo encontrado: {abs(x_min - vertice_teorico):.6f}")
            elif "x²" in nombre and "2x" not in nombre:
                # Vértice de y = x² es x = 0
                vertice_teorico = 0
                print(f"✓ Vértice teórico: x = {vertice_teorico:.4f}")
                print(f"✓ Diferencia con mínimo encontrado: {abs(x_min - vertice_teorico):.6f}")

def demostracion_derivadas_teoricas():
    """
    Demostración de las derivadas teóricas vs calculadas
    """
    print("\n" + "=" * 70)
    print("COMPARACIÓN: DERIVADAS TEÓRICAS VS CALCULADAS")
    print("=" * 70)
    
    x = np.arange(-50, 50.1, 0.1)
    
    # Función 1: y = x², derivada: 2x
    y1 = function1(x)
    grad_teorico1 = 2 * x
    grad_calculado1 = np.diff(y1) / np.diff(x)
    x_grad = (x[:-1] + x[1:]) / 2
    
    error1 = np.mean(np.abs(grad_calculado1 - grad_teorico1[:-1]))
    print(f"Función y = x²:")
    print(f"  • Derivada teórica: 2x")
    print(f"  • Error promedio: {error1:.6f}")
    
    # Función 2: y = 2x² + 2x, derivada: 4x + 2
    y2 = function2(x)
    grad_teorico2 = 4 * x + 2
    grad_calculado2 = np.diff(y2) / np.diff(x)
    
    error2 = np.mean(np.abs(grad_calculado2 - grad_teorico2[:-1]))
    print(f"Función y = 2x² + 2x:")
    print(f"  • Derivada teórica: 4x + 2")
    print(f"  • Error promedio: {error2:.6f}")
    
    # Función 3: y = sin(x/2), derivada: (1/2)cos(x/2)
    x3 = np.arange(0, 50.1, 0.1)
    y3 = function3(x3)
    grad_teorico3 = 0.5 * np.cos(x3 / 2)
    grad_calculado3 = np.diff(y3) / np.diff(x3)
    
    error3 = np.mean(np.abs(grad_calculado3 - grad_teorico3[:-1]))
    print(f"Función y = sin(x/2):")
    print(f"  • Derivada teórica: (1/2)cos(x/2)")
    print(f"  • Error promedio: {error3:.6f}")

def main():
    """
    Función principal del programa
    """
    print("📈 CÁLCULO DE GRADIENTES Y ANÁLISIS DE FUNCIONES")
    print("=" * 70)
    
    try:
        # Problemas 1-4: Función lineal
        print("FASE 1: FUNCIÓN LINEAL")
        x, y = problema_1_funcion_lineal()
        array_xy = problema_2_combinar_arrays(x, y)
        x_gradient, gradient = problema_3_calcular_gradiente(x, y)
        problema_4_visualizar_funcion_gradiente(x, y, x_gradient, gradient)
        
        # Problema 5: Funciones variadas
        print("\nFASE 2: FUNCIONES VARIADAS")
        resultados = problema_5_funciones_variadas()
        
        # Problema 6: Encontrar mínimos
        print("\nFASE 3: BÚSQUEDA DE MÍNIMOS")
        problema_6_encontrar_minimos(resultados)
        
        # Demostración adicional
        demostracion_derivadas_teoricas()
        
        # Resumen final
        print("\n" + "=" * 70)
        print("✅ EJERCICIO COMPLETADO EXITOSAMENTE")
        print("=" * 70)
        print("🎯 CONCEPTOS APRENDIDOS:")
        print("   • Cálculo de gradientes con diferencias finitas")
        print("   • Visualización de funciones y sus derivadas")
        print("   • Uso de NumPy para operaciones matemáticas")
        print("   • Búsqueda de mínimos en funciones")
        print("   • Comparación entre derivadas teóricas y numéricas")
        
        print("\n📁 ARCHIVOS GENERADOS:")
        print("   • funcion_lineal_gradiente.png")
        print("   • y_x2.png")
        print("   • y_2x2_2x.png")
        print("   • y_sinxdiv2.png")
        
    except Exception as e:
        print(f"❌ Error durante la ejecución: {e}")
        import traceback
        traceback.print_exc()

# Ejecutar el programa
if __name__ == "__main__":
    main()
