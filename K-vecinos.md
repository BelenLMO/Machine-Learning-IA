# Machine-Learning
Diferentes ejercicios 
K-vecinos: K Vecinos Más Cercanos para Clasificación
En este ejercicio debes desarrollar una función que aplique el algoritmo de los k vecinos más cercanos (KNN) para un problema de clasificación.

Supongamos que tienes un conjunto de datos que contiene información sobre diferentes tipos de flores, y deseas predecir el tipo de flor en función de las características de pétalos y sépalos.
Utilizaremos el conjunto de datos Iris, que es un conjunto de datos de clasificación ampliamente utilizado en el aprendizaje automático.

def knn_clasificacion(datos, k=3):

# Ejemplo de uso con el conjunto de datos Iris
data = pd.read_csv('iris.csv')  # Reemplaza 'iris.csv' con tu archivo de datos
modelo_knn = knn_clasificacion(data, k=3)
 
# Estimaciones de clasificación para nuevas muestras
nuevas_muestras = pd.DataFrame({
    'LargoSepalo': [5.1, 6.0, 4.4],
    'AnchoSepalo': [3.5, 2.9, 3.2],
    'LargoPetalo': [1.4, 4.5, 1.3],
    'AnchoPetalo': [0.2, 1.5, 0.2]
})
 
estimaciones_clasificacion = modelo_knn.predict(nuevas_muestras)
print("Estimaciones de Clasificación:")
print(estimaciones_clasificacion)


Resultados:
Estimaciones de Clasificación:
['setosa' 'versicolor' 'setosa']
