from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List
import matplotlib
import matplotlib.pyplot as plt
import torch
import pandas as pd
import kmeans_cpp

matplotlib.use('Agg')  # Usar backend no interactivo
app = FastAPI()

# Definir el modelo para el vector
class VectorF(BaseModel):
    vector: List[float]

def kmeans(data, n_clusters, n_iters=10):
    # Convertir datos de PyTorch a listas de Python
    data_list = data.tolist()
    # Llamar a la implementación en C++
    centroids, assignments = kmeans_cpp.kmeans(data_list, n_clusters, n_iters)
    # Convertir resultados de vuelta a tensores
    return torch.tensor(centroids), torch.tensor(assignments)

@app.post("/kmeans")
def calculo(x1: VectorF, y1: VectorF, n_clusters: int, n_iters: int):
    # Generar datos de ejemplo
    df = pd.DataFrame({
        'x': x1.vector,
        'y': y1.vector
    })
    data = torch.tensor(df.values, dtype=torch.float32)

    # Ejecutar K-Means
    centroids, assignments = kmeans(data, n_clusters, n_iters)

    # Agregar asignaciones al DataFrame
    df['cluster'] = assignments

    # Generar gráfica
    scatter = df.plot.scatter(x='x', y='y', c='cluster', colormap='viridis', figsize=(8, 6))
    output_file = "dispersion_diag.png"
    plt.savefig(output_file)
    plt.close()  # Cierra la figura para evitar acumulación de memoria

    # Regresar el archivo como respuesta
    return FileResponse(output_file, media_type="image/png", filename="dispersion_diag.png")
