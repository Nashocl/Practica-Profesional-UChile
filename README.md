# Reclutamiento de crisis en Epileptores acoplados

Práctica profesional de Ignacio Tobar en el Centro de Modelamiento Matemático (CMM) de la Universidad de Chile, con Joaquín Fontbona como profesor guía.

El trabajo estudia la reducción a dos variables del Epileptor con acoplamiento por permitividad, que corresponde a la ecuación (8) de Proix et al. (2014). Parte con una región aislada, sigue con dos regiones acopladas y termina con una red en la que las crisis pasan de un nodo a otro.

## Contenido

- `sitio/`: página web que presenta el trabajo. Tiene laboratorios interactivos para N = 1, N = 2, el modelo con ruido, el grafo dinámico y su versión en 3D, además de las figuras de las simulaciones.
- `Práctica_OFICIAL.ipynb`: simulaciones y análisis para N = 1 y N = 2, el modelo con ruido y la extensión a N regiones.
- `Grafo_Dinamico_Epileptogenico_Colab_3D.ipynb`: construcción de la matriz de adyacencia, grafo dinámico y reclutamiento en la red. Incluye el grafo en 3D, las capas temporales, la animación y la comparación con campo medio.

## Ver la página

Está publicada con GitHub Pages en https://nashocl.github.io/Practica-Profesional-UChile/

Para verla en tu computador necesitas Python. Desde la carpeta del repositorio:

```bash
python -m http.server 5173 --directory sitio
```

Después abre `http://localhost:5173` en el navegador. Si abres `sitio/index.html` con doble clic la página se ve, pero el modelo 3D de la portada no carga.

La página necesita conexión a internet, porque las fuentes, las fórmulas (KaTeX) y las animaciones (Motion) se cargan desde servidores externos.

## Los notebooks

Usan `numpy`, `scipy`, `matplotlib`, `networkx` y `pandas`. Se pueden abrir en Jupyter o subir a Google Colab.

## Referencia

Proix, T., Bartolomei, F., Chauvel, P., Bernard, C., Jirsa, V. K. (2014). Permittivity Coupling across Brain Regions Determines Seizure Recruitment in Partial Epilepsy. *The Journal of Neuroscience*, 34(45), 15009-15021. https://doi.org/10.1523/JNEUROSCI.1570-14.2014

El PDF del paper no está en el repositorio. Se puede leer en el enlace anterior.

## Créditos

- Three.js (licencia MIT), incluido en `sitio/vendor/three/`.
- El modelo 3D del cerebro (`sitio/assets/models/brain.glb`) es de un tercero. Falta agregar aquí su autor y su licencia.
