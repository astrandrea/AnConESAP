# **Análisis contratación Escuela Superior de Administración Pública 2020-2024**

## Introducción

En este documento se presenta la aplicación de diferentes metodologías estudiadas en el curso de Programación II de la Maestría en Inteligencia de Negocios de la Universidad de Externado de Colombia. Lo anterior, se da en un ambiente de análisis y comprensión del fenómeno de la contratación pública a partir de la exploración de  diferentes escenarios aplicados a la realidad de las Entidades Públicas de Colombia. 

En este sentido, se exploran y analizan los resultados del modelo de efectos fijos del panel de datos de la contratación de la Escuela Superior de Administración Pública como individuo de análisis del 2020 a 2024 según los datos abiertos publicados por la Agencia Nacional de Contratación Pública - Colombia Compra Efienciete sobre el data set [SECOP II - Contratos Electrónicos](https://www.datos.gov.co/Gastos-Gubernamentales/SECOP-II-Contratos-Electr-nicos/jbjy-vk9h/about_data).

 De esta forma, se exploran ambientes y técnicas propicios en el contexto del aprendizaje sobre programación y en la aplicación de métodos estadísticos para la profundización y reconocimiento del fenómeno de estudio a la luz del procesamiento, depuración y análisis de la información, todos ellos necesarios en el actual contexto en donde es indispensable el uso de herramientas para la gestión de los datos y de la información en las organizaciones para el análisis y la toma de decisiones sobre los fenónemos que les afectan.

 ## Objetivo general

 Analizar el efecto que tiene el tiempo de ejecución del proceso de contratación sobre el valor total de los contratos en la Escuela Superior de Administración Pública.

 ## Objetivos específicos

 *$1.$* Explorar métodos de limpieza, depuración y automatización de la bases datos de contratos de la Escuela Superior de Administración Pública a partir del uso y aprovechamiento de los datos e información de del data set [SECOP II - Contratos Electrónicos](https://www.datos.gov.co/Gastos-Gubernamentales/SECOP-II-Contratos-Electr-nicos/jbjy-vk9h/about_data)

*$2.$* Aplicar técnicas para medir el efecto del tiempo de ejecución de la contratación pública sobre el valor de los contratos celebrados por la Escuela Superior de Administración Pública durante el período 2020-2024

*$3.$* Analizar el fenómeno de la contratación pública con Colombia con el propósito de brindar recomendaciones sobre el buen manejo de los recursos públicos a los sujetos responsables de su ejecución

## Antecedentes

A partir de la adopción de los principios de la Función Administrativa (Art. 209) de la Constitución Política de 1991, el Sector Público Colombiano ha desarrollado diferentes acciones para sinterizar un Sistema de Gestión que responda a las necesidades de articulación de los elementos que intervienen al momento de dirigir la gestión pública  hacia “el mejor desempeño institucional y a la consecución de resultados para la satisfacción de las necesidades y el goce efectivo de los derechos de los ciudadanos, en el marco de la legalidad y la integridad” (DAFP, 2019, p. 19).
<!--  -->
El Sistema de Gestión vigente (Art. 133 - Ley 1753 de 2015) integra el Sistema de Desarrollo Administrativo (Ley 489 de 1998), el Sistema de Gestión de la Calidad (Ley 872 de 2003) y se articula con el Sistema de Control Interno (Ley 87 de 1993 y del Art. 27 al 19 de la Ley 489); que finalmente constituyen el MIPG como el mecanismo que facilita la integración y articulación de los Sistemas de Planeación y Gestión que hasta el momento existían en la entidades públicas colombianas (Decreto 1499 de 2017), integró en 2021 a partir del la Política de Gestión y desempeño **Compras y Contatación Pública** con el fin de que las entidades estatales gestionen adecuadamente sus compras y contrataciones públicas a través de herramientas tecnológicas, lineamientos normativos, documentos estándar, técnicas de abastecimiento estratégico e instrumentos de agregación de demanda. 

De esta forma, con esta política la Agencia Nacional de Contratación Pública - Colombia Compra Efienciente, proyecta el suministro de una herramienta como lo es el SECOP II para que las entidades estatales  se alinien con el propósito de generar buenas mejores prácticas en el abastecimiento y contratación y con ello fortalecer la satisfacción de las necesidades públicas (eficacia), con menores recursos (eficiencia), altos estándares de calidad, pluralidad de oferentes y garantía de transparencia y rendición de cuentas.

Sin embargo, aun cuando la integración de lineamientos en materia de contratación pública sea un propósito de la administración pública conducente al buen manejo del prespuesto y a la planeación de las finanzas públicas, esto resulta en un reto su implementación y aprovechamiento para la toma de decisones en el ámbito de la contratación pública. Los procedimientos que rigen la contratación deben verse reflejados en la planeación de la contratación y se deben implementar en contextos de alta incertidumbre que las organizaciones constantemente reflejan.

En este sentido, para evidenciar lo anteriormente expuesto, se trata de analizar el efecto que tiene el tiempo de ejecución de los procesos de contratación sobre el valor de la contratación de la Escuela Superior Pública, vista en este análisis como un individuo sobre el el cual es posible su estudio a la luz de los objetivos planteados en este proyecto y según la información disponible en el data set datos abiertos publicados por la Agencia Nacional de Contratación Pública - Colombia Compra Efienciete sobre el data set [SECOP II - Contratos Electrónicos] (https://www.datos.gov.co/Gastos-Gubernamentales/)


### _Libreriras_

!pip install sodapy
!pip install seaborn
!pip install pandas
!pip install plotly
!pip install linearmodels
!pip install scikit-learn

#Pandas y Nunmpy
import pandas as pd
import numpy as np
#Transformación de fecha
from datetime import datetime
#Gráficas
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
#Se importa la clase PanelOLS, que se utiliza para realizar regresiones lineales en modelos de panel
# o conjuntos de series temporales cruzadas y estimar relaciones lineales.
from linearmodels.panel import PanelOLS
import pandas as pd
#Se importa la biblioteca "statsmodels" y la renombra como "sm" 
#Statsmodels es una biblioteca que proporciona herramientas y modelos estadísticos para el análisis 
#de datos en Python. Se utiliza comúnmente para realizar análisis de regresión, 
#series de tiempo y otras técnicas estadísticas.
import statsmodels.api as sm

### _Exploración de la base de datos_

#### _Cargar base de datos_

Se utiliza la base de datos contratos_secop_esap del data set [SECOP II - Contratos Electrónicos] (https://www.datos.gov.co/Gastos-Gubernamentales/)
