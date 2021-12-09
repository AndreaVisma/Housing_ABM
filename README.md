# Housing_ABM
NetLogo code for our ABM research project on housing. A project by Leonie Wolff, Lukas Guyer and Andrea Vismara

# 1. Description of Model Features

### Initial distribution of income across households
Each household belongs to a certain 'class' assigned randomly at creation. Consequently, each household is assigned an initial income which depends on 1) a fixed number for each class; 2) the number of adults in the household; 3) a random number extracted from a gamma distribution with kappa = 6 and theta = 9 -> this ensures the creation of an income distribution with Gini coefficient around 0.3, which is qualitatively similar to what we seei in major European cities. The calculation of the Gini coefficients associated with different functional forms of the gamma distribution is performed in the python file "gamma_distr_ginis.py".

The functional form of the income determining function is: Y = 1) * 2) * random-gamma (kappa = 6, theta = 9)
