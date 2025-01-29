# Read flare parameter .csv file and analyse distributions of relevant 
# parameters.

library('readr')
library('poweRlaw')

datafolder <- '/home/giovanni/Projects/flares_TESS_fullsample/results_cluster/'
data1 <- read_csv(paste(datafolder, 'flare_params.csv', sep=''))

teff <- data1$'Teff [K]'
radius <- data1$'radius'
flag = radius < 1.0 & teff < 4200.

ampl <- data1$'Amplitude'[flag]
dur <- data1$'Duration [min]'[flag]

m_pl = conpl$new(ampl)
est = estimate_xmin(m_pl)
m_pl$setXmin(est)
plot(m_pl, xlab='Amplitude', ylab='CCDF')
lines(m_pl, col = 2)
#bs = bootstrap(fpl, no_of_sims=1000, threads=6)

