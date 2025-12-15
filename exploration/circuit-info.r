library(tidyverse)

data = read_csv('./data/grid-results.csv')

unique(data['circuitId']) %>% write_csv(file = "./data/circuit-type.csv")
