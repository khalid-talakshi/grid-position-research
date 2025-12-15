library(tidyverse)

data = read_csv("./data/grid-results.csv")
circuit_info = read_csv("./data/circuit-type.csv")

merged_data = merge(data, circuit_info, by.x = "circuitId", by.y = "circuitId")
merged_data[TrackType] = merged_data[type]
colnames(merged_data)

data_2018 = data %>% 
  filter(EventName == "Dutch Grand Prix") 
data_2018_count = data_2018 %>% 
  count(GridPosition, ClassifiedPosition)

all_data_count = data %>% count(GridPosition, ClassifiedPosition)


ggplot(data = all_data_count, mapping = aes(x = GridPosition, y = ClassifiedPosition, fill = n)) +
  geom_tile() +
  scale_fill_viridis_c()

data_position = data %>% 
  filter(GridPosition == 11) 

data_position = data %>% count(GridPosition, ClassifiedPosition)

ggplot(data = data_position, mapping = aes(x = ClassifiedPosition)) +
  geom_histogram()

ggplot(data = data, mapping = aes(x = as.factor(GridPosition), y = ClassifiedPosition)) +
  geom_boxplot()

ggplot(data = data_position, mapping = aes(x = ClassifiedPosition, y = n, fill = as.factor(GridPosition) ))+
  geom_col(position = "dodge")


merged_data %>% 
    filter(GridPosition <= 8) %>% 
    ggplot(mapping = aes(x = ClassifiedPosition)) +
    geom_histogram(fill = as.factor(type)) +
    facet_wrap(~ as.factor(GridPosition), nrow = 2)
