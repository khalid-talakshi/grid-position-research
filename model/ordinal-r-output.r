library(ggplot2)
probs = read_csv("./model/ordinal-logistic-grid-pos-probs.csv")

win_pdf = function(c, g) {
  p = probs %>%
    filter(grid_position == g, finish_position == c) %>%
    pull(probability)
  return(p)
}

win_cdf = function(c, g) {
  p = probs %>%
    filter(grid_position == g, finish_position <= c) %>%
    summarise(prob = sum(probability)) %>%
    pull(prob)
  return(p)
}

prob_matrix <- probs |>
  tidyr::pivot_wider(names_from = finish_position, values_from = probability) |>
  tibble::column_to_rownames("grid_position") |>
  as.matrix()

prob_matrix

heatmap(
  prob_matrix[nrow(prob_matrix):1, ],
  Rowv = NA,
  Colv = NA,
  xlab = "Finish Position",
  ylab = "Grid Position",
  main = "P(finish position | grid position)",
  col = colorRampPalette(c("white", "blue"))(100),
  scale = "none"
)

ggplot() +
  geom_line(data = probs, aes(x = grid_position, y = probability)) +
  facet_wrap(~ finish_position, ncol = 4) 
