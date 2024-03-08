#!/usr/bin/env Rscript
# devtools::install_github("Hy4m/linkET", force = TRUE)
library(linkET)
library(dplyr)
## 
## Attaching package: 'dplyr'
## The following objects are masked from 'package:stats':
## 
##     filter, lag
## The following objects are masked from 'package:base':
## 
##     intersect, setdiff, setequal, union
library(ggplot2)
varechem <- read.delim("1.txt", row.names=1, stringsAsFactors=TRUE)
varespec <- read.delim("2.txt", row.names=1, stringsAsFactors=TRUE)
# data("varechem", package = "vegan")
# data("varespec", package = "vegan")
class(varechem)
## [1] "data.frame"
class(varespec)
## [1] "data.frame"
glimpse(varechem)
glimpse(varespec)

# mantel
mantel <- mantel_test(varespec, varechem,
                      spec_select = list(transpiration.rates = 1:1,
                                         CO2.ssimilation.rates = 2:2,
                                         water.use.efficiency = 3:3)) %>% 
  mutate(rd = cut(r, breaks = c(-Inf, 0.2, 0.4, Inf), # 
                  labels = c("< 0.2", "0.2 - 0.4", ">= 0.4")),
         pd = cut(p, breaks = c(-Inf, 0.01, 0.08, Inf), # P
                  labels = c("< 0.01", "0.01 - 0.05", ">= 0.05")))

# plot
qcorrplot(correlate(varechem), type = "lower", diag = FALSE) +
  geom_square() +
  geom_couple(aes(colour = pd, size = rd), 
              data = mantel, 
              curvature = nice_curvature()) +
  
  # color and name
  scale_fill_gradientn(colours = RColorBrewer::brewer.pal(11, "RdBu")) +
  scale_size_manual(values = c(0.5, 1, 2)) +
  scale_colour_manual(values = color_pal(3)) +
  guides(size = guide_legend(title = "Mantel's r",
                             override.aes = list(colour = "grey35"), 
                             order = 2),
         colour = guide_legend(title = "Mantel's p", 
                               override.aes = list(size = 3), 
                               order = 1),
         fill = guide_colorbar(title = "Pearson's r", order = 3))
