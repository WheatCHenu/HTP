#!/usr/bin/env Rscript
qyf <- read.delim("1.txt", stringsAsFactors=TRUE)
data <- qyf
library(GGally)
ggpairs(data=qyf, 
        columns = c("TPA","Fresh.weight"),
        aes(color = treat,alpha = 0.7)) +
  theme(panel.grid = element_blank(),
        panel.border = element_rect(fill=NA),
        axis.text =  element_text(color='black'))