#!/usr/bin/env Rscript
## ----------------------------------------------------
library(ggplot2)
library(ggsci)
library(ggridges)
library(cowplot)
library(readxl)
#RGB+HSI
t1 = read_excel("1.xlsx",sheet = 1)
#Total
t2 = read_excel("1.xlsx",sheet = 2)
#RGB
t3 = read_excel("1.xlsx",sheet = 3)
#RGB:T1-T4
t4 = read_excel("1.xlsx",sheet = 4)

p1 <- ggplot(data = t1, aes(x = h2)) +
  geom_density(alpha = 0.4, aes(fill = type)) +
  xlim(0.01,1)+
  #scale_fill_lancet() +
  scale_fill_aaas() +
  ggtitle('RGB+HSI') +
  #scale_y_discrete(expand = c(0, 0)) +
  #scale_x_continuous(expand = c(0, 0)) +
  #theme_minimal_hgrid() +
  theme_classic() +
  theme(legend.background = element_blank(), 
        legend.position = c(0.2, 0.8))

p2 <- ggplot(data = t1, aes(x = h2)) +
  geom_density(alpha = 0.4, aes(fill = date)) +
  xlim(0.01,1)+
  #scale_fill_lancet() +
  scale_fill_aaas() +
  ggtitle('Total:T1-T4') +
  #scale_y_discrete(expand = c(0, 0)) +
  #scale_x_continuous(expand = c(0, 0)) +
  #theme_minimal_hgrid() +
  theme_classic() +
  theme(legend.background = element_blank(), 
        legend.position = c(0.2, 0.8))

p3 <- ggplot(data = t2, aes(x = h2)) +
  geom_density(alpha = 0.4, aes(fill = angle)) +
  xlim(0.01,1)+
  #scale_fill_lancet() +
  scale_fill_aaas() +
  ggtitle('RGB:side-fu-top') +
  #scale_y_discrete(expand = c(0, 0)) +
  #scale_x_continuous(expand = c(0, 0)) +
  #theme_minimal_hgrid() +
  theme_classic() +
  theme(legend.background = element_blank(), 
        legend.position = c(0.2, 0.8))

p4 <- ggplot(data = t2, aes(x = h2)) +
  geom_density(alpha = 0.4, aes(fill = date)) +
  xlim(0.01,1)+
  #scale_fill_lancet() +
  scale_fill_aaas() +
  ggtitle('RGB:T1-T4') +
  #scale_y_discrete(expand = c(0, 0)) +
  #scale_x_continuous(expand = c(0, 0)) +
  #theme_minimal_hgrid() +
  theme_classic() +
  theme(legend.background = element_blank(), 
        legend.position = c(0.2, 0.8))

library(patchwork)
p1 / p2 / p3
p1 / p2 / p3 + guide_area()
p1 / p2 / p3 + guide_area() + plot_layout(guides = "collect")
##
ggplot(data = qyf, aes(x = h2, y = date)) +
  geom_density_ridges(aes(fill = date)) +
  scale_fill_aaas() +
  xlim(0,1) +
  #scale_y_discrete(expand = c(0, 0)) +
  #scale_x_continuous(expand = c(0, 0)) +
  theme_minimal_hgrid() +
  # theme_classic() +
  theme(legend.background = element_blank(), 
        legend.position = c(0.35, 0.9),
        legend.direction = "horizontal")
