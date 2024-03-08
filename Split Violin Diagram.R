#!/usr/bin/env Rscript
library(pacman)
# p_install_gh("psyteachr/introdataviz") #github??װR??
# devtools::install_github("psyteachr/introdataviz")
library(introdataviz)
library(tidyverse)
colours <- c("#5BC9CC","#F09C95")
ggplot(qyf11, aes(x = condition, y = side.T2.drr.FDIC, fill = language)) +
  introdataviz::geom_split_violin(alpha = .4,width = .9, scale = "width") +
  geom_boxplot(width = .5, alpha = .6, show.legend = FALSE) +
  stat_summary(fun.data = "mean_se", geom = "pointrange", show.legend = F,
               position = position_dodge(.5)) +
  # scale_x_discrete(name = "", labels = c("T1", "T2","T3","T4")) +
  #limits = c(0, 0.75),
  scale_y_continuous(name = "Side-PC1(RGB)") +
  scale_fill_manual(values = colours, name = "treat") +
  ylim(0,0.75)+
  theme_classic()
# theme_minimal()
p + geom_density()
p2 <- ggplot(qyf11, aes(x = condition, y = topT1.CK.GPA, fill = language)) +
  introdataviz::geom_split_violin(alpha = .4) +
  geom_boxplot(width = .2, alpha = .6, show.legend = FALSE) +
  stat_summary(fun.data = "mean_se", geom = "pointrange", show.legend = F, 
               position = position_dodge(.175)) +
  scale_x_discrete(name = "Stage", labels = c("T1", "T2","T3","T4")) +
  #limits = c(0, 0.75),
  scale_y_continuous(name = "GPA(RGB)") +
  scale_fill_manual(values = colours, name = "treat") +
  theme_minimal()
p3 <- ggplot(qyf33, aes(x = condition, y = rt, fill = language)) +
  introdataviz::geom_split_violin(alpha = .4) +
  geom_boxplot(width = .2, alpha = .6, show.legend = FALSE) +
  stat_summary(fun.data = "mean_se", geom = "pointrange", show.legend = F, 
               position = position_dodge(.175)) +
  scale_x_discrete(name = "Stage", labels = c("T1", "T2","T3","T4")) +
  scale_y_continuous(limits = c(-1, 0.75),name = "ddt102(Plant-HSI)") +
  scale_fill_manual(values = colours, name = "treat") +
  theme_minimal()

p4 <- ggplot(qyf66, aes(x = type, y = number, fill = factor(type))) +
  geom_half_violin(aes(fill = type), 
                   side = 'l', # l:left;r:right
                   position = position_nudge(x = .01, y = 0), # λ??΢??
                   adjust = 1/2) + # ???? binwidth) +
  geom_boxplot(width = 0.1,aes(fill = type), outlier.shape = 21,
               position = position_nudge(x = .01, y = 0)) +
scale_y_continuous(limits = c(0, 0.75),name = "ddT83(Seed-HSI)") +
  geom_signif(
    # comparisons = list(c("dr-T3", "dr-T1"),
    #                    c("dr-T3", "ck-T3"),
    #                    c("dr-T3", "ck-T1")),
    comparisons = list(c("DR", "CK")),
    # step_increase = 0.1,
    map_signif_level = t,
    # y_position = c(3.3, 4.2),
    test = t.test)+
  scale_fill_lancet() +
  xlab('') +
  theme_cowplot() +
  scale_x_discrete(labels = abbreviate, name = "treat")