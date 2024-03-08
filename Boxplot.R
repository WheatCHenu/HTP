#!/usr/bin/env Rscript
library(tidyverse)
library(ggplot2)
library(ggsci)
library(cowplot)
library(ggsignif)
library(ggrepel)
library(ggpubr)
qyf1 <- read.delim("1.txt", stringsAsFactors=TRUE)
qyf2 <- read.delim("2.txt", stringsAsFactors=TRUE)
qyf3 <- read.delim("3.txt", stringsAsFactors=TRUE)

a<-ggplot(qyf1, aes(x=treat, y=X155.gdrr.2.dT.68, fill=treat)) +
  geom_boxplot() +
  # stat_compare_means(method = "t.test", label="p.signif")+
  geom_signif(
    comparisons = list(c("H1", "H2")),
    # step_increase = 0.1,
    map_signif_level = TRUE,
    y_position = c(0.8, 0.4),
    test = t.test) +
  
  scale_fill_lancet() +
  xlab('') +
  # ylim(0,1)+
  theme_cowplot()
b<-ggplot(qyf2, aes(x=treat, y=Ratio.of.grain.length.to.grain.width, fill=treat)) +
  geom_boxplot() +
  # stat_compare_means(method = "t.test", label="p.signif")+
  geom_signif(
    comparisons = list(c("H1", "H2")),
    # step_increase = 0.1,
    map_signif_level = TRUE,
    y_position = c(2.5, 0.4),
    test = t.test) +
  
  scale_fill_lancet() +
  xlab('') +
  # ylim(0,1)+
  theme_cowplot()
c<-ggplot(qyf3, aes(x=treat, y=Grain.roundness, fill=treat)) +
  geom_boxplot() +
  # stat_compare_means(method = "t.test", label="p.signif")+
  geom_signif(
    comparisons = list(c("H1", "H2")),
    # step_increase = 0.1,
    map_signif_level = TRUE,
    y_position = c(0.6, 0.4),
    test = t.test) +
  
  scale_fill_lancet() +
  xlab('') +
  # ylim(0,1)+
  theme_cowplot()


library(ggpubr)
pdf("phe.pdf",width = 8,height = 8)
ggarrange(a,b,c,d + rremove("x.text"), ncol = 1, nrow = 4)
dev.off()


library(ggpubr)
a<-ggplot(phe,aes(x=group,y=TL,fill=group)) +geom_boxplot()+geom_signif(comparisons = list(c("strong","weak")),map_signif_level = TRUE,test = t.test,y_position = c(80,30),tip_length = c(0.05,0.4))+theme_bw()+ scale_fill_manual(values = c("#DE6757","#5B9BD5"))+theme(panel.grid.major = element_blank(),panel.grid.minor = element_blank())+labs(x="Group", y="TL value")+theme(axis.title.x =element_text(size=12,face = "bold"), axis.title.y=element_text(size=12,face = "bold"),axis.text = element_text(size = 12,face = "bold"))
b<-ggplot(phe,aes(x=group,y=SA,fill=group)) +geom_boxplot()+geom_signif(comparisons = list(c("strong","weak")),map_signif_level = TRUE,test = t.test,y_position = c(17,5),tip_length = c(0.05,0.4))+theme_bw()+ scale_fill_manual(values = c("#DE6757","#5B9BD5"))+theme(panel.grid.major = element_blank(),panel.grid.minor = element_blank())+labs(x="Group", y="SA value")+theme(axis.title.x =element_text(size=12,face = "bold"), axis.title.y=element_text(size=12,face = "bold"),axis.text = element_text(size = 12,face = "bold"))
c<-ggplot(phe,aes(x=group,y=AD,fill=group)) +geom_boxplot()+geom_signif(comparisons = list(c("strong","weak")),map_signif_level = TRUE,test = t.test,y_position = c(0.9,0),tip_length = c(0.05,0.05))+theme_bw()+ scale_fill_manual(values = c("#DE6757","#5B9BD5"))+theme(panel.grid.major = element_blank(),panel.grid.minor = element_blank())+labs(x="Group", y="AD value")+theme(axis.title.x =element_text(size=12,face = "bold"), axis.title.y=element_text(size=12,face = "bold"),axis.text = element_text(size = 12,face = "bold"))
d<-ggplot(phe,aes(x=group,y=NR,fill=group)) +geom_boxplot()+geom_signif(comparisons = list(c("strong","weak")),map_signif_level = TRUE,test = t.test,y_position = c(40,0),tip_length = c(0.05,0.4))+theme_bw()+ scale_fill_manual(values = c("#DE6757","#5B9BD5"))+theme(panel.grid.major = element_blank(),panel.grid.minor = element_blank())+labs(x="Group", y="NR value")+theme(axis.title.x =element_text(size=12,face = "bold"), axis.title.y=element_text(size=12,face = "bold"),axis.text = element_text(size = 12,face = "bold"))
pdf("phe.pdf",width = 8,height = 8)
ggarrange(a,b,c,d + rremove("x.text"), ncol = 2, nrow = 2)
dev.off()


library(ggplot2)
library(gridExtra)
library(gapminder)
library(dplyr)
# grid.arrange(p1, p2, p3, p4,nrow = 2)
grid.arrange(a, b, c,nrow = 1)
