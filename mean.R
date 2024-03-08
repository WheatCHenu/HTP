#!/usr/bin/env Rscript
##means
library(tidyverse)
df <- read.table('iris.txt',header = T,sep='\t')
df2 <- aggregate(df,by=list(df$Species),mean,na.rm=T)
write.table(df2,'ckh.txt',row.names = F,quote = F,sep='\t')