#!/usr/bin/env Rscript
library(clusterProfiler)
library(org.TaestivumCSv1.1.eg.db)
genelist=read.table('1.txt',header=F)
go_result=enrichGO(genelist$V1,OrgDb = 'org.TaestivumCSv1.1All.eg.db', ont = "ALL",keyType = 'GID')
dotplot(go_result, showCategory=10,font.size=10)