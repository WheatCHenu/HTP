setwd("")
library(ggplot2)
library(ggsci)
library(cowplot)
library(gridExtra)
MLP6 <- read.delim("MLP1.txt", stringsAsFactors=TRUE)
ggplot()+
  xlim(0,0.8)+
  geom_histogram(data=MLP6,aes(MLP,..density..,fill=type),color='white',alpha=0.5,binwidth = 0.02)+
  geom_density(data=MLP6,aes(MLP,..density..,color=type),size=1,linetype=1)+
  theme(legend.background = element_blank(), 
        legend.position = c(0.1, 0.8))+
  theme_classic()

 

##????????
PH<-data.frame(rnorm(300,75,5))
names(PH)<-c('PH')
#??ʾ????
head(PH)

library(ggplot2)
library(gridExtra)
p1<-ggplot(data=PH,aes(PH))+
  geom_histogram(color='white',fill='gray60')+ #??????ɫ
  ylab(label = 'total number') #?޸?Y????ǩ

#?޸?????֮???ľ???
p2<-ggplot(data=PH,aes(PH))+
  geom_histogram(color='white',fill='gray60',binwidth = 3)

#????????????
p3<-ggplot(data=PH,aes(PH,..density..))+
  geom_histogram(color='white',fill='gray60',binwidth = 3)+
  geom_line(stat='density')

#?޸??????Ĵ?ϸ
p4<-ggplot(data=PH,aes(PH,..density..))+
  geom_histogram(color='white',fill='gray60',binwidth = 3)+
  geom_line(stat='density',size=0.5)

#?ϲ?????ͼ??ƴͼ??
grid.arrange(p1,p2,p3,p4)

#?????ܶ?????
p1<-ggplot(data=PH,aes(PH,..density..))+
  geom_density(size=1.5)
#?޸???????ʽ
p2<-ggplot(data=PH,aes(PH,..density..))+
  geom_density(size=1.5,linetype=2)
p3<-ggplot(data=PH,aes(PH,..density..))+
  geom_density(size=1.5,linetype=5)

#?޸???ɫ
p4<-ggplot(data=PH,aes(PH,..density..))+
  geom_density(size=0.5,linetype=1,colour='red')

grid.arrange(p1,p2,p3,p4)

##????????չʾ
#????��??????
df<-data.frame(c(rnorm(200,5000,200),rnorm(200,5000,600)),rep(c('BJ','TJ'),each=200))    
names(df)<-c('salary','city')

library(ggplot2)
# p1<-ggplot()+
#   geom_histogram(data=df,aes(salary,..density..,fill=city),color='white')
# p2<-ggplot()+
#   geom_histogram(data=df,aes(salary,..density..,fill=city),color='white',alpha=0.5)
# p3<-ggplot()+
#   geom_density(data=df,aes(salary,..density..,color=city))
ggplot()+
  geom_histogram(data=df,aes(salary,..density..,fill=city),color='white',alpha=0.5)+
  geom_density(data=df,aes(salary,..density..,color=city),size=1,linetype=1)






