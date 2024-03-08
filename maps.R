#!/usr/bin/env Rscript
library(geojsonsf) 
library(sf)
library(tidyverse)
library(rgdal)

= st_read(dsn = "http://xzqh.mca.gov.cn/data/quanguo.json", 
                stringsAsFactors=FALSE) 
st_crs(shp) = 4326
plot

ggplot(shp)+
  geom_sf()+
  labs(title="Ministry of Civil of PRC",x="Lon",y="Lat") 

China

colors <- c("#33A02C","#B2DF8A","#FDBF6F","#1F78B4","#999999",
            "#E31A1C","#E6E6E6","#A6CEE3")

library(dplyr)
data2<- read.delim("66661")
# shlot()+
  geom_sf(data=shp,fill="NA",size=0.1,color="grey", alpha=0.1)+
  # geom_sf(data=shp,aes(fill = NULL))+
  # annotation_scale(location = "bl") +
  # annotation_north_arrow(location="tl",
  #                        style = north_arrow_nautical(
  #                          fill = c("grey40","white"),
  #                          line_col = "grey20"))+
  # geom_tile(data=df_shanxi,aes(x=x,y=y,fill=layer),show.legend = F)+
  #geom_polygon(data = mapqyf, aes(x=long, y = lat, group = group),colour = "blue",size=0.1,linetype = "dotdash") +
  #geom_sf(data = rivers_cropped,col='blue',size=0.1)+
  scale_fill_gradientn(colours=colors,na.value="transparent")+
  labs(x=NULL,y=NULL)+
  # geom_polygon(data = mapChina, aes(x=long, y = lat, group = group),fill="grey", alpha=0.1) +
  #geom_point( data=data2, aes(x=long, y=lat,size=pop), color="red",alpha=0.4) +
  geom_point( data=data2, aes(x=long, y=lat), 
              # position = position_jitter(width = 1), 
              size=1,color="red",alpha=0.8) +
  theme_bw()+
  theme(panel.grid.major=element_blank(),
        panel.grid.minor=element_blank(),
        panel.background = element_blank(),
        legend.title = element_blank())

# ȫaxian = st_read(dsn = "http://xzqh.mca.gov.cn/data/xian_quanguo.json", 
                    stringsAsFactors=FALSE) 
# ת??crs(Chinaxian) = 4326
# plot
ggplot(Chinaxian)+
  geom_sf()+
  labs(title="Ministry of Civil of PRC",x="Lon",y="Lat")

# ?