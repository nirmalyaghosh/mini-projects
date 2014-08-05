library(maptools)
library(RColorBrewer)
library(rgeos)
library(stringdist)

##############################################################################
## Reads a tab-separated file containing Ebola alerts collected from the  
## WHO site and a CSV file indicating the path to the required shape files; 
## and accordingly plots the affected districts on a map
## 
## Created on Tue Aug 05 10:14:03 2014
## 
## Author: Nirmalya Ghosh
##############################################################################

## Read the known affected prefectures/districts
## NOTE : Prefix 'ob' for outbreak
ob_areas = read.table("WHO-Alerts-Ebola.tsv", header=T, stringsAsFactors=F)
ob_areas$URL <- NULL # Drop the 'URL' column, since not required
## Read the file indicating the path of the required shape files
shapeFiles <- read.csv("shapefiles.csv", header=TRUE, stringsAsFactors=FALSE)

## Construct a data frame
ob_df <- merge(ob_areas, shapeFiles,by=c("Country"))

## Read the shape file for Africa - this is the high level container
shp_Africa <- readShapePoly(shapeFiles[shapeFiles$Country=="Africa",]$FilePath)

## Create the bounding box 
## NW corner : 12.264193,-17.468262, SE corner : 6.987347,-6
lats <- c(12.264193,6.987347)
lons <- c(-17.468262,-6)
bBox <- bbox(SpatialPoints(as.data.frame(cbind(lons, lats))))
label_cols <- c(brewer.pal(11, "Spectral"))

## Mark the affected districts
ob_district_name_list <- c()
ob_district_shp_list <- c()
ob_countries <- unique(ob_df$Country)
for (country_name in ob_countries) {
    shp_file_path <- unique(ob_df[ob_df$Country == country_name,]$FilePath)
    shp = readShapePoly(shp_file_path)
    ob_districts_raw = ob_areas[ob_areas$Country==country_name,]$District
    adm_level_to_use = shapeFiles[shapeFiles$Country==country_name,]$ADM_Level
    if(adm_level_to_use==2){ 
      matched_indices = amatch(ob_districts_raw,shp$ADM2,maxDist=2)
    } else {
      matched_indices = amatch(ob_districts_raw,shp$ADM1,maxDist=2)
    }
    matched_indices <- matched_indices[!is.na(matched_indices)] # Remove NAs
    ob_districts = shp$ADM2[matched_indices]   
    for (ob_district in ob_districts) {
        shp_ob_district = shp[shp$ADM2==ob_district,]
        ob_district_shp_list <- c(ob_district_shp_list, shp_ob_district)
        legend_str = paste(ob_district, " (",country_name,")", sep="")
        ob_district_name_list = c(ob_district_name_list, legend_str)
    }
}

## Plot the container map (Africa)
xlim <- c(bBox[1,1]  , bBox[1,2] )
ylim <- c(bBox[2,1] , bBox[2,2] )
plot(shp_Africa,axes=TRUE,xlim=xlim,ylim=ylim)

## Plot the affected districts
i = 0
for (shp_ob_district in ob_district_shp_list) {
    i = i + 1
    plot(shp_ob_district,add=TRUE,col=label_cols[i])
}

## Add the title and legend
title(main="Districts Affected By Ebola")
labels <- ob_district_name_list
legend("bottomleft", NULL, labels, fill=label_cols, density, bty="n", cex=.8)