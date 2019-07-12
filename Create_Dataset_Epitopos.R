#UTILS LIBRARIES, FOR EXTRATIFYING THE TRAIN AND TEST, AND SAVING THE DATA IN
#ZIP FOR THE MOVELETS

library(data.table)
library(ggplot2)
library(stringr)
library(Hmisc)

#UTILS FUNCTIONS, FOR EXTRATIFYING THE TRAIN AND TEST, AND SAVING THE DATA IN
#ZIP FOR THE MOVELETS

from.data.create.zip.file <- function (dt.list, mainDir, subDir, ext, zipfilename){
    
    dir.create(file.path(mainDir, subDir))
    n <- length(dt.list)
    lapply(1:length(dt.list),function(xi){
        x <- dt.list[[xi]]
        filename <- paste0(str_pad(xi, nchar(n), pad = "0")," ","s",unique(x$tid)," ","c",unique(x$label))
        filepath <- paste0(mainDir, "/", subDir, "/", filename, ext)
        cat(filepath,"\n")
        write.table(subset(x, select = -c(tid,label)), file = filepath, quote=FALSE, sep = ",", row.names = FALSE, col.names = FALSE)
    })
    
    if ( file.exists(file.path(mainDir,zipfilename)) )  
        unlink( x = file.path(mainDir,zipfilename), force = TRUE )
    
    program <- "\"C:/Program Files/7-Zip/7z.exe\" a "
    cmd <- paste0( program, " ", file.path(mainDir,zipfilename), " ", file.path(mainDir, subDir), "/*" )
    
    try(system(cmd, intern = TRUE))
    
    if (dir.exists(file.path(mainDir, subDir)))
        unlink( file.path(mainDir, subDir), recursive = TRUE, force = TRUE )
}

dt.to.list <- function (dt){
    lapply(unique(dt$tid), function(x){
        subset(dt, tid == x)
    })
}


get.stratified <- function (tids, pTrain){
    
    set.seed(1)
    
    tids.train <- unlist(lapply(unique(tids$label), function(x){
        y <- subset(tids, label == x)
        sample(y$tid, round(pTrain * length(y$tid)) )
    }))
    
    tids.test <- subset(tids, tid %nin% tids.train)$tid
    
    list(train = tids.train, test = tids.test)
}

################################################################
################################################################

#READ THE CSV WITH THE DATASET OF EPITOPOS

#Criar um aquivo dos epitopos juntando os bons com ruins
#Separar 70% para treino 30% para teste
#extratificar o item acima
#rodar os master pivots encima do conjunto de treino
#Jogar para dentro de um algorítimo SVM, uma linha vai representar os movelets tirados.
#Gerar o modelo e classificar com o teste
#Analisar a acurácia

setwd("./")
dt<-fread("ep_vdd.csv")
dir()
all.tids <- unique(subset(dt, select=c(tid,label) ))

tids <- get.stratified(all.tids,0.7)

