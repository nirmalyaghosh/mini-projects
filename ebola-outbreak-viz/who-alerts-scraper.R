library(plyr)
library(RCurl)
library(XML)


# Function to extract fields from a given node (one per alert)  
prfx <- "http://www.who.int"
read_alert_node <- function(node) {
    a_date <- xpathSApply(node, "a", xmlValue)
    a_url <- paste0(prfx, xpathSApply(node, 'a', xmlGetAttr, 'href'))
    a_hdln <- xpathSApply(node, "span", xmlValue)
    df <- data.frame(matrix(unlist(c(a_date, a_url, a_hdln)), nrow=1, byrow=T))
    return(df)
}


reportProgress <- function(mssg) {
        cat(format(Sys.time(), "%H:%M:%S"),mssg,"\n")
        flush.console()
}


scrape_list <- function() {
    url = paste("http://www.who.int/csr/don/archive/year/",year(Sys.Date()),
                "/en/",sep="")
    useragent <- c('User-Agent' = "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.114 Safari/537.36")
    
    # Scrape the main page listing all alerts
    reportProgress("Acquiring the alerts")
    html.raw <- getURL(url, followlocation = TRUE, httpheader = useragent)
    doc = htmlTreeParse(url, useInternalNodes = T)
    alert_nodes = getNodeSet(doc, '//*[@id="content"]/div/div[1]/ul/li')
    
    # Read all nodes and return a data frame
    df = ldply(alert_nodes, read_alert_node)
    df$X4 <- NULL # Drop the 'X4' column, since not required
    colnames(df) <- c("Date", "URL", "Headline") # Rename the columns
    write.table(df,"WHO-Alerts.txt.txt", row.names=F, quote=F, sep="\t") # TODO remove later
    reportProgress(paste(dim(df)[1],"alert(s) acquired"))
    
    # Save for future use
    save(df, file=filename)
    dim(df)
    return(df)
}


 # Main
filename <- "WHO-Alerts.Rda"
rm(df)
if (file.exists(filename)) {
  reportProgress(paste(filename,"exists"))
  load(filename)
  if(is.null(df) || dim(df)[1] < 1) { df <- scrape_list() }
} else { 
  reportProgress(paste(filename,"does not exist"))
  df <- scrape_list()
}