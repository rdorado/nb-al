library(ggplot2)
cairo_pdf("~/project/nb-al/al/graphics/fig3.pdf", height=8, width=8)

data <- read.csv("~/project/nb-al/al/graphics/data/outfile.dat", header = FALSE)
attach(data)

plot(data[,1],data[,2],pch=0,cex=0.6, ylab="Accuracy", xlab="Number of examples")
points(data[,1],data[,6],pch=1,cex=0.6,col="#1E90FF")

#par(mfrow=c(2,2), mar=c(5,4,1,1))

#plot(data[,1],data[,2], pch=1, col="#AAAAAA", cex=0.25, ylab="Accuracy", xlab="Number of examples")
#lines(lowess(data[,1],data[,2]), lwd=2)

#plot(data[,1],data[,3], pch=1, col="#AAAAAA", cex=0.25, ylab="Micro-precision", xlab="Number of examples")
#lines(lowess(data[,1],data[,3]), lwd=2)

#plot(data[,1],data[,4], pch=1, col="#AAAAAA", cex=0.25, ylab="Micro-recall", xlab="Number of examples")
#lines(lowess(data[,1],data[,4]), lwd=2)

#plot(data[,1],data[,5], pch=1, col="#AAAAAA", cex=0.25, ylab="Micro-F1 score", xlab="Number of examples")
#lines(lowess(data[,1],data[,5]), lwd=2)

dev.off()


qplot(data[,1],data[,2],colour="#CC0000")
qplot(data[,1],data[,6],colour="#000099")
ggsave("~/project/nb-al/al/graphics/ggplot3.pdf", device=cairo_pdf)

dev.off()
