library(ggplot2)
cairo_pdf("~/project/nb-al/al/graphics/fig1.pdf", height=8, width=8)

data <- read.csv("data/rand1.dat", header = FALSE)
data <- rbind(data, read.csv("data/rand2.dat",header = FALSE))
data <- rbind(data, read.csv("data/rand3.dat",header = FALSE))
data <- rbind(data, read.csv("data/rand4.dat",header = FALSE))
data <- rbind(data, read.csv("data/rand5.dat",header = FALSE))
data <- rbind(data, read.csv("data/rand6.dat",header = FALSE))
data <- rbind(data, read.csv("data/rand7.dat",header = FALSE))
data <- rbind(data, read.csv("data/rand8.dat",header = FALSE))
data <- rbind(data, read.csv("data/rand9.dat",header = FALSE))
data <- rbind(data, read.csv("data/rand10.dat",header = FALSE))

#attach(data)
# plot(data[,0],data[,1],type="n", ylim=c(0,1),pch=4, xlab="Examples",ylab="Micro-aver Accuracy")

par(mfrow=c(2,2), mar=c(5,4,1,1))

plot(data[,1],data[,2], pch=1, col="#AAAAAA", cex=0.25, ylab="Accuracy", xlab="Number of examples")
lines(lowess(data[,1],data[,2]), lwd=2)

plot(data[,1],data[,3], pch=1, col="#AAAAAA", cex=0.25, ylab="Micro-precision", xlab="Number of examples")
lines(lowess(data[,1],data[,3]), lwd=2)

plot(data[,1],data[,4], pch=1, col="#AAAAAA", cex=0.25, ylab="Micro-recall", xlab="Number of examples")
lines(lowess(data[,1],data[,4]), lwd=2)

plot(data[,1],data[,5], pch=1, col="#AAAAAA", cex=0.25, ylab="Micro-F1 score", xlab="Number of examples")
lines(lowess(data[,1],data[,5]), lwd=2)

dev.off()


qplot(data[,1],data[,2])
ggsave("~/project/nb-al/al/graphics/ggplot1.pdf", device=cairo_pdf)

dev.off()
