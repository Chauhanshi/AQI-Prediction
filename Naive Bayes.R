air<- read.csv(file.choose())

air$level<-0

air <- air[-c(185), ]



for(i in 1:1092){
  if(air[i,9]<=50){
    air[i,10]<-"Good"
  }
  if(air[i,9]>50 & air[i,9]<=100 ){
    air[i,10]<-"Good"
  }
  if(air[i,9]>100 & air[i,9]<=150 ){
    air[i,10]<-"Unhealthy for SG"
  }
  if(air[i,9]>150 & air[i,9]<=200 ){
    air[i,10]<-"Unhealthy"
  }
  if(air[i,9]>200 & air[i,9]<=300 ){
    air[i,10]<-"Very Unhealthy"
  }
  if(air[i,9]>300 & air[i,9]<=500 ){
    air[i,10]<-"Hazardous"
  }
  
}

air$level<- as.factor(air$level)

smp_size <- floor(0.75 * nrow(air))
set.seed(123)
train_ind <- sample(seq_len(nrow(air)), size = smp_size)
training <-air[train_ind, ]
testing <- air[-train_ind, ]

#create objects x which holds the predictor variables and y which holds the response variables
remove(x)
x = training[,-10]
y = training$level

x$PM.2.5<- NULL


install.packages("e1071")
library(e1071)

model<-naiveBayes(x,y)
model

library(caret)

Predict <- predict(model,newdata = testing )
confusionMatrix(Predict, testing$level )
