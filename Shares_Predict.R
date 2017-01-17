library("party")
library("rpart")
library("tree")
library("randomForest")
library("miscTools")
library("caret")
library("class")
library("ROCR")
library("pROC")
library("e1071")
library("C50") #C5.0
library("rpart") #R part
library("rpart.plot")
library("rattle")
library("RColorBrewer")
library("ggplot2")
library("kernlab")
library("e1071")
library('kknn')
library('stats')
library('glmnet')
library('lars')

#install.packages('lars')
# for normalization
##################Functions################
#Function 1
# my_normalization: Scaled the features.
my_normalization <- function(news){
  total_column <- names(news)
  no_need_normal <- c(
    "url", "timedelta",
    "data_channel_is_lifestyle",
    "data_channel_is_entertainment",
    "data_channel_is_bus",
    "data_channel_is_world",
    "data_channel_is_socmed",
    "data_channel_is_tech",
    "weekday_is_monday",
    "weekday_is_tuesday",
    "weekday_is_wednesday",
    "weekday_is_thursday",
    "weekday_is_friday",
    "weekday_is_saturday",
    "weekday_is_sunday",
    "is_weekend",
    "shares"
  )
  need_normal <- setdiff(total_column,no_need_normal)
  my_sd <- Map(sd, news[,need_normal])
  my_mean <- Map(mean, news[,need_normal])
  news[,need_normal] <- (news[,need_normal] - my_mean)/my_sd
 
  return(news)
}
#Function 2
#Plot the combination ROC curves.
my_plot <- function(real,combine){
	ROCCurve<-par(pty = "s")
	count = 1
	for(sub in combine){
		flag <- ifelse(count>1,TRUE,FALSE)
		plot(performance(prediction(as.numeric(sub[5:length(sub)]),as.numeric(real)),'tpr','fpr'),
		col=as.numeric(sub[4]), lwd=3,add=flag)	
		text(as.numeric(sub[1]),as.numeric(sub[2]),sub[3],col=as.numeric(sub[4]))
		count = count + 1
		}
}
#############Load File################################
news=read.csv('OnlineNewsPopularity.csv')
news=news[!news$n_unique_tokens==701,]
#####################Drop Boruta#######
Boruta <- c("n_tokens_content","n_unique_tokens","n_non_stop_words","n_non_stop_unique_tokens",
	"num_hrefs","num_self_hrefs","num_imgs","average_token_length","num_keywords","data_channel_is_entertainment",
	"data_channel_is_bus","data_channel_is_socmed","data_channel_is_tech","data_channel_is_world","kw_min_min",
	"kw_max_min","kw_avg_min","kw_min_max","kw_max_max","kw_avg_max","kw_min_avg","kw_max_avg","kw_avg_avg",
	"self_reference_min_shares","self_reference_max_shares","self_reference_avg_sharess","is_weekend","LDA_00",                       
	"LDA_01","LDA_02","LDA_03","LDA_04","global_subjectivity","global_sentiment_polarity","global_rate_positive_words",
	"global_rate_negative_words","rate_positive_words","rate_negative_words","avg_positive_polarity","max_positive_polarity",
	"avg_negative_polarity","min_negative_polarity","shares") 
######################Data Clean #########
news = my_normalization(news)
non_predict = c(1,2)
news = news[,-non_predict]
# Dataset for classification
news$shares <- as.factor(ifelse(news$shares > 1400,1,0))
#set random situation
#set.seed(100)
# Select traning data and prediction data
ind<-sample(2,nrow(news),replace=TRUE,prob=c(0.7,0.3))
#####################PCA#############################
prin_comp <- prcomp(news[ind==1,-59], scale. = T)
std_dev <- prin_comp$sdev
pr_var <- std_dev^2
prop_varex <- pr_var/sum(pr_var)
#cumulative scree plot
plot(cumsum(prop_varex), xlab = "Principal Component",
              ylab = "Cumulative Proportion of Variance Explained",
              type = "b")
#add a training set with principal components
train.data <- predict(prin_comp, newdata = news[ind==1,])
#########Select 40 Components#############
train.data <- data.frame(train.data[,1:40])
test.data <- predict(prin_comp, newdata = news[ind==2,])
test.data <- data.frame(test.data[,1:40])

train.data <- cbind(train.data,news[ind==1,]$shares)
colnames(train.data)[41] <- "shares"
test.data <- cbind(test.data,news[ind==2,]$shares)
colnames(test.data)[41] <- "shares"
##################################################################
################################KNN######################
newscla.knn <- kknn(shares ~.,news[ind==1,],news[ind==2,],k=300)
newscla.knn_Boru <- kknn(shares ~.,news[ind==1,Boruta],news[ind==2,Boruta],k=300)
newscla.knn_pca <- kknn(train.data$shares ~.,train.data,test.data,k=300)
# Confusion matrix
confusionMatrix(newscla.knn$fitted.values, news[ind==2,]$shares)
confusionMatrix(newscla.knn_pca$fitted.values, news[ind==2,]$shares)
confusionMatrix(newscla.knn_Boru$fitted.values, news[ind==2,]$shares)
############Plot###########
KNN <- c(0.15,0.6,"KNN",1,newscla.knn$prob[,2])
KNN_Boru <- c(0.45,0.2,"KNN_Boruta",3,newscla.knn_Boru$prob[,2])
KNN_pca <- c(0.55,0.6,"KNN_PCA",2,newscla.knn_pca$prob[,2])
combine <- matrix(list(),nrow=3,ncol=1)
combine[[1]] <- KNN
combine[[2]] <- KNN_pca
combine[[3]] <- KNN_Boru
my_plot(news[ind==2,]$shares,combine)
###########################################################
##################CART#####################################
##Training
newscla.cart<-rpart(shares ~.,news[ind==1,],method='class')
newscla.cart_Boru<-rpart(shares ~.,news[ind==1,Boruta],method='class')
newscla.cart_pca<-rpart(train.data$shares ~.,train.data,method='class')
# Plot tree
#fancyRpartPlot(newscla.cart)
#Prediction
newscla.cart.pred<-predict( newscla.cart,news[ind==2,] ,type="class")
newscla.cart.pred_pca<-predict( newscla.cart_pca,test.data ,type="class")
newscla.cart.pred_Boru<-predict( newscla.cart_Boru,news[ind==2,Boruta] ,type="class")

# Confusion matrix
confusionMatrix(newscla.cart.pred, news[ind==2,]$shares)
confusionMatrix(newscla.cart.pred_pca, news[ind==2,]$shares)
confusionMatrix(newscla.cart.pred_Boru, news[ind==2,]$shares)

#Predict probability
newscla.cart.prob<-predict( newscla.cart,news[ind==2,] ,type="prob")
newscla.cart.prob_Boru<-predict( newscla.cart_Boru,news[ind==2,Boruta] ,type="prob")
newscla.cart.prob_pca<-predict(newscla.cart_pca,test.data ,type="prob")
############Plot###########
CART <- c(0.15,0.6,"CART",1,newscla.cart.prob[,2])
CART_pca <- c(0.55,0.6,"CART_PCA",2,newscla.cart.prob_pca[,2])
CART_Boru <- c(0.45,0.2,"CART_Boruta",3,newscla.cart.prob_Boru[,2])
combine <- matrix(list(),nrow=3,ncol=1)
combine[[1]] <- CART
combine[[2]] <- CART_pca
combine[[3]] <- CART_Boru
my_plot(news[ind==2,]$shares,combine)
####################################################
######################NaiveBayes####################
##Training
newscla.Bayes<-naiveBayes(shares ~.,data = news[ind==1,])
newscla.Bayes_Boru<-naiveBayes(shares ~.,data = news[ind==1,Boruta])
newscla.Bayes_pca<-naiveBayes(train.data$shares ~.,data = train.data)
#Prediction
newscla.Bayes.pred = predict(newscla.Bayes,news[ind==2,],type="class")
newscla.Bayes.pred_pca = predict(newscla.Bayes_pca,test.data,type="class")
newscla.Bayes.pred_Boru = predict(newscla.Bayes_Boru,news[ind==2,Boruta],type="class")

newscla.Bayes.prob<-predict(newscla.Bayes,news[ind==2,],type="raw")
newscla.Bayes.prob_Boru<-predict(newscla.Bayes_Boru,news[ind==2,Boruta],type="raw")
newscla.Bayes.prob_pca<-predict(newscla.Bayes_pca,test.data ,type="raw")
# Confusion matrix
confusionMatrix(newscla.Bayes.pred, news[ind==2,]$shares)
confusionMatrix(newscla.Bayes.pred_pca, news[ind==2,]$shares)
confusionMatrix(newscla.Bayes.pred_Boru, news[ind==2,]$shares)
############Plot###########
NB <- c(0.15,0.6,"NB",1,newscla.Bayes.prob[,2])
NB_pca <- c(0.55,0.6,"NB_PCA",2,newscla.Bayes.prob_pca[,2])
NB_Boru <- c(0.45,0.2,"NB_Boruta",3,newscla.Bayes.prob_Boru[,2])
combine <- matrix(list(),nrow=3,ncol=1)
combine[[1]] <- NB
combine[[2]] <- NB_pca
combine[[3]] <- NB_Boru
my_plot(news[ind==2,]$shares,combine)

#######################################################
####################RandomForest#######################
#Training
newscla.rf_pca<-randomForest(train.data$shares ~.,train.data,ntree=100)
newscla.rf<-randomForest(shares ~.,news[ind==1,],ntree=100)
newscla.rf_Boru<-randomForest(shares ~.,news[ind==1,Boruta],ntree=100)
plot(newscla.rf,main='Error v.s. Number of Trees')

#predict
newscla.rf.pred<-predict( newscla.rf,news[ind==2,], type="class")
newscla.rf.pred_pca<-predict( newscla.rf_pca,test.data, type="class")
newscla.rf.pred_Boru<-predict( newscla.rf_Boru,news[ind==2,Boruta], type="class")

newscla.rf.prob_pca<-predict( newscla.rf_pca,test.data, type="prob")
newscla.rf.prob<-predict( newscla.rf,news[ind==2,], type="prob")
newscla.rf.prob_Boru<-predict( newscla.rf_Boru,news[ind==2,Boruta], type="prob")
# Confusion matrix
confusionMatrix(newscla.rf.pred, news[ind==2,]$shares)
confusionMatrix(newscla.rf.pred_pca, news[ind==2,]$shares)
confusionMatrix(newscla.rf.pred_Boru, news[ind==2,]$shares)
############Plot###########
RF <- c(0.15,0.6,"RF",1,newscla.rf.prob[,2])
RF_pca <- c(0.55,0.6,"RF_PCA",2,newscla.rf.prob_pca[,2])
RF_Boru <- c(0.45,0.2,"RF_Boruta",3,newscla.rf.prob_Boru[,2])
combine <- matrix(list(),nrow=3,ncol=1)
combine[[1]] <- RF
combine[[2]] <- RF_pca
combine[[3]] <- RF_Boru
my_plot(news[ind==2,]$shares,combine)
#########################################################
###################Logistic Regression###################
#Training
model <- glm(shares ~.,family=binomial(link='logit'),data=news[ind==1,])
model_Boruta <- glm(shares ~.,family=binomial(link='logit'),data=news[ind==1,Boruta])
model_pca <- glm(train.data$shares ~.,family=binomial(link='logit'),data=train.data)

#Prediction
newscla.LR.prob <- predict(model,newdata=news[ind==2,],type='response')
newscla.LR.prob_Boru <- predict(model_Boruta,newdata=news[ind==2,Boruta],type='response')
newscla.LR.prob_pca <- predict(model_pca,newdata=test.data,type='response')

newscla.LR.preb <- ifelse(newscla.LR.prob > 0.5,1,0)
newscla.LR.preb_pca <- ifelse(newscla.LR.prob_pca > 0.5,1,0)
newscla.LR.preb_Boru <- ifelse(newscla.LR.prob_Boru > 0.5,1,0)
# Confusion matrix
confusionMatrix(newscla.LR.preb, news[ind==2,]$shares)
confusionMatrix(newscla.LR.preb_pca, news[ind==2,]$shares)
confusionMatrix(newscla.LR.preb_Boru, news[ind==2,]$shares)
############Plot###########
LR <- c(0.15,0.6,"LogisticRegression",1,newscla.LR.prob)
LR_pca <- c(0.55,0.6,"LogisticRegression_PCA",2,newscla.LR.prob_pca)
LR_Boru <- c(0.45,0.2,"LogisticRegression_Boruta",3,newscla.LR.prob_Boru)
combine <- matrix(list(),nrow=3,ncol=1)
combine[[1]] <- LR
combine[[2]] <- LR_pca
combine[[3]] <- LR_Boru
my_plot(news[ind==2,]$shares,combine)

#####################################################################################################################################################################################
################Plot the combination of all the classification model based on oringinal data set########
KNN <- c(0.45,0.6,"KNN",1,newscla.knn$prob[,2])
CART <- c(0.3,0.4,"CART",2,newscla.cart.prob[,2])
NB <- c(0.11,0.5,"Naive Bayes",3,newscla.Bayes.prob[,2])
RF <- c(0.3,0.8,"Random Forest",4,newscla.rf.prob[,2])
LR <- c(0.21,0.7,"Logistic Regression",5,newscla.LR.prob)

combine <- matrix(list(),nrow=5,ncol=1)
combine[[1]] <- KNN
combine[[2]] <- CART
combine[[3]] <- NB
combine[[4]] <- RF
combine[[5]] <- LR
my_plot(news[ind==2,]$shares,combine)

################Plot the combination of all the classification model based on PCA dimension reduction########
KNN_pca <- c(0.45,0.6,"KNN_PCA",1,newscla.knn_pca$prob[,2])
CART_pca <- c(0.3,0.4,"CART_PCA",2,newscla.cart.prob_pca[,2])
NB_pca <- c(0.11,0.5,"Naive Bayes_PCA",3,newscla.Bayes.prob_pca[,2])
RF_pca <- c(0.3,0.8,"Random Forest PCA",4,newscla.rf.prob_pca[,2])
LR_pca <- c(0.21,0.7,"Logistic Regression PCA",5,newscla.LR.prob_pca)

combine <- matrix(list(),nrow=5,ncol=1)
combine[[1]] <- KNN_pca
combine[[2]] <- CART_pca
combine[[3]] <- NB_pca
combine[[4]] <- RF_pca
combine[[5]] <- LR_pca
my_plot(news[ind==2,]$shares,combine)
########################
################Plot the combination of all the classification model based on Boruta feature selection########
KNN_Boru <- c(0.45,0.6,"KNN Boruta",1,newscla.knn_Boru$prob[,2])
CART_Boru <- c(0.3,0.4,"CART Boruta",2,newscla.cart.prob_Boru[,2])
NB_Boru <- c(0.11,0.5,"Naive Bayes Boruta",3,newscla.Bayes.prob_Boru[,2])
RF_Boru <- c(0.3,0.8,"Random Forest Boruta",4,newscla.rf.prob_Boru[,2])
LR_Boru <- c(0.21,0.7,"Logistic Regression Boruta",5,newscla.LR.prob_Boru)

combine <- matrix(list(),nrow=5,ncol=1)
combine[[1]] <- KNN_Boru
combine[[2]] <- CART_Boru
combine[[3]] <- NB_Boru
combine[[4]] <- RF_Boru
combine[[5]] <- LR_Boru
my_plot(news[ind==2,]$shares,combine)
########################