# Loading raw data:

train <- read.csv("train.csv", header = T)
test <- read.csv("test.csv", header = T)
str(train)
str(test)


# the test dataset is missing one variable (survived)

# So this code add this variable to the dataset:
#test$Survived <- "NA"
test.survived <- data.frame(Survived = rep("NA", nrow(test)), test[,])

# combine both data set
data.comb <- rbind(train,test.survived)

str(data.comb)


# Convert variables (Pclass & Survived) to factors:
data.comb$Survived <- as.factor(data.comb$Survived)
data.comb$Pclass <- as.factor(data.comb$Pclass)


# Predictors selection:

# Survival variable table:
table(data.comb$Survived)

# 549 not survived
# 342 survived
# 418 unknow ( the goal is to predict the unknown)

# Let's examine the Pclass variable
# distribution of class:
table(data.comb$Pclass)

# Let's visualize
counts <- table( train$Survived, train$Pclass)

barplot(height = counts, col = c('red','light blue'), ylim = c(0,600),
        xlab = "Passenger Class", ylab = "Frequecny", main = "Survival Among Passenger Class")
legend(x = "topleft", legend = c("Perish","Survived"),
       col = c("red","light blue"),
       pch = 15, pt.cex = 2, cex = 1)

# First class people have higher survival numbers
# conclusion: Include the variable

# examine the names variable:
data.com

data.comb$Name <- as.character(data.comb$Name)
str(data.comb)
head(data.comb$Name, 50)
# the name variable seems unuesful, but we can see the titles like Miss & Mr.
# are used for every name, thus let's see if there is a pettern 

# title extraction function:
extractTitle <- function(Name){
  Name <- as.character(Name)
  
  if (length(grep("Miss.", Name)) > 0) {
    return ("Miss.")
  } else if (length(grep("Master.", Name)) > 0) {
    return ("Master.")
  } else if (length(grep("Mrs.", Name)) > 0) {
    return ("Mrs.")
  } else if (length(grep("Mr.", Name)) > 0) {
    return ("Mr.")
  } else {
    return ("Other")
  }
}

titles <- NULL
for (i in 1:nrow(data.comb)) {
  titles <- c(titles, extractTitle(data.comb[i,"Name"]))
}
data.comb$title <- as.factor(titles)

# lets graph the title on only the train part of the data
library(ggplot2)

ggplot(data = data.comb[1:891,], aes( x = title, fill = Survived))+
  geom_bar(width = 0.5)+
  xlab ("Title")+
  ylab ("Frequency")+
  labs(fill = "Survived")+
  ggtitle("Sruvival & Name Title")

# there seems to be a pattern as those with Mr. title were more
# likely to not survive

# with the pclass
ggplot(data = data.comb[1:891,], aes( x = title, fill = Survived))+
  geom_bar(width = 0.5)+
  facet_wrap(~Pclass)+
  xlab ("Title")+
  ylab ("Frequency")+
  labs(fill = "Survived")+
  ggtitle("Sruvival in Terms of Name Title and Passenger Class")

# Conclsuion: include thr variable title 



# let's lookk at the distribution of sex:
table(data.comb$Sex)

# let's visualize 3 variables sex, pclass & sruvival

ggplot(data = data.comb[1:891,], aes( x = Sex, fill = Survived))+
  geom_bar(width = .5)+
  xlab("Sex")+
  ylab("Frequency")+
  labs(fill = "Survived")+
  ggtitle("Sex and Survival")
# male survive more 


ggplot(data = data.comb[1:891,], aes( x = Sex, fill = Survived))+
  geom_bar(width = .5)+
  facet_wrap(~Pclass)+
  xlab("Sex")+
  ylab("Frequency")+
  labs(fill = "Survived")+
  ggtitle("Sex and Survival among Passnger Class")


# Conclsuion: include thr variable sex


# let's look at age:
summary(data.comb$Age)
# 263 of age values are missing which is a lot

# will infer the age values ( with the mean)

age.median <- median(data.comb$Age, na.rm = T)

data.comb[is.na(data.comb$Age),"Age" ] <- age.median
summary(data.comb$Age)


ggplot(data = data.comb[1:891,], aes( x = Age,  fill = Survived))+
  geom_histogram(binwidth = 10)+
  xlab("Age")+
  ylab("Frequency")

# Will include the age variable, but it will influcen the final 
# predicitoins as the missing values are estimated with the mean 


# moving to sib  variable:
summary(data.comb$SibSp)

# can we treat it as a factor
length(unique(data.comb$SibSp))

data.comb$SibSp <- as.factor(data.comb$SibSp)

ggplot(data = data.comb[1:891,], aes(x=SibSp, fill=Survived))+
  geom_bar()+
  xlab("Sib")+
  ylab("Frequency")

# It seems useful and will include it
#

#looking at parah variable:

data.comb$Parch <- as.factor(data.comb$Parch)
levels(data.comb$Parch)

ggplot(data.comb[1:891,], aes(x = Parch, fill = Survived))+
  geom_bar()+
  xlab("Parah")+
  ylab("Frequency")

# it seems useflul too


# let's look at the Ticket varaible:
str(data.comb$Ticket)

# better to change the ticket variable to a string instead of factor:
data.comb$Ticket <- as.character(data.comb$Ticket)
data.comb$Ticket[1:10]

# It seems it's not useful


# Next lets look fare:
str(data.comb$Fare)
summary(data.comb$Fare)

# lets visulaize:
ggplot(data = data.comb, aes(x =Fare))+
  geom_histogram(binwidth = 5)+
  xlab("Fare") +
  ylab("Total Count") +
  ggtitle("Fare Histogram")

# there is one missing values and we will estimate it with the mean 

fare.mean <- mean(data.comb$Fare, na.rm = T)

data.comb[is.na(data.comb$Fare),"Fare" ] <- fare.mean

summary(data.comb$Fare)

# lets look ar it with survival
ggplot(data = data.comb[1:891,], aes(x =Fare, fill = Survived))+
  geom_histogram(binwidth = 5)+
  xlab("Fare") +
  ylab("Total Count") +
  ggtitle("Fare Histogram")

# will include it 


# Now let's look at cabin:
str(data.comb$Cabin)

# switch it to string
data.comb$Cabin <- as.character(data.comb$Cabin)

data.comb$Cabin

# Replacing empty with U
data.comb[which(data.comb$Cabin == ""), "Cabin"] <- "E"
data.comb$Cabin

# lets look at the first letter:
cabin.firt.char <- as.factor(substr(data.comb$Cabin, 1,1))
str(cabin.firt.char)
levels(cabin.firt.char)

# add it to the combined data:
data.comb$cabin.firt.char <- cabin.firt.char

ggplot(data = data.comb[1:891,], aes(x =cabin.firt.char, fill = Survived))+
  geom_bar()+
  xlab("cabin.firt.char") +
  ylab("Total Count") 


ggplot(data = data.comb[1:891,], aes(x =cabin.firt.char, fill = Survived))+
  geom_bar()+
  facet_wrap(~Pclass)+
  xlab("cabin.firt.char") +
  ylab("Total Count") 

# Will include it as it seems useful

# finally embarked :
str(data.comb$Embarked)

summary(data.comb$Embarked)

# 2 are misisng and we will add them to them to the S level

data.comb[which(data.comb$Embarked == ""), "Embarked"] <- "S"


# Plot data for analysis
ggplot(data.comb[1:891,], aes(x = Embarked, fill = Survived)) +
  geom_bar() +
  xlab("embarked") +
  ylab("Total Count")+
  labs(fill = "Survived") 

# Overall Conclusion:

# Iclude all variables expect for Ticket and Name.
# will replace name with title
colnames(data.comb)
data.comb <- data.comb[,-c(1,4,9,11)]
str(data.comb)

################
# Exploratory modeling:
##################

## LDA:

# Model 1

train.set <- data.comb[1:891, c("Survived", "Pclass", "title")]

library(MASS)
ldamod = lda(Survived ~ ., data= train.set)

table(Predicted=predict(ldamod)$class)
# accuracyrate:
(acc1 <- 1-mean(predict(ldamod)$class !=  train.set$Survived))

# Achieved 0.7878788 accuracy with training set

# Model 2

train.set2 <- data.comb[1:891, c("Survived", "Pclass", "title", "Age")]
ldamod2 = lda(Survived ~ ., data= train.set)

table(Predicted2=predict(ldamod2)$class)
# accuracy rate:
(acc2 <- 1-mean(predict(ldamod2)$class !=  train.set$Survived))

# No improvement with Age ( So will not use it )



# Model 3
train.set3 <- data.comb[1:891, c("Survived", "Pclass", "title","SibSp")]

ldamod3 = lda(Survived ~ ., data= train.set3)

table(Predicted=predict(ldamod3)$class)
# accuracy rate:
(acc3 <-1-mean(predict(ldamod3)$class !=  train.set$Survived))


# Achieved 0.8181818 accuracy with training set (sharp improvement)
test.submit.df <- data.comb[892:1309, ]

lda3.test <- predict(ldamod3, newdata =test.submit.df)$class

submit.df <- data.frame(PassengerId = rep(892:1309) , Survived = lda3.test)
head(submit.df)

write.csv(submit.df, file = "submit.lda3.csv",row.names = F)

# Achieved 0.78468 accuracy on the testing set!

# Model 4
head(data.comb)
train.set4 <- data.comb[1:891, c("Survived", "Pclass", "title","SibSp", "Fare" )]

ldamod4 = lda(Survived ~ ., data= train.set4)

table(Predicted=predict(ldamod4)$class)
# accuracy rate:
(acc4 <- 1-mean(predict(ldamod4)$class !=  train.set$Survived))


# Achieved 0.8204265 accuracy with the training set.

lda4.test <- predict(ldamod4, newdata =test.submit.df)$class
submit.df <- data.frame(PassengerId = rep(892:1309) , Survived = lda4.test)
head(submit.df)

write.csv(submit.df, file = "submit.lda4.csv",row.names = F)

# Achieved 0.78468 accuracy on the testing set! (same as model 3)



# Model 5
head(data.comb)
train.set5 <- data.comb[1:891, c("Survived", "Pclass", "title","SibSp", "Fare",  "Sex" )]

ldamod5 = lda(Survived ~ ., data= train.set5)

table(Predicted=predict(ldamod5)$class)
# accuracy rate:
(acc5 <- 1-mean(predict(ldamod5)$class !=  train.set$Survived))

# Achieved 0.8260382 accuracy with the training set (little improvement from model 4)


# Model 6:

head(data.comb)
train.set6 <- data.comb[1:891, c("Survived", "Pclass", "title","SibSp", "Fare",  "Sex", "Embarked")]

ldamod6 = lda(Survived ~ ., data= train.set6)

table(Predicted=predict(ldamod6)$class)
# accuracy rate:
(acc6 <- 1-mean(predict(ldamod6)$class !=  train.set$Survived))

# Achieved 0.8260382 accuracy with the training set (no improvement from model 5)
# will not use "Embarked"


# Model 7:

head(data.comb)
train.set7<- data.comb[1:891, c("Survived", "Pclass", "title","SibSp", "Fare",  "Sex", "cabin.firt.char")]

ldamod7 = lda(Survived ~ ., data= train.set7)

table(Predicted=predict(ldamod7)$class)
# accuracy rate:
(acc7 <- 1-mean(predict(ldamod7)$class !=  train.set$Survived))

# Achieved 0.8249158 accuracy with the training set (no improvement from model 5 & 6)

##### Conclusion: Model 5 perform the best on the training set with less predictors ########


train.set5 <- data.comb[1:891, c("Survived", "Pclass", "title","SibSp", "Fare",  "Sex" )]

ldamod5 = lda(Survived ~ ., data= train.set5)

table(Predicted=predict(ldamod5)$class)
# accuracy rate:
1-mean(predict(ldamod5)$class !=  train.set$Survived)

# let's see the prediction error on the testing set:
lda5.test <- predict(ldamod5, newdata =test.submit.df)$class
submit.df <- data.frame(PassengerId = rep(892:1309) , Survived = lda5.test)
head(submit.df)

write.csv(submit.df, file = "submit.lda5.csv",row.names = F)


# Achieved 0.78468 accuracy on the testing 

# Thus, the maximum accuracy achieved with Linear Discriminate Analysis on 
# the testing set from Kaggle is 0.78468w hich is also achieved from other models 3 & 4

# to plot the accuracy of the training set
acc <- rbind(acc1,acc2,acc3,acc4,acc5,acc6,acc7)
plot(acc, type = "b", 
     ylab = "Accuracy",
     xlab = "Model",
     main = "The Accuracy of Different LDA Models on Training Set")






###
### Logistic Regreesion ####
###

# first model
train.set <- data.comb[1:891, c("Survived", "Pclass", "title")]
logit1 = glm(Survived ~ ., data = train.set, family = "binomial")
summary(logit1)

# Predicted probabilities:
prob1 <- predict(logit1)

pred1 <- rep("0", nrow(train.set))
pred1[prob1 > 0.5] = "1"

(training.accuracy1 <- (1-mean(pred1 != train.set$Survived)))

# Achieved 0.7946128 on the training set

# let's examine the testing set:
test.submit.df <- data.comb[892:1309, ]
prob1.1 <- predict(logit1, newdata = test.submit.df)

pred1.1 <- rep("0", nrow(test.submit.df))
pred1.1[prob1.1 > 0.5] = "1"


submit.df <- data.frame(PassengerId = rep(892:1309) , Survived = pred1.1)
head(submit.df)

write.csv(submit.df, file = "submit.logit11.csv",row.names = F)

# Achieved 0.76555 on the testing set



# Second model:
train.set2 <- data.comb[1:891, c("Survived", "Pclass", "title", "Age")]
logit2  = glm(Survived ~ ., data = train.set2, family = "binomial")
summary(logit2)

# Predicted probabilities:
prob2 <- predict(logit2)

pred2 <- rep("0", nrow(train.set2))
pred2[prob2 > 0.5] = "1"

(training.accuracy2 <- (1-mean(pred2 != train.set2$Survived)))

# Achieved 0.8058361 on the training set

# let's examine the testing set:
test.submit.df <- data.comb[892:1309, ]
prob2.1 <- predict(logit2, newdata = test.submit.df)

pred2.1 <- rep("0", nrow(test.submit.df))
pred2.1[prob2.1 > 0.5] = "1"


submit.df <- data.frame(PassengerId = rep(892:1309) , Survived = pred2.1)
head(submit.df)

write.csv(submit.df, file = "submit.logit2.csv",row.names = F)

# Achieved 0.76076 accuracy on the testing set



# Third model:
train.set3 <- data.comb[1:891, c("Survived", "Pclass", "title", "Age", "SibSp")]
logit3  = glm(Survived ~ ., data = train.set3, family = "binomial")
summary(logit3)

# Predicted probabilities:
prob3 <- predict(logit3)

pred3 <- rep("0", nrow(train.set3))
pred3[prob3 > 0.5] = "1"

(training.accuracy3 <- (1-mean(pred3 != train.set3$Survived)))

# Achieved 0.8159371 on the training set



# Fourth Model
train.set4 <- data.comb[1:891, c("Survived", "Pclass", "title", "Age", "SibSp", "Fare")]
logit4 = glm(Survived ~ ., data = train.set4, family = "binomial")
summary(logit4)

# Predicted probabilities:
prob4 <- predict(logit4)

pred4 <- rep("0", nrow(train.set4))
pred4[prob4 > 0.5] = "1"

(training.accuracy4 <- (1-mean(pred4 != train.set4$Survived)))

# Achieved 0.8226712 on the training set



# Fifth Model
train.set5 <- data.comb[1:891, c("Survived", "Pclass", "title", "Age", "SibSp", "Fare", "Embarked")]

logit5 = glm(Survived ~ ., data = train.set5, family = "binomial")
summary(logit5)

# Predicted probabilities:
prob5 <- predict(logit5)

pred5 <- rep("0", nrow(train.set5))
pred5[prob5 > 0.5] = "1"

(training.accuracy5 <- (1-mean(pred5 != train.set5$Survived)))
# Achieved 0.8327722 accuracy on the training set! (improved)

# let's examine the testing set:
test.submit.df <- data.comb[892:1309, ]
prob2.5 <- predict(logit5, newdata = test.submit.df)

pred2.5 <- rep("0", nrow(test.submit.df))
pred2.5[prob2.5 > 0.5] = "1"


submit.df <- data.frame(PassengerId = rep(892:1309) , Survived = pred2.5)
head(submit.df)

write.csv(submit.df, file = "submit.logit5.csv",row.names = F)

# achieved 0.77990 accuracy on the testing set  

# Sixth model:

train.set6 <- data.comb[1:891, c("Survived", "Pclass", "title", "Age", "SibSp", "Fare", "Embarked", "Sex")]

logit6 = glm(Survived ~ ., data = train.set6, family = "binomial")
summary(logit6)

# Predicted probabilities:
prob6 <- predict(logit6)

pred6 <- rep("0", nrow(train.set6))
pred6[prob6 > 0.5] = "1"

(training.accuracy6 <- (1-mean(pred6 != train.set6$Survived)))
# Achieved 0.8372615 accuracy on the training set! (improved little from model the 5th model)

# let's examine the testing set:
test.submit.df <- data.comb[892:1309, ]
prob2.6 <- predict(logit6, newdata = test.submit.df)

pred2.6 <- rep("0", nrow(test.submit.df))
pred2.6[prob2.6 > 0.5] = "1"


submit.df <- data.frame(PassengerId = rep(892:1309) , Survived = pred2.6)
head(submit.df)

write.csv(submit.df, file = "submit.logit6.csv",row.names = F)

# achieved 0.77990 accuracy on the testing set 


# Seventh Model

train.set7 <- data.comb[1:891, c("Survived", "Pclass", "title", "Age", "SibSp", "Fare", "Embarked", "Sex", "cabin.firt.char")]

logit7 = glm(Survived ~ ., data = train.set7, family = "binomial")
summary(logit7)

# Predicted probabilities:
prob7 <- predict(logit7)

pred7 <- rep("0", nrow(train.set7))
pred7[prob7 > 0.5] = "1"

(training.accuracy7 <- (1-mean(pred7 != train.set7$Survived)))
# Achieved 0.8327722 accuracy on the training set! (The accuracy decreased from model 6th)


# let's examine the testing set:
test.submit.df <- data.comb[892:1309, ]
prob2.7 <- predict(logit7, newdata = test.submit.df)

pred2.7 <- rep("0", nrow(test.submit.df))
pred2.7[prob2.7 > 0.5] = "1"


submit.df <- data.frame(PassengerId = rep(892:1309) , Survived = pred2.7)
head(submit.df)

write.csv(submit.df, file = "submit.logit77.csv",row.names = F)

# achieved 0.77033 accuracy on the testing set (which decreased from model 6)



# Conclusion: Model 6th is the best in terms of the trainig and testing set
# Model 5 achieved exact similar accuracy on the testing set


# plot the accuracy for the 7th models:

accuracy <- rbind(training.accuracy1,training.accuracy2,training.accuracy3,
                  training.accuracy4,training.accuracy5,training.accuracy6,
                  training.accuracy7)

plot(accuracy, type = "b", 
     ylab = "Accuracy",
     xlab = "Model",
     main = "The Accuracy of Different Logistic
     Regression Models on Training Set")

####
#### Random Forest Model #########
###
library(randomForest)



# Model 1:

train.1 <-  data.comb[1:891,-1]
label.1 <- as.factor(train$Survived)
set.seed(1234)
rf.1 <- randomForest(x= train.1, y = label.1, importance = T, ntree = 1000)
rf.1

# Accuracy: 1- .1605 = 0.8395

# Imporatcnce of variables:
varImpPlot(rf.1)

# Accuray of the testing set
test1.submit.df <- data.comb[892:1309, -1]
rf.1.preds <- predict(rf.1, newdata = test1.submit.df)
submit.df <- data.frame(PassengerId = rep(892:1309) , Survived = rf.1.preds)
head(submit.df)
write.csv(submit.df, file = "submit1_rf.1.csv",row.names = F)

# 0.76555 (overfit)


# Model 2:
train.2 <-  data.comb[1:891,c("Pclass", "title", "Fare", "Sex")]
label.1 <- as.factor(train$Survived)
set.seed(1234)
rf.2 <- randomForest(x= train.2, y = label.1, importance = T, ntree = 1000)
rf.2

# Accuracy: 1- .1717 = 0.8283

# Imporatcnce of variables:
varImpPlot(rf.2)

# Accuray of the testing set
test2.submit.df <- data.comb[892:1309, c("Pclass", "title", "Fare", "Sex")]
rf.2.preds <- predict(rf.2, newdata = test2.submit.df)
submit.df <- data.frame(PassengerId = rep(892:1309) , Survived = rf.2.preds)
head(submit.df)
write.csv(submit.df, file = "submit1_rf.2.csv",row.names = F)

# 0.79904 (sharp improvement)


#############################
##### Cross Validation ######
#############################

# LDA:

# Vector to store predictions:
n = nrow(test.submit.df)

pred.holdout = rep(0, n)


for (i in 1:n)
{
  # Remove one observation at a time:
  test = test.submit.df[i,]
  # The rest are training set
  train = train.set5[-i,]
  ldamod5 = lda(Survived ~ ., data= train)
  pred.holdout[i] = predict(ldamod5, test)$posterior[2] > 0.5
}

submit.df <- data.frame(PassengerId = rep(892:1309) , Survived =pred.holdout )
head(submit.df)

write.csv(submit.df, file = "submitlda,cv.csv",row.names = F)

# 0.78468 accuracy achieved same as before 


## Logistic Regression:

# Will use model 6 from the logistic
test.submit.df.cv2 <- data.comb[ 892:1309,c("Survived", "Pclass", "title", "Age", "Fare", "Embarked", "Sex")]

n = nrow(test.submit.df)
pred.holdout2 = rep(0, n)

for (i in 1:n)
{
  # Remove one observation at a time:
  test = test.submit.df[i,]
  # The rest are training set
  train = train.set6[-i,]
  logit6 = glm(Survived ~ ., data = train, family = "binomial")
  pred.holdout2[i] = predict(logit6, test, type = "response") > 0.5
}

submit.df <- data.frame(PassengerId = rep(892:1309) , Survived = pred.holdout2)
head(submit.df)


write.csv(submit.df, file = "submit.logit6.cv.csv",row.names = F)

#  0.77511 accuracy





# Random Forest:
library(randomForest)



#install.packages("caret")
library(caret)
#install.packages("doSNOW")
library(doSNOW)

set.seed(2348)
cv.10.folds <- createMultiFolds(label.1, k=10, times = 10)


ctrl.1 <- trainControl(method = "repeatedcv", number = 10, repeats = 10,
                       index = cv.10.folds)

cl <- makeCluster(6, type = "SOCK")
registerDoSNOW(cl)

# Set seed for reproducibility and train (Takes a while)
set.seed(34324)
rf.2.cv.1 <- train(x = train.2, y = label.1, method = "rf", tuneLength = 3,
                   ntree = 1000, trControl = ctrl.1)

#Shutdown cluster
stopCluster(cl)

# Check out results
rf.2.cv.1

# Accuray of the testing set
test2.submit.df <- data.comb[892:1309, c("Pclass", "title", "Fare", "Sex")]
rf.2.cv.1.preds <- predict(rf.2.cv.1, newdata = test2.submit.df)
submit.df <- data.frame(PassengerId = rep(892:1309) , Survived = rf.2.cv.1.preds)
head(submit.df)
write.csv(submit.df, file = "submit1_rf.2.cv.csv",row.names = F)

# 0.77511 (not so good)


# with 3 folds
set.seed(23488)
cv.3.folds <- createMultiFolds(label.1, k=3, times = 3)


ctrl.2 <- trainControl(method = "repeatedcv", number = 3, repeats = 3,
                       index = cv.3.folds)

cl <- makeCluster(6, type = "SOCK")
registerDoSNOW(cl)

# Set seed for reproducibility and train (Takes a while)
set.seed(343244)
rf.2.cv.2 <- train(x = train.2, y = label.1, method = "rf", tuneLength = 2,
                   ntree = 1000, trControl = ctrl.2)

#Shutdown cluster
stopCluster(cl)

# Check out results
rf.2.cv.2

# Accuray of the testing set
test2.submit.df <- data.comb[892:1309, c("Pclass", "title", "Fare", "Sex")]
rf.2.cv.2.preds <- predict(rf.2.cv.2, newdata = test2.submit.df)
submit.df <- data.frame(PassengerId = rep(892:1309) , Survived = rf.2.cv.2.preds)
head(submit.df)
write.csv(submit.df, file = "submit1_rf.22.cv.csv",row.names = F)

# 0.79904 (improved from previous 10 folds)



