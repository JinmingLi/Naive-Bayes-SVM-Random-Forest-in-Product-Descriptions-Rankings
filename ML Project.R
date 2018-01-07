library(readxl)
library(writexl)
library(jiebaR)
library(jiebaRD)
library(plyr)
library(stringr)
library(NLP)
library(tm)
library(pROC)
library(ggplot2)
library(klaR)
library(e1071)
library(randomForest)


##### Text Data Preprocessing #####

##### Read Text #####
MyText <- read_excel("C:/Users/Kimmy/Desktop/textinformation.xlsx", col_names = TRUE)

##### Cut Words (Self-assigned Phrases and Stop Words) #####
engine <- worker(user = 'C:/Users/Kimmy/Desktop/alltextwords.txt', stop_word = 'C:/Users/Kimmy/Desktop/allstopwords.xlsx')
cutwords <- llply(MyText$Text, segment, engine)

##### Eliminate Numbers and Alphabets #####
Text <- lapply(cutwords, str_replace_all, '[0-9a-zA-Z]', '')

##### Eliminate NULL Characters #####
nullcharacter <- which(Text == '')
Text2 <- Text[-nullcharacter]

##### Eliminate NULL Characters' Results #####
Text3 <- llply(Text2, function(x) x[!x == ''])

##### Convert Cutwords into Corpus #####
text_corpus <- Corpus(VectorSource(Text3))

##### Establish Document-Term Matrix using TF-IDF Method ######
DTMatrix <- DocumentTermMatrix(x = text_corpus, control = list(weighting = weightTfIdf, wordLengths = c(2, Inf)))
DTMatrix

##### Control Document-Term Matrix Sparsity #####
DTMatrixSparsity <- removeSparseTerms(x= DTMatrix, sparse = 0.95)
DTMatrixSparsity

##### Convert Document-Term Matrix into Dataframe #####
Mydata <- as.data.frame(as.matrix(DTMatrixSparsity))

##### Output Dataset #####
write_xlsx(Mydata,path ="dataset.xlsx")




##### Classify text data with three classifiers: Naive Bayes, SVM and Random Forest #####

##### Input dataset #####
Mydata<-read_excel("C:/Users/Kimmy/Desktop/dataset.xlsx", col_names = TRUE)

##### Change data type #####
Mydata$Class <- factor(Mydata$Class)

##### Divide dataset: 75% Trainset and 25% in Testset #####
set.seed(1)
index <- sample(1:nrow(Mydata), size = 0.75*nrow(Mydata))
Trainset <- Mydata[index,]
Testset <- Mydata[-index,]




##### Naive Bayes #####
bayes <- NaiveBayes(Class~., Trainset, fL = 1)  
#bayes <- NaiveBayes(x = Trainset, grouping = Trainset$Class, fL = 1)

##### Predict Testset with Naive Bayes #####
predict_bayes <- predict(bayes, newdata = Testset)
frequency_bayes <- table(predict_bayes$class, Testset$Class)

##### Confusion Matrix #####
frequency_bayes

##### Compute Accuracy #####
sum(diag(frequency_bayes))/sum(frequency_bayes)

### Mapping ROC Curve and AUC ###
roc_bayes <- roc(Testset$Class, factor(predict_bayes$class, ordered = T))
Specificity <- roc_bayes$specificities
Sensitivity <- roc_bayes$sensitivities

p <- ggplot(data = NULL, mapping = aes(x = 1-Specificity, y = Sensitivity))
p + geom_line(colour = 'red', size = 1) + coord_cartesian(xlim = c(0,1), ylim = c(0,1)) + geom_abline(intercept = 0, slope = 1) + annotate('text', x = 0.5, y = 0.25, label = paste('AUC=', round(roc_bayes$auc,2))) + labs(x = '1-Specificity', y = 'Sensitivity', title = 'Naive Bayes ROC Curve') + theme(plot.title = element_text(hjust = 0.5, face = 'bold', colour = 'brown'))




##### Random Forest #####
rf <- randomForest(Class~., data = Trainset)

##### Predict Testset with Random Forest #####
predict_rf <- predict(rf, newdata = Testset)
frequency_rf <- table(predict_rf, Testset$Class)

##### ConfusionMmatrix #####
frequency_rf

##### Compute Accuracy #####
sum(diag(frequency_rf))/sum(frequency_rf)

##### Mapping ROC Curve and AUC #####
roc_rf <- roc(Testset$Class, factor(predict_rf, ordered = T))
Specificity <- roc_rf$specificities
Sensitivity <- roc_rf$sensitivities

p <- ggplot(data = NULL, mapping = aes(x = 1-Specificity, y = Sensitivity))
p + geom_line(colour = 'red', size = 1) + coord_cartesian(xlim = c(0,1), ylim = c(0,1)) + geom_abline(intercept = 0, slope = 1) + annotate('text', x = 0.5, y = 0.25, label = paste('AUC=', round(roc_rf$auc,2))) + labs(x = '1-Specificity', y = 'Sensitivity', title = 'Random Forest ROC Curve') + theme(plot.title = element_text(hjust = 0.5, face = 'bold', colour = 'brown'))




##### Linear SVM #####
svmlinear <- svm(Class~., data = Trainset, kernel = "linear", scale = FALSE) # linear svm, scaling turned off

##### Predict Testset with Linear SVM #####
predict_svmlinear <- predict(svmlinear, newdata = Testset)
frequency_svmlinear <- table(predict_svmlinear, Testset$Class)

##### Confusion Matrix #####
frequency_svmlinear

##### Compute Accuracy #####
sum(diag(frequency_svmlinear))/sum(frequency_svmlinear)

##### Mapping ROC Curve and AUC ##### 
roc_svmlinear <- roc(Testset$Class, factor(predict_svmlinear, ordered = T))
Specificity <- roc_svmlinear$specificities
Sensitivity <- roc_svmlinear$sensitivities

p <- ggplot(data = NULL, mapping = aes(x = 1-Specificity, y = Sensitivity))
p + geom_line(colour = 'red', size = 1) + coord_cartesian(xlim = c(0,1), ylim = c(0,1)) + geom_abline(intercept = 0, slope = 1) + annotate('text', x = 0.5, y = 0.25, label = paste('AUC=', round(roc_svmlinear$auc,2))) + labs(x = '1-Specificity', y = 'Sensitivity', title = 'Linear SVM ROC Curve') + theme(plot.title = element_text(hjust = 0.5, face = 'bold', colour = 'brown'))




##### Tune SVM to find the optimum parameters #####
Tune <- tune.svm(Class~., data = Trainset, gamma = 10^(-6:-1), cost = 10^(1:2)) 
summary (Tune) # The best parameter: gama: 0.001, cost = 100 

##### SVM with Kernels ######
svmradial <- svm(Class~., data = Trainset, kernel = "radial", cost = 100, gamma = 0.001, scale = FALSE) # radial svm, scaling turned OFF
#svm <- svm(Class~., data = Trainset, kernel = "polynomial", cost = 10, gamma = 0.000001, scale = FALSE) # polynomial svm, scaling turned OFF
#svm <- svm(Class~., data = Trainset, kernel = "sigmoid", cost = 10, gamma = 0.000001, scale = FALSE) # sigmoid svm, scaling turned OFF
print(svm)

##### Predict Testset with SVM Radial Basis Function kernel #####
predict_svmradial <- predict(svmradial, newdata = Testset)
frequency_svmradial <- table(predict_svmradial, Testset$Class)

##### Confusion Matrix #####
frequency_svmradial
##### Accuracy #####
sum(diag(frequency_svmradial))/sum(frequency_svmradial)

##### Mapping ROC Curve and AUC #####  
roc_svmradial <- roc(Testset$Class, factor(predict_svmradial, ordered = T))
Specificity <- roc_svmradial$specificities
Sensitivity <- roc_svmradial$sensitivities

p <- ggplot(data = NULL, mapping = aes(x = 1-Specificity, y = Sensitivity))
p + geom_line(colour = 'red', size = 1) + coord_cartesian(xlim = c(0,1), ylim = c(0,1)) + geom_abline(intercept = 0, slope = 1) + annotate('text', x = 0.5, y = 0.25, label = paste('AUC=', round(roc_svmradial$auc,2))) + labs(x = '1-Specificity', y = 'Sensitivity', title = 'SVM with Radial Basis Function Kernel ROC Curve') + theme(plot.title = element_text(hjust = 0.5, face = 'bold', colour = 'brown'))



##### End ##### 

