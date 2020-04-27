######################################################
##PROJECT: Predicting Infection of Organization Endpoints by Cybersecurity Threats 
#using Ensemble Machine Learning Techniques

##AUTHOR: VISHAKHA BHATTACHARJEE
##Version 1.1
##Date: 30/03/2020
######################################################

#############INSTALLING PACKAGES
install.packages('mice')
install.packages('caret')

#############LOADING PACKAGES
library(mice)
library(caret)

############READING DATA
df = read.csv('Train.csv')
View(df)
sapply(df, function(x) sum(is.na(x)))
#Firewall - Factor - logreg
#IsProtected - Factor - logreg
#SystemVolumeTotalCapacity - numeric - pmm
#PrimaryDiskTotalCapacity - numeric - pmm
#IsAlwaysOnAlwaysConnectedCapable - Factor - logreg
#AdminApprovalMode - Factor - logreg
#TotalPhysicalRAM - Numeric - pmm
#IsSecureBootEnabled - Factor - logreg
#IsGamer - Factor - logreg


##############converting into factors(categorical variables)
df$HasTpm = as.factor(df$HasTpm)
df$IsProtected = as.factor(df$IsProtected)
df$Firewall = as.factor(df$Firewall)
df$AdminApprovalMode = as.factor(df$AdminApprovalMode)
df$HasOpticalDiskDrive = as.factor(df$HasOpticalDiskDrive)
df$IsSecureBootEnabled = as.factor(df$IsSecureBootEnabled)
df$IsPenCapable = as.factor(df$IsPenCapable)
df$IsAlwaysOnAlwaysConnectedCapable = as.factor(df$IsAlwaysOnAlwaysConnectedCapable)
df$IsGamer = as.factor(df$IsGamer)
df$IsInfected = as.factor(df$IsInfected)

str(df)
ncol(df)

###############REMOVING MachineId FROM DATA FRAME
df = df[,-c(1)]


##############IMPUTATION OF MISSING DATA USING MICE
init = mice(df, maxit=0) 
meth = init$method
predM = init$predictorMatrix

#Excluding the output column IsInfected as a predictor for Imputation
predM[, c("IsInfected")]=0

#Excluding these variables from imputation
meth[c("ProductName","HasTpm","Platform","Processor","SkuEdition","DeviceType","HasOpticalDiskDrive","IsPenCapable","IsInfected")]=""


#Specifying the imputation methods for the varaibles with missing data                                                                                                                                                              
meth[c("SystemVolumeTotalCapacity","PrimaryDiskTotalCapacity","TotalPhysicalRAM")]="cart" 
meth[c("Firewall","IsProtected","IsAlwaysOnAlwaysConnectedCapable","AdminApprovalMode","IsSecureBootEnabled","IsGamer")]="logreg" 
meth[c("PrimaryDiskTypeName","AutoUpdate","GenuineStateOS")]="polyreg"

#Setting Seed for reproducibility 
set.seed(103)

#Imputing the data
imputed = mice(imputed, method=meth, predictorMatrix=predM, m=5)
imputed <- complete(imputed)

sapply(imputed, function(x) sum(is.na(x)))

sum(is.na(imputed))

#####IMPUTING MANUALLY A FEW BLANK ROWS OF DATA
imputed$PrimaryDiskTypeName[imputed$PrimaryDiskTypeName == ""] = "UNKNOWN"

imputed$AutoUpdate[imputed$AutoUpdate == ""] = "UNKNOWN"

imputed$GenuineStateOS[imputed$GenuineStateOS == ""] = "UNKNOWN"

#writing imputed data into a new file for future use and backup
write.csv(imputed,'MICE_Imputed_Data.csv')


#########################MODEL CREATION
imputed = read.csv('MICE_Imputed_Data.csv')
str(imputed)

##############converting into factors(categorical variables)
imputed$HasTpm = as.factor(imputed$HasTpm)
imputed$IsProtected = as.factor(imputed$IsProtected)
imputed$Firewall = as.factor(imputed$Firewall)
imputedAdminApprovalMode = as.factor(imputed$AdminApprovalMode)
imputed$HasOpticalDiskDrive = as.factor(imputed$HasOpticalDiskDrive)
imputed$IsSecureBootEnabled = as.factor(imputed$IsSecureBootEnabled)
imputed$IsPenCapable = as.factor(imputed$IsPenCapable)
imputed$IsAlwaysOnAlwaysConnectedCapable = as.factor(imputed$IsAlwaysOnAlwaysConnectedCapable)
imputed$IsGamer = as.factor(imputed$IsGamer)

#Converting the output variables classes into single letters F and T from 0 and 1 respectively
#This is required by the CARET package for ROC metric functionality
imputed$IsInfected[imputed$IsInfected == 0] = "F"
imputed$IsInfected[imputed$IsInfected == 1] = "T"

#Converting the output variable to a factor variable
imputed$IsInfected = as.factor(imputed$IsInfected)

##Omitting two columns in the begining of the data that do not contribute
imputed = imputed[,-c(1,2)]
nrow(imputed)
sum(is.na(imputed))
sapply(imputed, function(x) sum(is.na(x)))
View(imputed)


##Function twoClassSummary is used to call upon metrics like ROC in CARET package function train()
head(twoClassSummary)

####################BOOSTING ALGORTIHMS

control <- trainControl(method="repeatedcv", number=10, repeats=3, classProbs = TRUE, summaryFunction = twoClassSummary)
seed <- 7

##Only one metric can be set at a time. Other metric like Accuracy can also be checked
metric <- "ROC"

################### C5.0
set.seed(seed)
fit.c50 <- train(IsInfected~., data=imputed, method="C5.0", metric=metric, trControl=control)
print(fit.c50)
confusionMatrix(fit.c50)


################### Stochastic Gradient Boosting
set.seed(seed)
fit.gbm <- train(IsInfected~., data=imputed, method="gbm", metric=metric, trControl=control, verbose=FALSE)
print(fit.gbm)
confusionMatrix(fit.gbm)
############ summarize results
boosting_results <- resamples(list(c5.0=fit.c50, gbm=fit.gbm))
summary(boosting_results)
dotplot(boosting_results)


#Plotting the Boosting Algorithms
trellis.par.set(caretTheme())
plot(fit.c50)
plot(fit.gbm)





##########################BAGGING ALGORTIHMS
control <- trainControl(method="repeatedcv", number=10, repeats=3, classProbs = TRUE, summaryFunction = twoClassSummary)
seed <- 7
metric <- "ROC"

################## Bagged CART
set.seed(seed)
fit.treebag <- train(IsInfected~., data=imputed, method="treebag", metric=metric, trControl=control)
print(fit.treebag)
confusionMatrix(fit.treebag)


################## Random Forest
set.seed(seed)
fit.rf <- train(IsInfected~., data=imputed, method="rf", metric=metric, trControl=control)
print(fit.rf)
confusionMatrix(fit.rf)


##################### summarize results
bagging_results <- resamples(list(treebag=fit.treebag, rf=fit.rf))
summary(bagging_results)
dotplot(bagging_results)



###Plotting Bagging Algorithms
trellis.par.set(caretTheme())
plot(fit.treebag)
plot(fit.rf)



########Generating aggregated statistics and plots of all models
model_list = list(C5.0 = fit.c50,
                  rf = fit.rf,
                  Bagged_CART = fit.treebag,
                  Stochastic_GB = fit.gbm)

resamp <- resamples(model_list)
resamp

summary(resamp)
lattice::bwplot(resamp, metric = "ROC")
