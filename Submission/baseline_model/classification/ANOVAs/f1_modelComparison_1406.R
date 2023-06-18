#
# Mixed ANOVA factorial: 1 (or more) between variable and 1 (or more) within variable 
#


# Clear variables
rm(list=ls())
#Get path
path<-getwd(); path
# Set path
setwd(path)

#Importing Data
#reading from SPSS
#install.packages("foreign")
library(foreign)
my.file <- "f1_SONAE.csv"
data1 <- read.csv(file=my.file)
mydata1 <- as.data.frame(data1)
mydata1 <- subset(mydata1, select = -c(GENDER, SEG_LIFESTYLE_CD))
str(mydata1)

#transformation from a wide format to a long format
#Changing the dependent variables names
names(mydata1)<-c("Customer", "f1_Baseline", "f1_RandomForest", "FamilyMembers", "Lifestage")
attach(mydata1)

# Cluster identification variable: subject and categorical variables must be classified as factor
is.factor(mydata1$Customer)
mydata1$Customer <- factor(mydata1$Customer);

is.factor(mydata1$FamilyMembers)
mydata1$FamilyMembers <- factor(mydata1$FamilyMembers);
levels(mydata1$FamilyMembers) <- c("(1, 2)","(3, 8)")

is.factor(mydata1$Lifestage)
mydata1$Lifestage <- factor(mydata1$Lifestage);
levels(mydata1$Lifestage) <- c("FamilyWithKids","ActiveAdults")
str(mydata1)

#vector with the dependent variables and repeated measures variables names: c('bdi', 'RM1_Name', 'RM2_Name',...)
v <- c('f1')
varying1 = sapply(v, grep, names(mydata1))
# convert the dataframe from a wide format to a long format
mydata1_long <- reshape(mydata1, idvar = "Customer", varying = t(varying1), v.names = v ,direction = "long", timevar = "Model")
str(mydata1_long)

#IMPORTANT: define within-subjects variable (Model) as a factor
is.factor(mydata1_long$Model)
mydata1_long$Model <- factor(mydata1_long$Model);
levels(mydata1_long$Model) <- c("Baseline","RandomForest")

is.factor(mydata1_long$FamilyMembers)
is.factor(mydata1_long$Lifestage)
str(mydata1_long)
attach(mydata1_long)

# using ezANOVA function: more complete function for a mixed ANOVA
library(ez)
model<-ezANOVA(data=mydata1_long, dv=.(f1), wid=.(Customer), within=.(Model), between=.(FamilyMembers, Lifestage), type=3, detailed=TRUE, return_aov=FALSE)
model


####### Simple effects for within-subjects #######
# Within-subjects effect at a fixed level of the between-subjects factor (model differences in one segment)
library(car)
# Get the wide format columns of the within-subjects variable (model)
Y<-cbind(f1_Baseline, f1_RandomForest)
# data.frame for the within variable: 2 levels
model_inRBp <- data.frame(TypeModel=gl(ncol(Y), 1))
# for Lifestage=="FamilyWithKids"
model1_1<- lm(Y ~ 1, data=mydata1, subset=(Lifestage=="FamilyWithKids"), contrasts = NULL)
model1_1_Anova_III <- Anova(model1_1, idata=model_inRBp, idesign=~TypeModel, type=3)
model1_1_Anova_III

# for Lifestage=="ActiveAdults"
model1_2<- lm(Y ~ 1, data=mydata1, subset=(Lifestage=="ActiveAdults"), contrasts = NULL)
model1_2_Anova_III <- Anova(model1_2, idata=model_inRBp, idesign=~TypeModel, type=3)
model1_2_Anova_III

# for FamilyMembers=="(1, 2)"
# data.frame for the within variable: 2 levels
model1_3<- lm(Y ~ 1, data=mydata1, subset=(FamilyMembers=="(1, 2)"), contrasts = NULL)
model1_3_Anova_III <- Anova(model1_3, idata=model_inRBp, idesign=~TypeModel, type=3)
model1_3_Anova_III

# for FamilyMembers=="(3, 8)"
model1_4<- lm(Y ~ 1, data=mydata1, subset=(FamilyMembers=="(3, 8)"), contrasts = NULL)
model1_4_Anova_III <- Anova(model1_4, idata=model_inRBp, idesign=~TypeModel, type=3, complete=TRUE)
model1_4_Anova_III

#simple effects for between-subjects
#Between-subjects effect at a fixed level of the within-subjects factor (segment differences for a givn model)

# for Model=Baseline
model2_1<- lm(f1_Baseline ~ FamilyMembers + Lifestage + FamilyMembers:Lifestage, data=mydata1, contrasts = list(FamilyMembers=contr.sum, Lifestage=contr.sum))
model2_1_Anova_III <- Anova(model2_1, type=3)
model2_1_Anova_III

# for Model = RandomForest
model2_2<- lm(f1_RandomForest ~ FamilyMembers + Lifestage + FamilyMembers:Lifestage, data=mydata1, contrasts = list(FamilyMembers=contr.sum, Lifestage=contr.sum))
model2_2_Anova_III <- Anova(model2_2, type=3)
model2_2_Anova_III

detach(mydata1)
detach(mydata1_long)

# Clear variables
rm(list=ls())
