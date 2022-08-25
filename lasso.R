
library(MXM)

df <- read.csv('C:/Users/USER/Desktop/file1.csv')
df <- df[,2:ncol(df)]

library(glmnet)

x=df[,1:2]
x=data.frame(x,df[,4:26])
y=df[,3]
x=as.matrix(x)
y=as.double(as.matrix(df$Attrition))
set.seed(100)
cv.lasso <- cv.glmnet(x, y, family='binomial', alpha=1, parallel=TRUE, standardize=TRUE, type.measure='auc')

cat('Min Lambda: ', cv.lasso$lambda.min, '\n 1Sd Lambda: ', cv.lasso$lambda.1se)
df_coef <- round(as.matrix(coef(cv.lasso, s=cv.lasso$lambda.min)), 2)

# See all contributing variables
df_coef[df_coef[, 1] != 0, ]