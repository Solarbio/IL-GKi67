library(survival)
library(survminer)

library("readxl")
data <- read_excel("C:\\Users\\GongHaiyu\\OneDrive\\病理\\基线+mh.xlsx")
print(data)

res.cut <- surv_cutpoint(data, time = "OS_mouths", event = "OS_status",
                         variables = c("MH", "auto_ki67", "manual_ki67") 
)

summary(res.cut)
# 2. Plot cutpoint for DEPDC1
plot(res.cut, "MH", palette = "npg")
## $DEPDC1


library(survival)
library(survminer)

library("readxl")
data <- read_excel("C:\\Users\\GongHaiyu\\OneDrive\\病理\\基线+mh - 副本.xlsx")
print(data)

res.cut <- surv_cutpoint(data, time = "OS_status", event = "OS_mouths",
                         variables = c("MH") 
)

summary(res.cut)
# 2. Plot cutpoint for DEPDC1
plot(res.cut, "MH")
## $DEPDC1