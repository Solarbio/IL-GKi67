#https://zhuanlan.zhihu.com/p/113676828
#https://zhuanlan.zhihu.com/p/60427340
#https://blog.csdn.net/weixin_43569478/article/details/108079548
if(!requireNamespace("survival",quietly = T))
  install.packages("survival")
if(!requireNamespace("survminer",quietly = T))
  install.packages("survminer")
library(survival)
library(survminer)
#读入数据，将数据命名为lung
#time：以天为单位的生存时间
#status：删失状态1 = 删失，2 = 出现失效事件(即死亡事件)
#age：岁
#sex：性别，男= 1女= 2
lung <- read.csv("lung.csv",row.names = 1)
#status有两种类型，一种是用0和1表示，另一种是我们数据中的用1和2表示，无论是用那种表示，
#数值大的那一个的结局是死亡，也就是我们希望看到的结果，数值小的部分就是数据删失

#下面开始进行数据的检验
fit <- survfit(Surv(time, status) ~ sex, data = lung)
print(fit)


ggsurvplot(fit,
           data = lung,  # 指定变量数据来源
           #fun = "cumhaz",# "event"绘制累积事件(f(y)=1-y)，# "cumhaz"绘制累积危害函数(f(y)=-log(y));# "pct"绘制生存概率(百分比)。
           linetype = 1, # 根据分层更改线型c(0,1) or  c("solid", "dashed") or "strata"
           surv.median.line = "hv", # 同时显示垂直和水平参考线 即增加中位生存时间   可选 "none"、"hv"、"h"、"v"
           palette = "hue" ,#定义颜色 可选调色板有 "hue" "grey","npg","aaas","lancet","jco", "ucscgb","uchicago","simpsons"和"rickandmorty".c("red","blue")
           #add.all = TRUE, # 添加总患者生存曲线
           
           #图标题和坐标轴标签 图例标题和位置
           xlab = "Time", # 指定x轴标签
           ylab = "Survival probability", # 指定y轴标签
           title = "",# 指定title标签
           legend = c(0.8,0.75), # 指定图例位置 "top"(默认),"bottom","left","right","none"
           legend.title = "Gender", # 设置图例标题
           legend.labs = c("Male", "Female") ,# 指定图例分组标签
           
           #坐标轴范围、刻度间距
           break.x.by = 100,# 设置x轴刻度间距
           xlim = c(0, 1180),#设置x轴范围
           #ylim = c(0, 600),#设置x轴范围
           break.y.by = .25,# 设置y轴刻度间距
           axes.offset = F, # 逻辑词，默认为TRUE。为FALSE，则生存曲线图的坐标轴从原点开始。
           
           #置信区间
           conf.int = TRUE,#增加置信区间，但是这东西并没啥实质作用
           ##conf.int.fill  # 设置置信区间填充的颜色
           #conf.int.style # 设置置信区间的类型，有"ribbon"(默认),"step"两种。
           conf.int.alpha = .3,# 数值，指定置信区间填充颜色的透明度； # 数值在0-1之间，0为完全透明，1为不透明。
           
           #P值文本大小和位置
           pval = TRUE, #log 秩检验  看两个曲线之间有无显著区别
           pval.size = 3,# 指定p值文本大小的数字，默认为 5。
           pval.coord = c(100,.3),# 长度为2的数字向量，指定p值位置x、y，如pval.coord=c(x,y)。
           #pval.method = T,#展示p统计的检验方法
           #pval.method.size = 3,# 指定检验方法 log.rank 文本的大小
           #pval.method.coord =  c(10,.3),# 指定检验方法 log.rank 文本的坐标
           
           #删失点
           censor = T, # 逻辑词，默认为TRUE，在图上绘制删失点。
           censor.shape = 3, # 数值或字符，用于指定删失点的形状；默认为"+"(3), 可选"|"(124)。
           censor.size = 4.5,# 指定删失点形状的大小，默认为4.5。
           
           #生存表
           risk.table = F, # 添加风险表""absolute" 显示处于风险中的绝对数量；# "percentage" 显示处于风险中的百分比数量# "abs_pct" 显示处于风险中的绝对数量和百分比
           risk.table.col = c(1),#"strata", # 根据分层更改风险表颜色 c(1, 2)  or c("solid", "dashed").
           fontsize = 4,# 指定风险表和累积事件表的字体大小。
           tables.y.text = T,# 逻辑词，默认显示生存表的y轴刻度标签；为FALSE则刻度标签被隐藏
           tables.y.text.col = F, # 逻辑词，默认FALSE；为TRUE，则表的y刻度标签将按strata着色。
           tables.height = .3,# 指定所有生存表的高度，数值在0-1之间，默认为0.25.
           
           #累积事件表
           #cumevents = T,#累计死亡人数
           #cumevents.height = .3,
           
           #累积删失表
           #cumcensor =T,#累计删失人数
           #cumcensor.height = .3,
           
           ggtheme = theme_survminer(), #图的主题
           tables.theme = theme_bw()#下面图的主题
)

res.cox <- coxph(Surv(time, status) ~ sex, data = lung)
summary(res.cox)

#首先查看likehood ration test , wald test, logrank test三种检验方法的p值，p值小于0.05, 
#这个回归方程是统计学显著的。说明自变量对生存时间具有影响的因素。
#HR值为0.588<1,该数据集中1=male, 2= female, HR表示的是数值大的风险/数值小的风险，在这里就是female/ male
#说明female死亡的相对较低。HR的值约为0.58, 说明female的死亡风险只占了male的58%， 相比male, female的死亡风险降低了42%。


res.cox <- coxph(Surv(time, status) ~ age, data = lung)
summary(res.cox)
#age的HR值大于1， 说明随着ph.ecog数值的增加，死亡风险会增加。

