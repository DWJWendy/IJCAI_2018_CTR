# -*-encoding:utf-8 -*- IJCAI 数据分析
library(ggplot2)
library(GGally)
library(gridExtra)

setwd("/home/dwj/Documents/实验室工作/IJCAI 阿里广告预测/data/")
getwd()
raw_data <- read.csv("round1_ijcai_18_train_20180301.csv",sep = " ")
str(raw_data)
summary(raw_data)


## 第一 单变量分析

## 1.购买率条形图
raw_data$is_trade <- factor(raw_data$is_trade)
ggplot(data = raw_data,aes(x=is_trade,fill=is_trade))+geom_bar(width=0.3)

## 用户维度
## 1.年龄分布
raw_data$user_age_level <- as.factor(raw_data$user_age_level)
ggplot(data = raw_data) + geom_bar(mapping = aes(x=user_age_level,fill= is_trade), position = "fill",width = 0.6)

## 2.性别分布
raw_data$user_gender_id<- as.factor(raw_data$user_gender_id)
ggplot(data = raw_data, aes(x=user_gender_id,fill= is_trade))+geom_bar(position = "fill",width = 0.3)


## 3.职业分布
raw_data$user_occupation_id<- as.factor(raw_data$user_occupation_id)
ggplot(data = raw_data, aes(x=user_occupation_id,fill=is_trade))+geom_bar(position = "fill", width = 0.4)

## 4.星级
raw_data$user_star_level<- as.factor(raw_data$user_star_level)
ggplot(data = raw_data, aes(x=user_star_level,fill=is_trade))+geom_bar(position = "fill", width = 0.6)


## 广告商品维度
## 1.城市分布 （选择城市高20个）
raw_data$item_city_id <- as.factor(raw_data$item_city_id)
city_data <- data.frame(city= c(levels(raw_data$item_city_id)),count = c(integer(length =length(levels(raw_data$item_city_id)))))
for (i in 1:length(raw_data$item_city_id)){
  city_data$count[raw_data$item_city_id[i]]= city_data$count[raw_data$item_city_id[i]] +1
}

city_data$count <- as.factor(city_data$count,ordered = TRUE)
city_data_subset = city_data[(city_data$count)>2000,]


ggplot(data = city_data_subset, aes(x =  city, y = count,fill=city)) +
  theme(axis.text.x = element_blank())+
  stat_summary(fun.y = "mean",
               geom = "bar",
               color = "red")+
  geom_text(aes(label=count))


## 2.价格等级
raw_data$item_price_level<- as.factor(raw_data$item_price_level)
ggplot(data = raw_data, aes(x=item_price_level,fill=is_trade))+geom_bar(alpha=0.7,width=0.4,col="black",position = "fill")

## 3.销量等级
raw_data$item_sales_level<- as.factor(raw_data$item_sales_level)
ggplot(data = raw_data, aes(x=item_sales_level,fill=is_trade))+geom_bar(col="black",position = "fill",width = 0.4)

## 4.收藏等级
raw_data$item_collected_level<- as.factor(raw_data$item_collected_level)
ggplot(data = raw_data, aes(x=item_collected_level,fill=is_trade))+geom_bar(col="gray",position = "fill",width = 0.4)

## 5.展示次数
raw_data$item_pv_level<- as.factor(raw_data$item_pv_level)
ggplot(data = raw_data, aes(x=item_pv_level,fill=is_trade))+geom_bar(col="gray",position = "fill",width = 0.4)

## 6.品牌分布
raw_data$item_brand_id<- as.factor(raw_data$item_brand_id)
ggplot(data = raw_data, aes(x=item_brand_id))+geom_bar(alpha=1,width=0.4,fill="cyan",col="black")

raw_data$item_brand_id <- as.factor(raw_data$item_brand_id)

brand_data <- data.frame(brand= c(levels(raw_data$item_brand_id)),count = c(integer(length =length(levels(raw_data$item_brand_id)))))
brand_data
for (i in 1:length(raw_data$item_brand_id)){
  brand_data$count[raw_data$item_brand_id[i]]=brand_data$count[raw_data$item_brand_id[i]] +1
}


brand_data_subset = brand_data[(brand_data$count)>3000,]

brand_data_subset
ggplot(data = brand_data_subset, aes(x =brand, y = count,fill=brand)) +
  theme(axis.text.x = element_blank())+
  stat_summary(fun.y = "mean",
               geom = "bar",
               color = "red")+
  geom_text(aes(label=count))

## 7.商品属性分布
raw_data$item_property_list<- as.factor(raw_data$item_property_list)
ggplot(data = raw_data, aes(x=item_property_list))+geom_bar(alpha=1,width=0.4,fill="cyan",col="black")

## 8.类目分布

## 店铺信息
## 1.店铺评价等级
raw_data$shop_review_num_level<- as.factor(raw_data$shop_review_num_level)
ggplot(data = raw_data, aes(x=shop_review_num_level,fill=is_trade))+geom_bar(col="gray",position = "fill",width = 0.4)

## 2.店铺星级
raw_data$shop_star_level<- as.factor(raw_data$shop_star_level)
ggplot(data = raw_data, aes(x=shop_star_level,fill=is_trade))+geom_bar(col="gray",position = "fill",width = 0.4)

## 3.好评率/服务态度/物流评分/描述评分

