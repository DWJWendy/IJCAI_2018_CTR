#IJCAI 2018 阿里妈妈广告预测算法

### 背景

本项目是天池的一个比赛，由阿里妈妈和天池大数据众智平台举办广告预测算法大赛，本次参赛人数多达5200多，而我们只取得了731的成绩，最遗憾的是当我们写好CNN预测结果准备上传，发现队伍被强制解散，只因为的大神队友忘记实名注册，没心很无奈，又非常不甘心，但是这就是规则，我们只有遵循的权利，难过伤心之后还是需要把整个项目进行整理总结。

---

### 目标

本次比赛以阿里电商广告为研究对象，提供了淘宝平台的海量真实交易数据，参赛选手通过人工智能技术构建预测模型预估用户的购买意向，即给定广告点击相关的用户（user）、广告商品（ad）、检索词（query）、上下文内容（context）、商店（shop）等信息的条件下预测广告产生购买行为的概率（pCVR），形式化定义为：pCVR=P(conversion=1 | query, user, ad, context, shop)。

结合淘宝平台的业务场景和不同的流量特点，我们定义了以下两类挑战：
（1）日常的转化率预估
（2）特殊日期的转化率预估

评估指标

![评估指标](/images/blog/2018-05-02-1.png  "评估指标"){:height="50%" width="50%"}

通过logarithmic loss（记为logloss）评估模型效果（越小越好）， 公式如下：
其中N表示测试集样本数量，yi表示测试集中第i个样本的真实标签，pi表示第i个样本的预估转化率。

---

### 数据说明
本次比赛为参赛选手提供了5类数据（基础数据、广告商品信息、用户信息、上下文信息和店铺信息）。基础数据表提供了搜索广告最基本的信息，以及“是否交易”的标记。广告商品信息、用户信息、上下文信息和店铺信息等4类数据，提供了对转化率预估可能有帮助的辅助信息。

**1 . 基础数据**

| 字段 | 解释 | 
|: -------: | :------ : |
| instance_id | 样本编号，Long|
|is_trade|是否交易的标记位，Int类型；取值是0或者1，其中1 表示这条样本最终产生交易，0 表示没有交易|
|item_id|广告商品编号，Long类型|
|user_id|用户的编号，Long类型|
|context_id|上下文信息的编号，Long类型|
|shop_id|店铺的编号，Long类型|


**2. 广告商品信息**

| 字段 | 解释 | 
|: -------: | :------ : |
|item_id|广告商品编号，Long类型|
|item_category_list|广告商品的的类目列表，String类型；从根类目（最粗略的一级类目）向叶子类目（最精细的类目）依次排列，数据拼接格式为 "category_0;category_1;category_2"，其中 category_1 是 category_0 的子类目，category_2 是 category_1 的子类目|
|item_property_list|广告商品的属性列表，String类型；数据拼接格式为 "property_0;property_1;property_2"，各个属性没有从属关系|
|item_brand_id|广告商品的品牌编号，Long类型|
|item_city_id|广告商品的城市编号，Long类型|
|item_price_level|广告商品的价格等级，Int类型；取值从0开始，数值越大表示价格越高|
|item_sales_level|广告商品的销量等级，Int类型；取值从0开始，数值越大表示销量越大|
|item_collected_level|广告商品被收藏次数的等级，Int类型；取值从0开始，数值越大表示被收藏次数越大|
|item_pv_level|广告商品被展示次数的等级，Int类型；取值从0开始，数值越大表示被展示次数越大|

**3. 用户信息**

| 字段 | 解释 | 
|: -------: | :------ : |
|user_id|用户的编号，Long类型|
|user_gender_id|用户的预测性别编号，Int类型；0表示女性用户，1表示男性用户，2表示家庭用户|
|user_age_level|用户的预测年龄等级，Int类型；数值越大表示年龄越大|
|user_occupation_id|用户的预测职业编号，Int类型|
|user_star_level|用户的星级编号，Int类型；数值越大表示用户的星级越高|

**4. 上下文信息**

| 字段 | 解释 | 
|: -------: | :------ : |
|context_id|上下文信息的编号，Long类型|
|context_timestamp|广告商品的展示时间，Long类型；取值是以秒为单位的Unix时间戳，以1天为单位对时间戳进行了偏移|
|context_page_id|广告商品的展示页面编号，Int类型；取值从1开始，依次增加；在一次搜索的展示结果中第一屏的编号为1，第二屏的编号为2|
|predict_category_property|根据查询词预测的类目属性列表，String类型；数据拼接格式为 “category_A:property_A_1,property_A_2,property_A_3;category_B:-1;category_C:property_C_1,property_C_2” ，其中 category_A、category_B、category_C 是预测的三个类目；property_B 取值为-1，表示预测的第二个类目 category_B 没有对应的预测属性|

**5. 店铺信息**

| 字段 | 解释 | 
|: -------: | :------ : |
|shop_id|店铺的编号，Long类型|
|shop_review_num_level|店铺的评价数量等级，Int类型；取值从0开始，数值越大表示评价数量越多|
|shop_review_positive_rate|店铺的好评率，Double类型；取值在0到1之间，数值越大表示好评率越高|
|shop_star_level|店铺的星级编号，Int类型；取值从0开始，数值越大表示店铺的星级越高|
|shop_score_service|店铺的服务态度评分，Double类型；取值在0到1之间，数值越大表示评分越高|
|shop_score_delivery|店铺的物流服务评分，Double类型；取值在0到1之间，数值越大表示评分越高|
|shop_score_description|店铺的描述相符评分，Double类型；取值在0到1之间，数值越大表示评分越高|

---
	
### 思路
我们的实验思路如下：

**统计分析 -> 数据预处理 -> 特征抽取 -> 特征表示 -> 模型拟合和预测 -> 模型选择**

其实从实验思路我们可以明显看出特征工程在这次比赛尤为重要，只有刻画好特征，才能利用模型得到好的预测结果，接下来我将按照实验思路进行总结。

---


### 实验
#### 1. 统计分析

**目的： 看清数据分布，了解广告、商品、店铺、用户与购买概率的关系**

基础数据的统计分析（饼图、柱状图和折线图结合) ，将数据按照is_trade属性分为两张子表，分别进行对比统计分析
    
-   购买的用户分析：
    单变量：性别分布、年龄分布、职业分布、星级分布
    交叉变量：（重点）性别-年龄、性别-星级、年龄-星级、职业-星级,（参考）年龄-职业，性别-职业

-   购买的商品分布对比
   （重点）标签分布、属性分布、品牌分布、价格分布、销量分布，展示次数（后四项需考虑粒度的粗细）
   （参考）城市分布、收藏次数分布
   
-   上下文信息对比
   （重点）时间戳分布
   （参考）页面分布（看能否精确到类别）、预测类目的准确度（？）

-   购买的店铺分布对比
   （重点）评论数分布，好评率分布、星级分布
   （参考）服务评分、物流评分、描述评分

**实施：利用R对数据分布进行了统计，代码在analysis目录下，图片在pic和pic2中**

**结果如下**

以在不同属性上is_trade=0/1为例, 简要分析

---

- 1.转化分布
![转化分布](/images/blog/1.is_trade.png){:height="50%" width="50%"}
从上图可以明显看出在给定情景下转化率很低，也就是说，**我们的训练数据存在了极度平衡的现象，甚至是可以把购买理解成异常值，我们的算法要能够极好的检测出异常实例。**

---
- 2.年龄 | 性别 | 星级 分布
![年龄分布](/images/blog/2.user_age_level.png){:height="50%" width="50%"}![用户星级](/images/blog/5.user_star_level.png){:height="50%" width="50%"}
从图1可以明显看出，**年龄越大，转化率先增加后减少**（-1表示未知年龄），这个结果与我们常识一致，中间年龄段更具有消费能力， 性别转化分布没有贴出来，结果跟我们常识也是一致的，**女性转化率高于男性**。从图2中可以看出，星级越高购买率相对要更高一些，但是差距不太明显。

---

- 3.价格 | 收藏 |展示 分布

![价格分布](/images/blog/7.item_sales_level.png){:height="50%" width="50%"}![收藏分布](/images/blog/8.item_collected_level.png){:height="50%" width="50%"}![展示分布](/images/blog/9.item_pv-level.png){:height="50%" width=50%"}

图1，可以看出**价格越高转化率先增加后降低，这与我们对电商平台的认知有关**，价格太低必然会让人觉得物品质量不佳，但是随着价格增加，购买会带来更高的风险，转化率自然会降低。图2收藏次数越高，购买的可能性越大，**收藏在电商市场的本质，就是商品入选了用户的购买集**，对相关商品综合排序后，收藏的商品更有可能转化。图3，总体趋势是**展示次数（广告效应）越多，购买率越高**。

---

- 4.商店星级 | 评论数量  分布

![商店星级](/images/blog/12.shop_star_level.png){:height="50%" width=50%"}![评论数量](/images/blog/11.shop_review_num_level.png){:height="50%" width="50%"}

图1.商店星级差异不明显。图2.**评论数量居中的购买率更高**

- 5 城市 | 商品标签 分布
![城市](/images/blog/6.city_d.png){:height="50%" width=50%"}![商品](/images/blog/10.brand_d.png){:height="50%" width=50%"}

这两幅图是仅仅选择了高频的城市和商标分布，**可以看出城市和商品图，都有集中表现类，而商品更为明显**。

**总结： 数据统计分析的目的是分析变量之间的关系，观察具体特征对转化率的影响，从而用于模型中初始化权重**

---


#### 2. 数据预处理 


1. 处理缺失值
主要处理缺失值，以及属性值为-1的值，因为后期特征表示时，我们调用的sklearn借口进行one-hot表征，而借口要求输入数据不包括负数

2. 特征映射
由于城市和商品的值字段太长，在表征时会出现错误，因此将他们分别映射，并更新原始数据，代码如下
```
def map_field(train_data, test_data, path):
    train_data = set(train_data.unique())
    test_data = set(test_data.unique())
    all_property = list(train_data.union(test_data))
    print(len(all_property))
    map_data = {}
    for i in range(len(all_property)):
        map_data[str(all_property[i])] = i
    with open(path, "w", encoding="utf-8") as dump:
        json.dump(map_data, dump)
    return map_data


def update_data(raw_data, field_1, field_2, path_1, path_2):
    with open(path_1, "r", encoding="utf-8") as dump:
        update_field_1 = json.load(dump)
    with open(path_2, "r", encoding="utf-8") as dump:
        update_field_2 = json.load(dump)
    raw_data[field_1] = raw_data[field_1].apply(lambda x: update_field_1[str(x)])
    raw_data[field_2] = raw_data[field_2].apply(lambda x: update_field_2[str(x)])
    return raw_data
```

#### 3. 特征抽取

本小节主要涉及对特征的转化和抽取，比如在上下文中的时间轴数据，**考虑到节假日流量问题（比赛提出的挑战解决方法）**，我节假日和周末前后时间戳进行映射， 代码见下：

```
        def convert_interval_time(hour, size=3):
        """
        方法：把一天24小时按照力粒度为size大小进行分割
        :param hour:
        :param size:
        :return:
        """
        interval_time = [list(range(i, i + size)) for i in range(0, 24, size)]
        interval_time_factor = {}
        for i in range(len(interval_time)):
            interval_time_factor[i] = interval_time[i]
        for factor, h in interval_time_factor.items():
            if hour in h:
                return factor
            else:
                pass

    def time_to_factor(self):
        """
        方法：将时间戳转为区间因子
        tips: add two features, is_weekend and is_holiday
        :param data: 输入带时间戳的数据,
        :param size:将原始时间戳转化为粒度为size个小时一个因子,默认为size=3
        :return: 对于时间戳转为化为区间因子
        """
        data_time = self.raw_data[self.modify_features["time"]]
        data_time["format_time"] = data_time["context_timestamp"].apply(lambda x: time.localtime(x))
        convert_data_time = pd.DataFrame(columns=["interval_time", "is_weekends", "is_holidays"])
        convert_data_time["is_weekends"] = \
            data_time["format_time"].apply(lambda x: 1 if x[6] in self.weekends else 0)
        convert_data_time["is_holidays"] = \
            data_time["format_time"].apply(lambda x: 1 if str(x[1]) + "." + str(x[2]) in self.holidays else 0)
        convert_data_time["interval_time"] = \
            data_time["format_time"].apply(lambda x: self.convert_interval_time(hour=x[3]))
        return convert_data_time	
```
 属性值除了离散的，还包括的连续属性值，比如店铺的好评率、服务态度评分等等，连续属性离散化的代码如下：
```
     def continuous_var_to_factor(self, size=0.01):
        """
        方法：将连续数据按照粒度为size进行转化
        :param data_continuous: 输入带有连续变量的数据
        :param properity: 指定属性
        :param size： 粒度默认为0.01
        :return:
        """
        data_continuous = self.raw_data[self.modify_features['continuous']]
        data_continuous = (data_continuous / size).round()
        data_continuous = data_continuous.astype('int64')
        return data_continuous
```

#### 4. 特征表示
在这一部分，我们主要是是通过One-hot对所有数据特征进行表征，然后用One-hot的最大问题，尤其是在电商环境下特征表征，我们可以想象，这个数据维度非常巨大，所以，在处理这个高维数据时，我们先将其分为五个大的领域进行表征，再对其使用SVD进行降维处理。
One-hot表征代码如下：
```
    def one_hot_represent(self, new_data, data_type='value'):
        """
        :param new_data: pd.DataFrame. The data used to transform to the one-hot form.
        :param data_type: str, default 'value', alternative 'lst'. Aimed at different forms, we can use this parameter to
        specify the method of handling.
        :return:
        """
        if data_type == 'value':
            df_one_hot = pd.DataFrame()
            for col in new_data.columns:
                # print(col)
                one_hot_matrix = self.one_hot_model.fit_transform(new_data[[col]])
                feature_names = [col + str(attr) for attr in self.one_hot_model.active_features_]
                one_hot_matrix = pd.DataFrame(one_hot_matrix.toarray(), columns=feature_names)
                df_one_hot = pd.concat([df_one_hot, one_hot_matrix], axis=1)
            return df_one_hot
            # return pd.DataFrame(df_one_hot.toarray(), columns=feature_names)
        elif data_type == 'lst':
            cols = new_data.columns
            df_one_hot = pd.DataFrame()
            for col in cols:
                # print(col)
                data = new_data[col]
                all_values = list(reduce(lambda x, y: x | y, data, set()))
                one_hot_dim = len(all_values)
                one_hot_matrix = []
                for line in data:
                    one_hot_vec = np.zeros(one_hot_dim)
                    for value in line:
                        one_hot_vec[all_values.index(value)] = 1
                    one_hot_matrix.append(one_hot_vec)
                one_hot_matrix = pd.DataFrame(one_hot_matrix, columns=all_values)
                df_one_hot = pd.concat([df_one_hot, one_hot_matrix], axis=1)
            return df_one_hot
        else:
            raise ValueError('Can\'t recognize the type, please enter \'value\' or \'lst\'')
```
SVD降维代码如下：

```
class SVDReduce(object):
    """
    The class is used to reduce the dimension of the data outputed from the class DataPreprocess with SVD method.
    """

    def __init__(self, data, dimension=500):
        """
        Initialize the class with the parameters.
        :param data: pd.DataFrame, the output data from the class DataPreprocess.
        :param dimension: int, default 500. To specify the output dimension.
        """
        self.data = data
        self.target_dim = dimension
        self.format_data_path = '../../data/format_2/'
        self.field = ['user', 'product', 'context', 'shop']
        # self.field = ['product']

    def judge(self, data):
        """
        Abandon
        方法：判读大领域的维度
        标准维度,判断：不足补零,大于转为svd()
        :return:
        """
        logger.info("judge the dimension...")
        field_matrix_shape = data.shape
        dimension = field_matrix_shape[1]
        if dimension > self.target_dim:
            return True
        else:
            return False

    def svd(self, field_matrix):
        """
        方法：对大的领域数据进行降维
        :param field_matrix: list(2d) or np.array, 每一行(list)表示一条record
        :return: 返回领域的降维矩阵
        """
        logger.info("use svd to reduce the dimension")
        indices = field_matrix.index
        fm = field_matrix
        field_matrix = np.array(field_matrix)
        field_matrix_dim = field_matrix.shape
        print(field_matrix_dim)

        # 对维度进行判断是否需要降维
        if field_matrix_dim[1] <= self.target_dim:
            logger.info('Filed_matrix_dim if smaller than the target, no need to perform reduction, thus we'
                        'only add extra zero element to make up the dimension.')
            dim_make_up = self.target_dim - field_matrix_dim[1]
            matrix_make_up = np.zeros([field_matrix_dim[0], dim_make_up])
            matrix_make_up = pd.DataFrame(matrix_make_up, index=indices)
            return pd.concat([fm, matrix_make_up], axis=1)
        else:
            svd = TruncatedSVD(n_components=self.target_dim)
            return pd.DataFrame(svd.fit_transform(field_matrix), index=indices)

    def run(self):
        """
        1. Extract the one-hot-form data from the self.new_data_one_hot according to the field-instruction.
        2. Based on the given self.target_dimension, judge the field matrix whether satisfy the dimension requirement.
        3. If so, do the svd method, else add extra zero element to achieve the self.target_dimension.
        """
        output_matrix = []
        for i, field_data in enumerate(self.data):
            # field_data = self.split_field(field=item)
            svd_matrix = self.svd(field_matrix=field_data)
            svd_matrix.to_csv(self.format_data_path + 'svd_' + self.field[i] + '.csv')
            output_matrix.append(svd_matrix)
        return output_matrix

```
除了通过分领域和SVD降维以外，商品的属性上高达10多万类，所以我们还对**商品属性计算了信息增益**从而筛选了部分重要的商品属性。

```
    def feature_selection_with_info_gain(self, data, num_feature=500, feature='item_property_list'):
        print(os.path.exists('../../data/format_2/infomation_gain.txt'))
        if os.path.exists('../../data/format_2/infomation_gain.txt'):
            with open('../../data/format_2/infomation_gain.txt', 'r', encoding='utf-8') as r:
                selected_feature = []
                for i in range(num_feature):
                    line = r.readline().replace('\ufeff', '').strip().split(',')
                    selected_feature.append(line[0])
        else:
            fea_s = list(data.columns)
            fea_s.remove(feature)

            property = []
            for lst in data[feature]:
                for pro in lst:
                    if pro not in property:
                        property.append(pro)

            info_gain = pd.Series()
            for pro in property:
                series = pd.Series([1 if pro in lst else 0 for lst in data[feature]], index=data.index, name=pro)
                concat_data = pd.concat([series, self.raw_data['is_trade']], axis=1)
                info_gain[pro] = self.cal_info_gain(data=concat_data, independent_variable=pro,
                                                    dependent_variable='is_trade')

            info_gain = info_gain.sort_values(ascending=False)
            info_gain.to_csv('../../data/format_2/infomation_gain.txt', encoding='utf-8')
            selected_feature = list(info_gain.index[: num_feature])

        new_feature = []
        for lst in data[feature]:
            new_fea = []
            for pro in lst:
                if pro in selected_feature:
                    new_fea.append(pro)
            new_feature.append(set(new_fea))
        data[feature] = new_feature
        return data
```
---
**总结：这几小节主要是对特征的映射、筛选、表征，遇到的最大困难就是数据维度太高以致于服务器多次出现memory error，所以我们对原始表征数据时按照它给定的基础数据、广告商品信息、用户信息、上下文信息和店铺信息5大块分别onehot，并通过信息增益和SVD进行降维处理，所有代码均在data_helper.py中**

---

#### 5.模型拟合和预测

在模型过程中，我们考虑了很多个模型，首先是在广告预测领域用得最多且效果还可以的逻辑回归、Field-aware Factorization Machines、卷积神经网络，以及非常常用的分类方法：随机森林、提升数、简单的感知器等等，最终表现效果最好的是卷积神经网络。

**实验框架如下：**

![实验框架](/images/blog/2018-05-02-3.jpeg){:height="60%" width="60%"}
一定是我写累了，因为喜欢花花绿绿，图上颜色就觉得特别开心幸福，O(∩_∩)O哈哈哈~

**实验过程：**

我们首先将原始训练数据和测试集，将比赛提供的loss评估指标作为我们的损失函数，通过卷积神经网络进行训练，事实上因为CNN就自带降维效果，所以，输入CNN的数据是没有用SVD进行降维的。


NN方法代码如下：

```
class CNN1(object):

    def __init__(self, n_input, n_output, x_shape, batch_size, load=0):
        if load == 1:
            saver = tf.train.import_meta_graph("../../data/model/cnn_model.ckpt.meta")
            self.sess = tf.Session()
            saver.restore(self.sess, "../../data/model/cnn_model.ckpt")
        else:
            logger.info('building the graph...')

            self.kb = 0.8
            self.batch_size = batch_size

            self.x = tf.placeholder(tf.float32, [None, n_input], name='input')
            self.y = tf.placeholder(tf.float32, [None, n_output], name='true_label')
            self.x_ = tf.reshape(self.x, shape=x_shape)

            # define the first convolution layer
            self.W_conv1 = self.weight_variable([2, 2, 1, 16])
            self.b_conv1 = self.bias_variable([16])
            self.h_conv1 = tf.nn.relu(self.conv2d(self.x_, self.W_conv1) + self.b_conv1)
            self.h_pool1 = self.max_pool_2x2(self.h_conv1)

            # define the second convolution layer
            self.W_conv2 = self.weight_variable([2, 2, 16, 32])
            self.b_conv2 = self.bias_variable([32])
            self.h_conv2 = tf.nn.relu(self.conv2d(self.h_pool1, self.W_conv2) + self.b_conv2)
            self.h_pool2 = self.max_pool_2x2(self.h_conv2)

            # transform the result of h_pool2 into 1D-form
            self.h_pool2_ = tf.reshape(self.h_pool2, [-1, (x_shape[1] // 4) * (x_shape[2] // 4) *32])
            h_pool2_shape = self.h_pool2_.get_shape()
            self.W_fc1 = self.weight_variable([h_pool2_shape[1].value, 500])
            self.b_fc1 = self.bias_variable([500])
            self.h_fc1 = tf.nn.relu(tf.matmul(self.h_pool2_, self.W_fc1) + self.b_fc1)

            # add a dropout layer
            self.keep_prob = tf.placeholder(tf.float32, name='keep')
            self.h_fc1_drop = tf.nn.dropout(self.h_fc1, self.keep_prob)

            # add a softmax layer, and get the final probability
            self.W_fc2 = self.weight_variable([500, n_output])
            self.b_fc2 = self.bias_variable([n_output])
            self.pred = tf.nn.softmax(tf.matmul(self.h_fc1_drop, self.W_fc2) + self.b_fc2, name='pred')

            # self.loss_func = tf.reduce_mean(- self.y * tf.log(self.pred), name='loss_func')
            self.loss_func = tf.reduce_mean(- self.y * tf.log(self.pred), name='loss_func') + \
                             0.001 * tf.nn.l2_loss(self.W_conv1) + \
                             0.001 * tf.nn.l2_loss(self.W_conv2) + \
                             0.001 * tf.nn.l2_loss(self.W_fc1) + \
                             0.001 * tf.nn.l2_loss(self.W_fc2)
            self.optm = tf.train.AdadeltaOptimizer(0.005).minimize(self.loss_func)
            self.init_op = tf.global_variables_initializer()

            self.sess = tf.Session()
            self.sess.run(self.init_op)

    @staticmethod
    def weight_variable(shape):
        """
        the method used to define the weight variables of the convolution layers
        :param shape:tuple or list, 该权重的形状
        :return:
        """
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    @staticmethod
    def bias_variable(shape):
        """
        the method used to define the weight variables of the bias of each convolution layer
        :param shape:
        :return:
        """
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    @staticmethod
    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    @staticmethod
    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    @staticmethod
    def get_batch_data(batch_size, train_data, label_data):
        total = len(train_data)
        if batch_size > 1:
            chose_samples = random.sample(range(total), batch_size)
        else:
            chose_samples = random.randint(0, total)
        return train_data[chose_samples], label_data[chose_samples]

    def train(self, train_data, label_data, epoches):
        train_data = np.array(train_data)
        label_data = np.array(label_data)
        # for i in range(epoches):
        #     logger.info('running the {}-th round of training process...'.format(i))
        #     attr_data, labe_data = self.get_batch_data(self.batch_size, train_data, label_data)
        #     _, loss = self.sess.run([self.optm, self.loss_func],
        #                             feed_dict={self.x: attr_data, self.y:labe_data, self.keep_prob: self.kb})
        #     if i + 1 == epoches:
        #         logger.info('finish training process and the loss is {}'.format(loss))
        #     elif (i + 10) % 10 == 0:
        #         logger.info('running the {}-th epoch and the loss is {}.'.format(i, loss))
        with tf.device("/gpu:0"):
            for i in range(epoches):
                logger.info('running the {}-th round of training process...'.format(i))
                attr_data, labe_data = self.get_batch_data(self.batch_size, train_data, label_data)
                _, loss = self.sess.run([self.optm, self.loss_func],
                                        feed_dict={self.x: attr_data, self.y:labe_data, self.keep_prob: self.kb})
                if i + 1 == epoches:
                    logger.info('finish training process and the loss is '.format(loss))
                elif (i + 100) % 100 == 0:
                    logger.info('running the {}-th epoch and the loss is {}.'.format(i, loss))

    def predict(self, test_data, test_label, mode='test'):
        logger.info('predicting the result...')
        if mode == 'test':
            pred, loss = self.sess.run([self.pred, self.loss_func], feed_dict={self.x: test_data, self.y: test_label, self.keep_prob: self.kb})
            return pred, loss
        elif mode == 'predict':
            result = self.sess.run(self.pred, feed_dict={self.x: test_data, self.keep_prob: self.kb})
            return result

    def load_model_predict(self, test_data, test_label, mode='test'):
        if mode == 'test':
            result = self.sess.run(['pred: 0', 'loss_func: 0'],  feed_dict={'input: 0': test_data, 'true_label: 0': test_label, 'keep: 0': 0.8})
            return  result
        elif mode == 'predict':
            result = self.sess.run('pred: 0', feed_dict={'input: 0': test_data, 'keep: 0': 0.8})
            return result

    def save_cnn_model(self, path):
        saver = tf.train.Saver()
        saver.save(self.sess, path)


def data_matrix(all_field_path, user_file, product_file, context_file, shop_file, mode=0):
    user_field = list(pd.read_csv(user_file, sep=',', nrows=3).columns)
    product_field = list(pd.read_csv(product_file, sep=',', nrows=3).columns)
    context_field = list(pd.read_csv(context_file, sep=',', nrows=3).columns)
    shop_field = list(pd.read_csv(shop_file, sep=',', nrows=3).columns)
    all_field_data = pd.read_csv(all_field_path, sep=',')

    all_field_attrs = list(all_field_data.columns)
    # exclude_attrs = ['user_id', 'item_id', 'context_id', 'shop_id']
    attrs = [user_field, product_field, context_field, shop_field]
    for field in attrs:
        for attr in field:
            if attr not in all_field_attrs:
                field.remove(attr)

    max_length = max([len(attr) for attr in attrs]) + 1
    label = 0
    for field in attrs:
        diff = max_length - len(field)
        if diff > 0:
            for i in range(label, label + diff):
                field.append('x' + str(i))
                all_field_data['x' + str(i)] = 0
            label += diff
        else:
            pass

    attrs_orders = reduce(lambda x, y: x + y, attrs, [])

    if mode == 0:
        return all_field_data[attrs_orders], max_length
    elif mode == 1:
        return all_field_data[attrs_orders], all_field_data.index


def split_train_test(data, label_data, ratio):
    data, label_data = np.array(data), np.array(label_data)
    total = len(data)
    train_chosen_samples = random.sample(range(total), int(ratio * total))
    test_chosen_samples = []
    for ind in range(total):
        if ind not in train_chosen_samples:
            test_chosen_samples.append(ind)
    train_set_attrs, train_set_target = data[train_chosen_samples], label_data[train_chosen_samples]
    test_set_attrs, test_set_target = data[test_chosen_samples], label_data[test_chosen_samples]
    return train_set_attrs, train_set_target, test_set_attrs, test_set_target  
```

除了上述的神经网络的方法，我也通过Sklean调取相关API计算了LR、贝叶斯分类器、随机森林、提升树、感知器与上诉方法进行对比

```

class Data_Preprocess(object):
    def __init__(self, train_path, test_path, raw_train_path, raw_test_path):
        """
        Read the data including the train_data/test_data of one hot, raw_train_data/test_data, and the label of
        raw_train_data.
        :param train_path:
        :param test_path:
        :param raw_train_path:
        :param raw_test_path:
        """
        self.raw_train_data = self.read_data(raw_train_path, data_type="raw")  # 获取is_trade
        # 需要把她它分为测试集和训练集
        self.X_data = self.read_data(train_path, data_type="one-hot").drop("instance_id", axis=1)
        self.Y_label = self.raw_train_data["is_trade"]
        self.predict_data = self.read_data(test_path, data_type="one-hot")
        self.predict_x = self.alignment_data()
        self.predict_index = self.read_data(raw_test_path, data_type="raw")["instance_id"]

        # 交叉验证数据集
        self.X_train, self.X_test, self.Y_train, self.Y_test = self.cross_data()

    @staticmethod
    def read_data(path, data_type):
        """
        Read data according to the path of data
        :param data_type:
        :param path:
        :return:
        """
        if data_type == "raw":
            return pd.read_csv(path, sep=" ")
        elif data_type == "one-hot":
            return pd.read_csv(path, sep=",")

    def alignment_data(self):
        logger.info("数据对齐...")
        return self.predict_data.reindex(columns=self.X_data.columns, fill_value=0)

    @staticmethod
    def save_model(obj, path):
        pickle.dump(obj, open(path, "wb"))
        logger.info('The model has been saved to ' + path + '...')

    def cross_data(self):
        X_train, X_test, Y_train, Y_test = train_test_split(self.X_data, self.Y_label, test_size=0.1, random_state=0)
        return X_train, X_test, Y_train, Y_test


class LR_Model(object):

    def __init__(self):
        self.train_x, self.train_y, self.test_x, self.test_y = X_train, Y_train, X_test, Y_test
        self.predict_x = predict_x
        self.predict_index = predict_index

    def lr_model(self):
        """
        Method: logisticRegression
        :return: return the probability of test data with list format
        """
        logger.info('LR_model beginning ...')
        classifier = LogisticRegression(solver="sag", class_weight="balanced")
        classifier.fit(self.train_x, self.train_y)
        index = list(classifier.classes_).index(1)
        test_y_predict = pd.DataFrame(classifier.predict_proba(self.test_x), columns=list(classifier.classes_))
        test_y_predict[index] = test_y_predict[index].apply(lambda x: 0 if x <= 0.01 else x)
        predict_y = list(map(lambda x: x[index], classifier.predict_proba(self.predict_x)))
        data_results.save_model(obj=classifier, path="../../data/results_2/lr_model.pk")
        return test_y_predict, predict_y

    @staticmethod
    def evaluate(y_true, y_pred):
        logger.info("LR_model evaluating...")
        logloss = log_loss(y_true, np.array(y_pred))
        logger.info("The value of logloss:" + str(logloss))
        return logloss

    def write_result(self, predict_pro, path="../../data/results_2/lr_results.txt"):
        logger.info('Write_result finishing ...')
        with open(path, "w", encoding="utf-8") as f:
            f.write("instance_id" + " " + "predicted_score" + "\r")
            for i in range(len(predict_pro)):
                if predict_pro[i] > 0.01:
                    f.write(str(self.predict_index[i]) + " " + str(predict_pro[i]) + "\r")
                else:
                    f.write(str(self.predict_index[i]) + " " + str(0.0) + "\r")
        logger.info('Write_result finished ...')

    def run(self):
        test_y_predict, predict_y = self.lr_model()
        self.evaluate(self.test_y, test_y_predict)
        self.write_result(predict_pro=predict_y)
        logger.info('lr_model finished ...')


class Bayes_Model(object):

    def __init__(self):
        self.train_x, self.train_y, self.test_x, self.test_y = X_train, Y_train, X_test, Y_test
        self.predict_x = predict_x
        self.predict_index = predict_index

    def bayes_model(self):
        logger.info('Bayes_model beginning ...')
        classifier = BernoulliNB()
        classifier.fit(self.train_x, self.train_y)
        index = list(classifier.classes_).index(1)
        test_y_predict = pd.DataFrame(classifier.predict_proba(self.test_x), columns=list(classifier.classes_))
        test_y_predict[index] = test_y_predict[index].apply(lambda x: 0 if x <= 0.01 else x)
        predict_y = list(map(lambda x: x[index], classifier.predict_proba(self.predict_x)))
        data_results.save_model(obj=classifier, path="../../data/results_2/bayes_model.pk")
        return test_y_predict, predict_y

    @staticmethod
    def evaluate(y_true, y_pred):
        logger.info("Bayes_model evaluating...")
        logloss = log_loss(y_true, np.array(y_pred))
        logger.info("The value of logloss:" + str(logloss))
        return logloss

    def write_result(self, predict_pro, path="../../data/results_2/bayes_results.txt"):
        logger.info('Write_result finishing ...')
        with open(path, "w", encoding="utf-8") as f:
            f.write("instance_id" + " " + "predicted_score" + "\r")
            for i in range(len(predict_pro)):
                if predict_pro[i] > 0.01:
                    f.write(str(self.predict_index[i]) + " " + str(predict_pro[i]) + "\r")
                else:
                    f.write(str(self.predict_index[i]) + " " + str(0.0) + "\r")
        logger.info('Write_result finished ...')

    def run(self):
        test_y_predict, predict_y = self.bayes_model()
        self.evaluate(self.test_y, test_y_predict)
        self.write_result(predict_pro=predict_y)
        logger.info('bayes_model finished ...')


class RandomTree(object):

    def __init__(self):
        self.train_x, self.train_y, self.test_x, self.test_y = X_train, Y_train, X_test, Y_test
        self.predict_x = predict_x
        self.predict_index = predict_index

    def randomtree_model(self):
        logger.info('RandomTree_model beginning ...')
        classifier = RandomForestClassifier(class_weight="balanced")
        classifier.fit(self.train_x, self.train_y)
        index = list(classifier.classes_).index(1)
        test_y_predict = pd.DataFrame(classifier.predict_proba(self.test_x), columns=list(classifier.classes_))
        test_y_predict[index] = test_y_predict[index].apply(lambda x: 0 if x <= 0.01 else x)
        predict_y = list(map(lambda x: x[index], classifier.predict_proba(self.predict_x)))
        data_results.save_model(obj=classifier, path="../../data/results_2/random_tree_model.pk")
        return test_y_predict, predict_y

    @staticmethod
    def evaluate(y_true, y_pred):
        logger.info("Random_tree_model evaluating...")
        logloss = log_loss(y_true,np.array(y_pred))
        logger.info("The value of logloss:" + str(logloss))
        return logloss

    def write_result(self, predict_pro, path="../../data/results_2/random_tree_results.txt"):
        logger.info('Write_result finishing ...')
        with open(path, "w", encoding="utf-8") as f:
            f.write("instance_id" + " " + "predicted_score" + "\r")
            for i in range(len(predict_pro)):
                if predict_pro[i] > 0.01:
                    f.write(str(self.predict_index[i]) + " " + str(predict_pro[i]) + "\r")
                else:
                    f.write(str(self.predict_index[i]) + " " + str(0.0) + "\r")
        logger.info('Write_result finished ...')

    def run(self):
        test_y_predict, predict_y = self.randomtree_model()
        self.evaluate(self.test_y, test_y_predict)
        self.write_result(predict_pro=predict_y)
        logger.info('random_tree_model finished ...')


class GTB(object):

    def __init__(self):
        self.train_x, self.train_y, self.test_x, self.test_y = X_train, Y_train, X_test, Y_test
        self.predict_x = predict_x
        self.predict_index = predict_index

    def gtb_model(self):
        logger.info('GTB_model beginning ...')
        classifier = GradientBoostingClassifier()
        classifier.fit(self.train_x, self.train_y)
        index = list(classifier.classes_).index(1)
        test_y_predict = pd.DataFrame(classifier.predict_proba(self.test_x), columns=list(classifier.classes_))
        test_y_predict[index] = test_y_predict[index].apply(lambda x: 0 if x <= 0.01 else x)
        predict_y = list(map(lambda x: x[index], classifier.predict_proba(self.predict_x)))
        data_results.save_model(obj=classifier, path="../../data/results_2/gtb_model.pk")
        return test_y_predict, predict_y

    @staticmethod
    def evaluate(y_true, y_pred):
        logger.info("GTB_model evaluating...")
        logloss = log_loss(y_true, np.array(y_pred))
        logger.info("The value of logloss:" + str(logloss))
        return logloss

    def write_result(self, predict_pro, path="../../data/results_2/gtb_results.txt"):
        logger.info('Write_result finishing ...')
        with open(path, "w", encoding="utf-8") as f:
            f.write("instance_id" + " " + "predicted_score" + "\r")
            for i in range(len(predict_pro)):
                if predict_pro[i] > 0.01:
                    f.write(str(self.predict_index[i]) + " " + str(predict_pro[i]) + "\r")
                else:
                    f.write(str(self.predict_index[i]) + " " + str(0.0) + "\r")
        logger.info('Write_result finished ...')

    def run(self):
        test_y_predict, predict_y = self.gtb_model()
        self.evaluate(self.test_y, test_y_predict)
        self.write_result(predict_pro=predict_y)
        logger.info('GTB_model finished ...')


class NeuralNetwork(object):

    def __init__(self):
        self.train_x, self.train_y, self.test_x, self.test_y = X_train, Y_train, X_test, Y_test
        self.predict_x = predict_x
        self.predict_index = predict_index

    def nn_model(self):
        logger.info('NN_model beginning ...')
        classifier = MLPClassifier(solver="sgd", hidden_layer_sizes=(500, 3))
        classifier.fit(self.train_x, self.train_y)
        index = list(classifier.classes_).index(1)
        test_y_predict = pd.DataFrame(classifier.predict_proba(self.test_x), columns=list(classifier.classes_))
        test_y_predict[index] = test_y_predict[index].apply(lambda x: 0 if x <= 0.01 else x)
        predict_y = list(map(lambda x: x[index], classifier.predict_proba(self.predict_x)))
        data_results.save_model(obj=classifier, path="../../data/results_2/nn_model.pk")
        return test_y_predict, predict_y

    @staticmethod
    def evaluate(y_true, y_pred):
        logger.info("NN_model evaluating...")
        logloss = log_loss(y_true, np.array(y_pred))
        logger.info("The value of logloss:" + str(logloss))
        return logloss

    def write_result(self, predict_pro, path="../../data/results_2/nn_results.txt"):
        logger.info('Write_result beginning ...')
        with open(path, "w", encoding="utf-8") as f:
            f.write("instance_id" + " " + "predicted_score" + "\r")
            for i in range(len(predict_pro)):
                if predict_pro[i] > 0.01:
                    f.write(str(self.predict_index[i]) + " " + str(predict_pro[i]) + "\r")
                else:
                    f.write(str(self.predict_index[i]) + " " + str(0.0) + "\r")
        logger.info('Write_result finished ...')

    def run(self):
        test_y_predict, predict_y = self.nn_model()
        self.evaluate(self.test_y, test_y_predict)
        self.write_result(predict_pro=predict_y)
        logger.info('NN_model finished ...')
```
对于不同的模型，我们使用自己分的测试集对其进行预测，并计算相关的损失函数，损失函数的结果如下：

**【注意：结果值越小越好】**

|Mehod|LR|Bayes|Random_T|GTB|NN|CNN|
|: -------: | :------ : |: -------: | :------ : |: -------: | :------ : |:------ : |
|Loss|4.10158|1.015423|0.539459|0.09011|0.089561|0.046641|

![](/images/blog/2018-05-02-2.png)

**从上表可以看出逻辑回归|贝叶斯分类器|随机森林|简单感知器|CNN的结果，而我们提出的CNN算法在测试集上的效果为0.046641明显优于其他方法，然后我们却没能够得到最终的验证，哭死哭死。**

---

### 总结
虽然这次比赛机器遗憾和难过，但是不得不说，真的学习了很多很多，不论是在对数据处理上、方法上还是码代码上，虽然遇到很多问题但是都通过自己和小伙伴的努力解决了，还是非常感谢小伙伴。

其实，对于很多事，除了【努力】、【机遇】、【幸运】，还是要注意细节，一直都喜欢对自己说，把每件小事做好了，那么结果一定不会太差，那么，未来，请继续努力吧！ 

毕竟海贼王的女人是不会认输的，O(∩_∩)O哈哈哈~，下面奉上今日新作《海贼王 路飞》

![路飞](/images/blog/2015-05-02-4.jpeg){:height="30%" width="30%"}

---

**【注意更多更完整的代码详见github】**

	用户名：DWJWendy
	链接：https://github.com/DWJWendy/IJCAI_2018_CTR.git




