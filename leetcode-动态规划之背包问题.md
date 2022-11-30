## 动态规划之背包问题

[TOC]

### 01背包问题

**问题描述：**有n 个物品，它们有各自的重量和价值，现有给定容量的背包，如何让背包里装入的物品具有最大的价值总和？

为什么叫做`0/1背包`呢？

因为每个物品`只有一个`，对于每个物品而言，只有两种选择`，选它或者不选，选它记为1，不选记为0。

**分析：**定义一个二维数组` dp `存储最大价值，其中` dp[i][j] `表示前` i `件物品体积不超过` j `的情况下能达到的最大价值。设第 i 件物品体积为 w，价值为 v，根据第 i 件物品是否添加到背包中，可以分两种情况讨论：

- 第 i 件物品没添加到背包，总体积不超过 j 的前 i 件物品的最大价值就是总体积不超过 j 的前 i-1 件物品的最大价值，`dp[i][j] = dp[i-1][j]`。
- 第 i 件物品添加到背包中，`dp[i][j] = dp[i-1][j-w] + v`。

那么0-1背包问题的状态转移方程：`dp[i][j] = max(dp[i-1][j], dp[i-1][j-w] + v)`

```python
// W 为背包总体积
// N 为物品数量
// weights 数组存储 N 个物品的重量
// values 数组存储 N 个物品的价值
def knapsack(W, N, weights, values) {
    dp = [[0] * (W + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        w = weights[i - 1]
    	v = values[i - 1]
        for j in range(1, W + 1):  
            if j >= w:
                dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - w] + v)
            else:
                dp[i][j] = dp[i - 1][j]            
    return dp[N][W]
```

**空间优化：**前 i 件物品的状态仅与前 i-1 件物品的状态有关，因此可以将 dp 定义为一维数组，其中 dp[j] 既可以表示 `dp[i-1][j] `也可以表示` dp[i][j]`。此时，`dp[j] = max(dp[j], dp[j-w] + v)`

因为` dp[j-w] `表示 `dp[i-1][j-w]`，计算` dp[j] `依赖 `dp[i-1][j-w]`，若先计算`dp[j-w]`，会把` dp[i][j-w]`覆盖掉。所以要**倒序**，先计算` dp[i][j] `再计算` dp[i][j-w]`。

如果一旦正序遍历了，那么物品0就会被重复加入多次！ 例如代码如下：

```
// 正序遍历
for (int j = weight[0]; j <= bagWeight; j++) {
    dp[0][j] = dp[0][j - weight[0]] + value[0];
}
```

例如`dp[0][1] `是15，到了`dp[0][2] = dp[0][2 - 1] + 15`; 也就是`dp[0][2] = 30 `了，那么就是物品0被重复放入了。所以一定要倒叙遍历，**保证物品0只被放入一次**！
https://leetcode-cn.com/problems/partition-equal-subset-sum/solution/bang-ni-ba-0-1bei-bao-xue-ge-tong-tou-by-px33/

https://www.shangmayuan.com/a/82408cfcb6994dd49bf2777e.html

```python
def knapsack(W, N, weights, values) {
    dp = [0] * (W + 1)
    for i in range(1, N + 1):
        w = weights[i - 1]
    	v = values[i - 1]
    	for j in range(W, 0, -1):  # for j in range(W, w-1, -1)
    		if j >= w:
    			dp[j] = max(dp[j], dp[j - w] + v)
    return dp[-1]
}
```

#### 类似题目

#### [474. 一和零](https://leetcode-cn.com/problems/ones-and-zeroes/)

题目：给你一个二进制字符串数组 strs 和两个整数 m 和 n 。

请你找出并返回 strs 的最大子集的大小，该子集中 最多 有 m 个 0 和 n 个 1 。

```
输入：strs = ["10", "0001", "111001", "1", "0"], m = 5, n = 3
输出：4
解释：最多有 5 个 0 和 3 个 1 的最大子集是 {"10","0001","1","0"} ，因此答案是 4 。
其他满足题意但较小的子集包括 {"0001","1"} 和 {"10","1","0"} 。{"111001"} 不满足题意，因为它含 4 个 1 ，大于 n 的值 3 。
```

解题：本题中strs 数组里的元素就是物品**，**每个物品都是一个！而m 和 n相当于是一个背包，两个维度的背包。

```python
class Solution:
    def findMaxForm(self, strs: List[str], m: int, n: int) -> int:
        if len(strs) == 0:
            return 0
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for s in strs:
            zeros = s.count("0")
            ones = s.count("1")
            for i in range(m, zeros - 1, -1):
                for j in range(n, ones - 1, -1):
                    dp[i][j] = max(dp[i][j], 1 + dp[i - zeros][j - ones])
        return dp[m][n] 
```



#### [416. 分割等和子集](https://leetcode-cn.com/problems/partition-equal-subset-sum/)

**题目：**给你一个 **只包含正整数** 的 **非空** 数组 `nums` 。请你判断是否可以将这个数组分割成两个子集，使得两个子集的元素和相等。

```
输入：nums = [1,5,11,5]
输出：true
解释：数组可以分割成 [1, 5, 5] 和 [11] 。
```

意思就是**在nums中找到`target=sum/2`的01背包问题**

##### 递归

对于每个元素，都有 选或不选它 去组成子序列。我们可以 DFS 回溯去穷举所有的情况。

每次考察一个元素，用索引`i`描述，还有一个状态：当前累加的`curSum`。

递归函数：基于已选的元素（和为curSum），从i开始继续选，能否选出和为sum/2的子集。

每次递归，都有两个选择：

+ 选nums[i]。基于选它，往下继续选（递归）：`dfs(curSum + nums[i], i + 1)`

+ 不选nums[i]。基于不选它，往下继续选（递归）：`dfs(curSum, i + 1)`

递归的终止条件有三种情况：

+ `curSum > target`，已经爆了，不用继续选数字了，终止递归，返回false。
+ `curSum == target`，满足条件，不用继续选数字了，终止递归，返回true。
+ 指针越界，考察完所有元素，能走到这步说明始终没有返回true，所以返回false。

```js
const canPartition = (nums) => {
    let sum = 0;
    for (const n of nums) { // 求数组和
        sum += n;
    }
    if (sum % 2 != 0) return false; // 如果 sum 为奇数，直接返回 false

    const target = sum / 2; // 目标和

    const dfs = (curSum, i) => {    // curSum是当前累加和，i是指针
        if (i == nums.length || curSum > target) { // 递归的出口
            return false;
        }
        if (curSum == target) {                    // 递归的出口
            return true;
        }
        // 选nums[i]，当前和变为curSum+nums[i]，考察的指针移动一位
        // 不选nums[i]，当前和还是curSum，考察的指针移动一位
        return dfs(curSum + nums[i], i + 1) || dfs(curSum, i + 1);
    };

    return dfs(0, 0); // 递归的入口，当前和为0，指针为0
};

```

##### 加入记忆化

描述一个子问题的两个变量是`curSum`和`i`，组成 key 字符串，存入 hashMap，值为对应的计算结果。

js版

```js
const canPartition = (nums) => {
    let sum = 0;
    for (const n of nums) { // 求数组和
        sum += n;
    }
    if (sum % 2 != 0) return false; // 如果 sum 为奇数，直接返回 false
    const memo = new Map();
    const target = sum / 2; // 目标和

    const dfs = (curSum, i) => {    // curSum是当前累加和，i是指针
        if (i == nums.length || curSum > target) { // 递归的出口
            return false;
        }
        if (curSum == target) {                    // 递归的出口
            return true;
        }
        const key = curSum + '&' + i;   // 描述一个问题的key
        if (memo.has(key)) {            // 如果memo中有对应的缓存值，直接使用
            return memo.get(key);
        }
        const res = dfs(curSum + nums[i], i + 1) || dfs(curSum, i + 1);
        memo.set(key, res);  // 计算的结果存入memo
        return res;
    };

    return dfs(0, 0); // 递归的入口，当前和为0，指针为0
};
```

python版

```python
class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        self.res = False
        total = sum(nums)
        if total & 1:
            return False
        memo = {}
        target = total // 2
        def dfs(i, cur):
            if (cur, i) in memo:
                return memo[(cur, i)]
            if cur == target:
                return True 
            if cur > target or i == len(nums):
                return False
            res = dfs(i + 1, cur + nums[i]) or dfs(i + 1, cur)
            memo[(cur, i)] = res
            return res 
        return dfs(0, 0)
```

##### 动态规划

1. 特例：如果sum为奇数，那一定找不到符合要求的子集，返回False。

2. dp[j]含义：有没有和为j的子集，有为True，没有为False。比如和为0一定可以取到（也就是子集为空），那么dp[0] = True。

3. 接下来开始遍历nums数组，对遍历到的数nums[i]有两种操作，一个是选择这个数，一个是不选择这个数。

   - 不选择这个数：dp不变

   - －选择这个数：dp中已为True的情况再加上nums[i]也为True。

     比如dp[0]已经为True，那么dp[0 + nums[i]]也是True

   状态转移方程：`dp[j] = dp[j] or dp[j - nums[i]]`

```python
class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        sumAll = sum(nums)
        if sumAll % 2:
            return False
        target = sumAll // 2

        dp = [False] * (target + 1)
        dp[0] = True

        for i in range(len(nums)):
            for j in range(target, nums[i] - 1, -1):
                dp[j] = dp[j] or dp[j - nums[i]]
        return dp[-1]
```



#### [494. 目标和](https://leetcode-cn.com/problems/target-sum/)

**题目：**给定一个非负整数数组，a1, a2, ..., an, 和一个目标数，S。现在你有两个符号 + 和 -。对于数组中的任意一个整数，你都可以从 + 或 -中选择一个符号添加在前面。返回可以使最终数组和为目标数 S 的所有添加符号的方法数。

```
输入：nums: [1, 1, 1, 1, 1], S: 3
输出：5
解释：
-1+1+1+1+1 = 3
+1-1+1+1+1 = 3
+1+1-1+1+1 = 3
+1+1+1-1+1 = 3
+1+1+1+1-1 = 3
一共有5种方法让最终目标和为3。
```

1. 01背包问题是选或者不选，但本题是必须选，是选+还是选-。先将本问题转换为01背包问题。
   假设所有符号为+的元素和为x，符号为-的元素和的绝对值是y。
   我们想要的 S = 正数和 - 负数和 = x - y
   而已知x与y的和是数组总和：x + y = sum
   可以求出` x = (S + sum) / 2 = target`
   也就是我们要从nums数组里选出几个数，令其和为target
   于是就转化成了**求容量为target的01背包问题** 
2. 特例判断
   如果S大于sum，不可能实现，返回0
   如果x不是整数，也就是S + sum不是偶数，不可能实现，返回0
3. `dp[j]`代表的意义：填满容量为j的背包，有dp[j]种方法。因为填满容量为0的背包有且只有一种方法，所以dp[0] = 1
4. 状态转移：`dp[j] = dp[j] + dp[j - num]`
   当前填满容量为j的包的方法数 = 之前填满容量为j的包的方法数 + 之前填满容量为`j - num`的包的方法数
   也就是当前数`num`的加入，可以把之前和为`j - num`的方法数加入进来。

```python
class Solution:
    def findTargetSumWays(self, nums: List[int], S: int) -> int:
        sumAll = sum(nums)
        if sumAll < S or (sumAll + S) % 2:
            return 0
        target = (sumAll + S) // 2
        dp = [0] * (target + 1)
        dp[0] = 1
        for num in nums:
            for j in range(target, num - 1, -1):
                dp[j] += dp[j-num]
        return dp[-1]
```

#### [1049. 最后一块石头的重量 II](https://leetcode-cn.com/problems/last-stone-weight-ii/)

有一堆石头，用整数数组 `stones` 表示。其中 `stones[i]` 表示第 `i` 块石头的重量。每一回合，从中选出**任意两块石头**，然后将它们一起粉碎。假设石头的重量分别为 `x` 和 `y`，且 `x <= y`。那么粉碎的可能结果如下：

- 如果 `x == y`，那么两块石头都会被完全粉碎；
- 如果 `x != y`，那么重量为 `x` 的石头将会完全粉碎，而重量为 `y` 的石头新重量为 `y-x`。

最后，**最多只会剩下一块** 石头。返回此石头 **最小的可能重量** 。如果没有石头剩下，就返回 `0`。

```
输入：stones = [2,7,4,1,8,1]
输出：1
解释：
组合 2 和 4，得到 2，所以数组转化为 [2,7,1,8,1]，
组合 7 和 8，得到 1，所以数组转化为 [2,1,1,1]，
组合 2 和 1，得到 1，所以数组转化为 [1,1,1]，
组合 1 和 1，得到 0，所以数组转化为 [1]，这就是最优值。
```

https://leetcode-cn.com/problems/last-stone-weight-ii/solution/gong-shui-san-xie-xiang-jie-wei-he-neng-jgxik/

将问题切换为 01 背包问题:**从 stones组中选择，凑成总和不超过 sum/2的最大价值。**

```python
class Solution:
    def lastStoneWeightII(self, stones: List[int]) -> int:
        sums = sum(stones)
        dp = [0] * (sums // 2 + 1)
        for s in stones:
            for i in range(sums // 2, s - 1, -1):
                dp[i] = max(dp[i], dp[i - s] + s)
        return sums - 2* dp[-1]
```



### 完全背包问题

每种物品都有无限件可用。

#### 类似题目

#### [322. 零钱兑换](https://leetcode-cn.com/problems/coin-change/description/)

题目：给一些面额的硬币，要求用这些硬币来组成给定面额的钱数，并且使得**硬币数量最少**。硬币可以重复使用。

- 物品：硬币
- 物品大小：面额
- 物品价值：数量

```
Example 1:
coins = [1, 2, 5], amount = 11
return 3 (11 = 5 + 5 + 1)

Example 2:
coins = [2], amount = 3
return -1.
```

解题：完全背包问题——填满容量为amount的背包最少需要多少硬币

1. dp[j]代表含义：填满容量为j的背包最少需要多少硬币
2. 初始化dp数组：因为硬币的数量一定不会超过amount，而amount <= 10^4 ，因此初始化数组值为10001；dp[0] = 0
3. 转移方程：`dp[j] = min(dp[j], dp[j - coin] + 1)`
4. 当前填满容量j最少需要的硬币 = min( 之前填满容量j最少需要的硬币, 填满容量 j - coin 需要的硬币 + 1个当前硬币）
   返回dp[amount]，如果dp[amount]的值为10001没有变过，说明找不到硬币组合，返回-1

```python
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        dp = [0] + [10001] * amount
        for coin in coins:
            for i in range(coin, amount+1):
                dp[i] = min(dp[i], dp[i - coin] + 1)
        return dp[-1] if dp[-1] != 10001 else -1
```



#### [518. 零钱兑换 II](https://leetcode-cn.com/problems/coin-change-2/)

题目：给定不同面额的硬币和一个总金额。写出函数来计算可以凑成总金额的**硬币组合数**。假设每一种面额的硬币有无限个。 

```
输入: amount = 5, coins = [1, 2, 5]
输出: 4
解释: 有四种方式可以凑成总金额:
5=5
5=2+2+1
5=2+1+1+1
5=1+1+1+1+1
```

解题：完全背包之组合问题——填满容量为amount的背包，有几种硬币组合

1. dp[j] 代表装满容量为j的背包有几种硬币组合
2. 转移方程：dp[j] = dp[j] + dp[j - coin]
   当前填满j容量的方法数 = 之前填满j容量的硬币组合数 + 填满j - coin容量的硬币组合数
   也就是当前硬币coin的加入，可以把j -coin容量的组合数加入进来
   和01背包差不多，唯一的不同点在于硬币可以重复使用，一个逆序一个正序的区别
3. 返回dp[-1]，也就是dp[amount]

```python
class Solution:
    def change(self, amount: int, coins: List[int]) -> int:
        dp = [1] + [0] * amount
        for coin in coins:
            for j in range(coin, amount + 1):
                dp[j] += dp[j - coin]
        return dp[-1]

```



#### [377. 组合总和 Ⅳ](https://leetcode-cn.com/problems/combination-sum-iv/)

题目：给你一个由 **不同** 整数组成的数组 `nums` ，和一个目标整数 `target` 。请你从 `nums` 中找出并返回总和为 `target` 的元素组合的个数。

```
输入：nums = [1,2,3], target = 4
输出：7
解释：
所有可能的组合为：
(1, 1, 1, 1)
(1, 1, 2)
(1, 2, 1)
(1, 3)
(2, 1, 1)
(2, 2)
(3, 1)
请注意，顺序不同的序列被视作不同的组合。
```

##### 递归

要求构成` target `有多少种组合方法，这里的变量应该是` target`，所以，令函数` dp(x)` 表示从 `nums `中挑选数字可以构成 x 的方法数（递归最基本的就是理解这个定义！）。最终返回的应该是` dp(target)`。

对于题目输入` nums = [1,2,3], target = 4 `时：要求有多少种方法能够成 4，即要求`dp(4)`。思考过程如下。

我们遍历` nums`，判断如果构成` target `的时候，选择了` nums[i]`，那么剩余的` target - nums[i] `仍在 nums 中选的话，会有多少种方法。

+ 对于` nums[0] = 1`, 我们要求有多少种方法能够成` target - nums[0] = 4 - 1 = 3`，即要求`dp(3)`；
  + 对于` nums[1] = 2`, 我们要求有多少种方法能够成` target - nums[1] = 4 - 2 = 2`，即要求` dp(2)`；
+ 对于` nums[2] = 3`, 我们要求有多少种方法能够成` target - nums[2] = 4 - 3 = 1`，即要求 `dp(1)`；

所以，`dp(4) = dp(3) + dp(2) + dp(1)`。然后调用函数继续求解 dp(3), dp(2) 和 dp(1)。

```python
class Solution(object):
    def combinationSum4(self, nums, target):
        if target < 0:
            return 0
        if target == 0:
            return 1
        res = 0
        for num in nums:
            res += self.combinationSum4(nums, target - num)
        return res
```

##### 记忆化递归

```python
class Solution(object):
    def combinationSum4(self, nums, target):
        self.dp = [-1] * (target + 1)
        self.dp[0] = 1
        return self.dfs(nums, target)

    def dfs(self, nums, target):
        if target < 0: return 0
        if self.dp[target] != -1:
            return self.dp[target]
        res = 0
        for num in nums:
            res += self.dfs(nums, target - num)
        self.dp[target] = res
        return res
```

##### 动态规划

这一组合方式与518题不同，本题考虑元素之间的顺序。

不考虑元素间顺序：**求组合数就是外层for循环遍历物品，内层for遍历背包**。

```
for coin in coins:
            for j in range(coin, amount + 1):
```

考虑元素间顺序：**求排列数就是外层for遍历背包，内层for循环遍历物品**。

```python
class Solution:
    def combinationSum4(self, nums: List[int], target: int) -> int:
        dp =  [1] + [0] * target
        for i in range(1, target + 1):
            for num in nums:
                if i >= num:
                    dp[i] += dp[i - num]
        return dp[-1]
```



### 三类常见背包问题

##### 1. 组合问题

[377. 组合总和 Ⅳ](#377. 组合总和 Ⅳ)
[494. 目标和](#494. 目标和)
[518. 零钱兑换 II](#518. 零钱兑换 II)

##### 2. True、False问题

[139. 单词拆分](https://leetcode-cn.com/problems/word-break/)
[416. 分割等和子集](#416. 分割等和子集)

##### 3. 最大最小问题

[474. 一和零](#474. 一和零)
[322. 零钱兑换](#322. 零钱兑换)

组合问题公式：`dp[i] += dp[i-num]`

True、False问题公式：`dp[i] = dp[i] or dp[i-num]`

最大最小问题公式：`dp[i] = min(dp[i], dp[i-num]+1)或者dp[i] = max(dp[i], dp[i-num]+1)`

以上三组公式是解决对应问题的核心公式。

当然拿到问题后，需要做到以下几个步骤：
1.分析是否为背包问题。
2.是以上三种背包问题中的哪一种。
3.是0-1背包问题还是完全背包问题。也就是题目给的nums数组中的元素是否可以重复使用。
4.如果是组合问题，是否需要考虑元素之间的顺序。需要考虑顺序有顺序的解法，不需要考虑顺序又有对应的解法。

接下来讲一下背包问题的判定
背包问题具备的特征：给定一个target，target可以是数字也可以是字符串，再给定一个数组nums，nums中装的可能是数字，也可能是字符串，问：能否使用nums中的元素做各种排列组合得到target。

##### 背包问题技巧

1.如果是0-1背包，即数组中的元素不可重复使用，nums放在外循环，target在内循环，且内循环**倒序**；

```
for num in nums:
    for i in range(target, num-1, -1):
```

2.如果是完全背包，即数组中的元素可重复使用，nums放在外循环，target在内循环。且内循环**正序**。

```
for num in nums:
    for i in range(num, target+1):
```

3.如果组合问题需考虑元素之间的顺序，需将target放在外循环，将nums放在内循环。

```
for i in range(1, target+1):
    for num in nums:
```



