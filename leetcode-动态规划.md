## leetcode题解 - 动态规划



[TOC]

### 递归

定义：在函数中**调用函数自身**的方法。

递归的两个必要条件：1.递归终止条件   2.建立递归函数之间的联系（转移）

举最简单的例子：爬楼梯

```python
def f(n):
    # 终止条件
	if n == 1:
		return 1
    if n == 2:
        return 2
    # 状态转移
    return f(n-1) + f(n-2)
```

使用递归函数的优点是逻辑简单清晰，缺点是过深的调用会导致栈溢出。

### 记忆化递归：特殊的递归函数

以上爬楼梯的递归函数会进行重复计算。我们可以将计算过的值进行存储，遇到相同参数时，不必重新计算。

例如：用一个字典`memo`存储参数`n`对应的`f(n)`的值

```python
memo = {}
def f(n):
  if n == 1: return 1
  if n == 2: return 2
  if n in memo: return memo[n]
  ans = func(n - 1) + func(n-2)
  memo[n] = ans
  return ans
```

在算法上，动态规划和**查表的递归（也称记忆化递归）** 有很多相似的地方。

递归中**如果**存在重复计算（我们称重叠子问题），那就是使用动态规划解题的强有力信号之一。动态规划的核心就是使用记忆化的手段消除重复子问题的计算。

### 动态规划的基本概念

**两个概念**

+ **无后效性**：子问题的解一旦确定，就不再改变，不受在这之后、包含它的更大的问题的求解决策影响。无后效性决定了是否可使用动态规划来解决。

+ **最优子结构**：如果问题的最优解所包含的子问题的解也是最优的，我们就称该问题具有最优子结构性质。最优子结构决定了具体如何解决。

#### 动态规划的步骤

1. **定义状态**
2. **写出状态转移方程（子关系递推关系）**
3. **确定DP数组的计算顺序**
4. **空间优化（滚动数组优化）**

爬楼梯问题递归转为动态规划

```
f(1) 与 f(2) 就是【边界】
f(n) = f(n-1) + f(n-2) 就是【状态转移公式】
```

```python
# dp[0] 与 dp[1] 就是【边界】
# dp[n] = dp[n - 1] + dp[n - 2] 就是【状态转移方程】
def climbStairs(self, n: int) -> int:
    dp = [0] * (n + 1)
    dp[0] = dp[1] = 1
    for i in range(2, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]
    return dp[-1]
```

他们的区别只不过是**递归用调用栈枚举状态， 而动态规划使用迭代枚举状态。**

#### 滚动数组优化

爬楼梯我们并没有必要使用一维数组，而是借助两个变量来实现。这个技巧叫做滚动数组。

```python
def f(n):
    a = b = 1
    for i in range(2, n + 1):
        a, b = b, a + b
    return b
```

**强调**

+ 如果说递归是从问题的结果倒推，直到问题的规模缩小到寻常。 那么动态规划就是从寻常入手， 逐步扩大规模到最优子结构。

+ 记忆化递归和动态规划没有本质不同。都是枚举状态，并根据状态直接的联系逐步推导求解。

+ 动态规划性能通常更好。 一方面是递归的栈开销，一方面是滚动数组的技巧。

#### 动态规划的应用场景

+ 一种是`求最优解类`

  典型问题是背包问题

  递推性质还有一个名字，叫做 「最优子结构」 ——即当前问题的最优解取决于子问题的最优解

+ 另一种就是`计数类`

  例如统计方案数的问题，它们都存在一定的递推性质。

  前问题的方案数取决于子问题的方案数。所以在遇到求方案数的问题时，我们可以往动态规划的方向考虑。



### 1. 打家劫舍

[力扣 198](https://leetcode-cn.com/problems/house-robber/)

题目：抢劫一排住户，但是不能抢邻近的住户，求最大抢劫量。

```
输入：[2,7,9,3,1]
输出：12
```

解题：

+ 定义状态：`dp[i]`表示前`i`个房子的最大抢劫量
+ 转移方程：`dp[n] = max(dp[n-1], dp[n-2] + num)`

```python
class Solution:
    def rob(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [0] * (n + 1)
        dp[1] = nums[0]
        for i in range(1, len(nums)):
            dp[i+1] = max(dp[i], dp[i-1] + nums[i])
        return dp[-1]
```

空间优化

```python
class Solution:
    def rob(self, nums: List[int]) -> int:
        cur, pre = 0, 0
        for num in nums:
            cur, pre = max(pre + num, cur), cur
            # 相当于
            # pre = cur
            # cur = max(pre + num, cur)
        return cur
```

[力扣 213. 打家劫舍 II](https://leetcode-cn.com/problems/house-robber-ii/)

题目：所有的房屋都 **围成一圈** ，这意味着第一个房屋和最后一个房屋是紧挨着的。不能抢邻近的住户，求最大抢劫量。

```
输入：nums = [2,3,2]
输出：3
解释：你不能先偷窃 1 号房屋（金额 = 2），然后偷窃 3 号房屋（金额 = 2）, 因为他们是相邻的。
```

解题：将问题拆成两部分  1.选nums[0]     2. 选nums[-1]， 这两部分的解决方法跟198题一样。

```python
class Solution:
    def rob(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 1: return nums[0]
        def robRange(nums):
            cur, pre = 0, 0
            for num in nums:
                cur, pre = max(pre + num, cur), cur
            return cur
        return max(robRange(nums[0: n - 1]), robRange(nums[1: n]))
```



### 2. 解码方法

[力扣 91](https://leetcode-cn.com/problems/decode-ways/)

**题目：**一条包含字母 A-Z 的消息通过以下方式进行了编码：

```
'A' -> 1
'B' -> 2
...
'Z' -> 26
```


给定一个只包含数字的非空字符串，请计算解码方法的总数。

```
输入：s = "226"
输出：3
解释：它可以解码为 "BZ" (2 26), "VF" (22 6), 或者 "BBF" (2 2 6) 。

输入：s = "06"
输出：0
解释："06" 不能映射到 "F" ，因为字符串开头的 0 无法指向一个有效的字符。 
```

**解题：**

动态转移方程：

+ 只选择当前字母：`dp[i] = dp[i-1]`
+ 选择当前字母和前一个字母的组合：`dp[i] = dp[i-2]`
+ 若两种情况都满足：`dp[i] = dp[i-1] + dp[i-2]`

```python
class Solution(object):
    def numDecodings(self, s: str) -> int:
        if s.startswith('0'):  # 开头有 ‘0’ 直接返回
            return 0

        n = len(s)
        dp = [0] * (n+1)
        dp[0] = dp[1] = 1

        for i in range(2, n+1):
            if s[i-1] != '0':
                dp[i] += dp[i-1]
            if '10' <= s[i-2:i] <= '26':
                dp[i] += dp[i-2]
        return dp[-1]
```

### 3.单词拆分

[力扣 139](https://leetcode-cn.com/problems/word-break/)

**题目：**给定一个非空字符串` s `和一个包含非空单词的列表` wordDict`，判定 `s` 是否可以被空格拆分为一个或多个在字典中出现的单词。

拆分时可以重复使用字典中的单词。

```
输入: s = "leetcode", wordDict = ["leet", "code"]
输出: true

输入: s = "leetcode", wordDict = ["lee", "code", "tc"]
输出: false
```

**状态转移方程：**

+ `dp[i]`：长度为`i`的`s[0:i-1]`子串是否能拆分成单词
+ `s[0:i] `子串的 `dp[i+1] `，是否为真（是否可拆分成单词），取决于两点：
  + 它的前缀子串` s[0:j-1] `的 `dp[j] `，是否为真。
  + 剩余子串` s[j:i]`，是否是一个独立的单词。

```python
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        wordset = set(wordDict)
        n = len(s)
        dp = [False] * (n + 1)
        dp[0] = True  #空字符串为True

        for i in range(1, n+1):
            for j in range(0, i):
                if dp[j] and s[j:i] in wordset:
                    dp[i] = True
        return dp[-1]
```



### 4. 最佳买卖股票时机含冷冻期

[力扣 309](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/)

**题目：**给定一个整数数组，其中第 i 个元素代表了第 i 天的股票价格 。

设计一个算法计算出最大利润。在满足以下约束条件下，你可以尽可能地完成更多的交易（多次买卖一支股票）:

+ 你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。、
+ 卖出股票后，你无法在第二天买入股票 (即冷冻期为 1 天)。

```
输入: [1,2,3,0,2]
输出: 3 
解释: 对应的交易状态为: [买入, 卖出, 冷冻期, 买入, 卖出]
```

**解题：**考虑有多少种状态，每种状态有哪些选择，或者是做了哪些选择后得到哪种状态。注意：到底是先选择了才有状态，还是先由状态才能选择。这里是先选择了，才有状态。

状态类型有2种：天数和是否持有。

+ 天数：一共为1-n天
+ 是否持有：分为**持有状态**、**不持股且当天没卖出**、**不持股且当天卖出了**

dp表示的含义：

+ `dp[i][0] `: **持有状态**
+ `dp[i][1] `: **不持股且当天没卖**
+ `dp[i][2] `: **不持股且当天卖出**

状态转移方程：

+ `dp[i][0]`: 第i天为持有状态时，可能第i-1天也是持有状态，或者第i-1天不持股且没有卖出

  所以`dp[i][0] = max(dp[i-1][0], dp[i-1][1]-prices[n])`

+ `dp[i][1]`: 第i天为没持有状态且没有卖出时，可能i-1天也没持有，可能是i-1天卖了也可能不是。
  
+ 无处理后达到该状态： `dp[i][1] = max(dp[i-1][1], dp[i-1][2]) `， 有两种到达该状态的情况，取最大那个
  
+ `dp[i][2]`: 第i天为没持有状态且在第i天卖出时，那么第i-1天一定持有股票
  
  + 卖出后达到该状态： `dp[i][2] = dp[i-1][0]+prices[i]`

最后`max(dp[n-1][1], dp[n-1][2])`就是题目所需答案。即第n-1天没持有股票时的最大收益

```cpp
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        n = len(prices)
        if n < 2: return 0
            
        dp = [[0, 0, 0] for _ in range(n)]
        dp[0][0] = -prices[0]
            
        for i in range(1, len(prices)):
            dp[i][0] = max(dp[i-1][0], dp[i-1][1] - prices[i])
            dp[i][1] = max(dp[i - 1][1], dp[i - 1][2])
            dp[i][2] = dp[i - 1][0] + prices[i]
        return max(dp[-1])
```

### 5. 最小路径和

[力扣 64](https://leetcode-cn.com/problems/minimum-path-sum/)

题目：给定一个包含非负整数的 `m x n` 网格 `grid` ，请找出一条从左上角到右下角的路径，使得路径上的数字总和为最小。

解题：转移方程：`dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + grid[i][j]`

```python
class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        dp = [[0] * n for _ in range(m)]

        dp[0][0] = grid[0][0]
        for j in range(1, n):
            dp[0][j] = dp[0][j-1] + grid[0][j]
        for i in range(1, m):
            dp[i][0] = dp[i-1][0] + grid[i][0]

        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + grid[i][j]
        return dp[-1][-1]
```

[力扣 62. 不同路径](https://leetcode-cn.com/problems/unique-paths/)

题目：一个机器人位于一个 m x n 网格的左上角 （起始点在下图中标记为 “Start” ）。机器人每次只能向下或者向右移动一步。问总共有多少条不同的路径？

解题：从左上角到右下角的走法 = 从右边开始走的路径总数+从下边开始走的路径总数

+ 所以可推出动态方程为
  `dp[i][j] = dp[i-1][j]+dp[i][j-1]`
+ 初始化第一行和第一列的值
  `dp[0][j] = 1，dp[i][0] = 1`
  因为一直向下或者一直向右走而不转向的话只有一种走法

```js
var uniquePaths = function(m, n) {
    let dp = new Array(m)
    for(let i = 0; i < m; i++) {
        dp[i] = new Array(n).fill(0)
    }
    for(let i = 0; i < m; i++) {
        for(let j = 0; j < n; j++) {
            if(i == 0 || j == 0) dp[i][j] = 1
            if(i > 0 && j > 0) dp[i][j] = dp[i-1][j]+dp[i][j-1]
        }
    }
    return dp[m-1][n-1]
};
```

### 6. 等差数列划分

[力扣 413](https://leetcode-cn.com/problems/arithmetic-slices/)

题目：函数要返回数组 A 中所有为等差数组的子数组个数。

```
A = [0, 1, 2, 3, 4]
返回: 6
[0, 1, 2],
[1, 2, 3],
[0, 1, 2, 3],
[0, 1, 2, 3, 4],
[ 1, 2, 3, 4],
[2, 3, 4]
```

解题：如果`A[i]-A[i-1]`和`A[i-1]-A[i-2]`相等那就找到了一个结果，`dp[i] = dp[i-1]+1`

```
dp[2] = 1
    [0, 1, 2]
dp[3] = dp[2] + 1 = 2
    [0, 1, 2, 3], // [0, 1, 2] 之后加一个 3
    [1, 2, 3]     // 新的递增子区间
dp[4] = dp[3] + 1 = 3
    [0, 1, 2, 3, 4], // [0, 1, 2, 3] 之后加一个 4
    [1, 2, 3, 4],    // [1, 2, 3] 之后加一个 4
    [2, 3, 4]        // 新的递增子区间
```

因为递增子区间不一定以最后一个元素为结尾，可以是任意一个元素结尾，因此需要返回 dp 数组累加的结果。

```python
class Solution:
    def numberOfArithmeticSlices(self, nums: List[int]) -> int:
        n = len(nums)
        if n < 3: return 0
        dp = [0] * n
        for i in range(2, n):
            if nums[i] - nums[i - 1] == nums[i - 1] - nums[i - 2]:
                dp[i] = dp[i-1] + 1
        return sum(dp)
```

### 7. 整数拆分

[力扣 343](https://leetcode-cn.com/problems/integer-break/)

题目：给定一个正整数 *n*，将其拆分为**至少**两个正整数的和，并使这些整数的乘积最大化。

```
输入: 10
输出: 36
解释: 10 = 3 + 3 + 4, 3 × 3 × 4 = 36。
```

解题：当 i≥2 时，假设对正整数 i拆分出的第一个正整数是 j，则有以下两种方案：

+ 将 i 拆分成 j 和 i−j 的和，且 i−j 不再拆分成多个正整数，此时的乘积是`j×(i−j)`；
+ 将 i 拆分成 j 和 i−j 的和，且 i−j 继续拆分成多个正整数，此时的乘积是` j×dp[i−j]`。

```python
class Solution:
    def integerBreak(self, n: int) -> int:
        dp = [0] * (n + 1)
        for i in range(2, n + 1):
            for j in range(i):
                dp[i] = max(dp[i], j * (i - j), j * dp[i - j])
        return dp[-1]
```

数学：尽可能拆成3和2的乘积。

```python
class Solution:
    def integerBreak(self, n: int) -> int:
        if n <= 3:
            return n - 1
        
        quotient, remainder = n // 3, n % 3
        if remainder == 0:
            return 3 ** quotient
        elif remainder == 1:
            return 3 ** (quotient - 1) * 4
        else:
            return 3 ** quotient * 2

```

### 8. 完全平方数

[279. 完全平方数](https://leetcode-cn.com/problems/perfect-squares/)

题目：给定正整数 *n*，找到若干个完全平方数（比如 `1, 4, 9, 16, ...`）使得它们的和等于 *n*。你需要让组成和的完全平方数的个数最少。

```
输入：n = 12
输出：3 
解释：12 = 4 + 4 + 4
```

解题：

类似于完全背包问题（322题 零钱兑换），在数组`[1,4, 9，...]`里找到`target = n `需要的最小零钱个数。

```python
class Solution:
    def numSquares(self, n: int) -> int:
        dp = [n] * (n + 1)
        dp[0] = 0
        for i in range(1, n + 1):
            for j in range(int(math.sqrt(i)), 0, -1):
                dp[i] = min(dp[i], dp[i - j*j] + 1)
        return dp[-1]
```

或

```python
class Solution:
    def numSquares(self, n: int) -> int:
        dp = [n] * (n + 1)
        dp[0] = 0
        for i in range(1, int(math.sqrt(n))+1):
            for j in range(i * i, n + 1):
                dp[j] = min(dp[j], dp[j - i * i] + 1)
        return dp[-1]
```

栈

```python
class Solution:
    def numSquares(self, n: int) -> int:
        res = 0
        queue = [(n, res)]
        while queue:
            num, j = queue.pop(0)
            for i in range(int(math.sqrt(num)), 0, -1):
                if num - i * i == 0: return j + 1
                queue.append((num - i * i, j + 1))
```



### 9. 摆动序列

[力扣 376](https://leetcode-cn.com/problems/wiggle-subsequence/)

题目：如果连续数字之间的差严格地在正数和负数之间交替，则数字序列称为**摆动序列。**第一个差（如果存在的话）可能是正数或负数。少于两个元素的序列也是摆动序列。

```
输入: [1,7,4,9,2,5]
输出: 6 
解释: 整个序列均为摆动序列。

输入: [1,17,5,10,13,15,10,5,16,8]
输出: 7
解释: 这个序列包含几个长度为 7 摆动序列，其中一个可为[1,17,10,13,10,16,8]。
```

解题：遍历序列：最长上升摆动序列`up`，最长下降摆动序列`down`

+ 当前数 = 前数：摆动序列长度不变
+ 当前数 > 前数：
  - 当前数放入最长下降摆动序列`down + 1 → up`
  - 当前数放入最长上升摆动序列，替换原末位值 或 舍弃` up → up`

+ 当前数 < 前数：
  - 当前数放入最长上升摆动序列` up + 1 → down`
  - 当前数放入最长下降摆动序列，替换原末位值 或 舍弃` down → down`

```python
class Solution:
    def wiggleMaxLength(self, nums: List[int]) -> int:
        n = len(nums)
        if n < 2:
            return n
        
        up = [1] + [0] * (n - 1)
        down = [1] + [0] * (n - 1)
        for i in range(1, n):
            if nums[i] > nums[i - 1]:
                up[i] = max(up[i - 1], down[i - 1] + 1)
                down[i] = down[i - 1]
            elif nums[i] < nums[i - 1]:
                up[i] = up[i - 1]
                down[i] = max(up[i - 1] + 1, down[i - 1])
            else:
                up[i] = up[i - 1]
                down[i] = down[i - 1]
        
        return max(up[n - 1], down[n - 1])
```

空间优化

```js
var wiggleMaxLength = function(nums) {
    if (nums.length < 2) return nums.length
    let up = 1, down = 1
    for (let i = 1; i < nums.length; i++) 
        if (nums[i] > nums[i - 1]) 
            up = Math.max(up, down + 1)
        else if (nums[i] < nums[i - 1]) 
            down = Math.max(down, up + 1)
    return Math.max(up, down)
};
```

### 10. 最长公共子序列

[力扣 1143](https://leetcode-cn.com/problems/longest-common-subsequence/)

题目：给定两个字符串 `text1` 和 `text2`，返回这两个字符串的最长 **公共子序列** 的长度。如果不存在 **公共子序列** ，返回 `0` 。

```
输入：text1 = "abcde", text2 = "ace" 
输出：3  
解释：最长公共子序列是 "ace" ，它的长度为 3 。
```

注意：在python中， dp二维数组 的初始化 不能 用`dp = [[0] * N ] * M`这种方式。这会导致 dp 中的每行的列表是同一个 id，所以对其中一行的操作都会表现为每一行的操作。

```python
class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        m, n = len(text1), len(text2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if text1[i - 1] == text2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    
        return dp[m][n]
```

[力扣 583](https://leetcode-cn.com/problems/delete-operation-for-two-strings/)

给定两个单词 *word1* 和 *word2*，找到使得 *word1* 和 *word2* 相同所需的最小步数，每步可以删除任意一个字符串中的一个字符。

```
输入: "sea", "eat"
输出: 2
解释: 第一步将"sea"变为"ea"，第二步将"eat"变为"ea"
```

解题：方法1：可以先求出最长公共子序列，最少删除字符数 = len1 + len2 - 最长公共子序列 * 2

​		   方法2：直接用动态规划。https://leetcode-cn.com/problems/delete-operation-for-two-strings/solution/liang-ge-zi-fu-chuan-de-shan-chu-cao-zuo-by-leetco/

```python
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        m, n = len(word1), len(word2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            for j in range(n + 1):
                if i == 0 or j == 0:
                    dp[i][j] = i + j
                elif word1[i - 1] == word2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = min(dp[i - 1][j], dp[i][j - 1]) + 1
        return dp[m][n]
```

[力扣 72. 编辑距离](https://leetcode-cn.com/problems/edit-distance/)

（583题的进阶版）

给你两个单词 word1 和 word2，请你计算出将 word1 转换成 word2 所使用的最少操作数 。

你可以对一个单词进行如下三种操作：1.插入一个字符 2.删除一个字符 3.替换一个字符

```
输入：word1 = "horse", word2 = "ros"
输出：3
解释：
horse -> rorse (将 'h' 替换为 'r')
rorse -> rose (删除 'r')
rose -> ros (删除 'e')
```

解题：动态规划

定义 `dp[i][j]`
+ `dp[i][j] `代表 word1 中前 i 个字符，变换到 word2 中前 j 个字符，最短需要操作的次数。需要考虑 word1 或 word2 一个字母都没有，即全增加/删除的情况，所以预留` dp[0][j] `和 `dp[i][0]`

状态转移

`word1[i-1] != word2[j-1]`

+ 增，`dp[i][j] = dp[i][j - 1] + 1`
+ 删，`dp[i][j] = dp[i - 1][j] + 1`
+ 改，`dp[i][j] = dp[i - 1][j - 1] + 1`

按顺序计算，`dp[i][j] ` = `min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1] )`

`word1[i-1] == word2[j-1]`

+ 如果刚好这两个字母相同，那么可以直接参考` dp[i - 1][j - 1] `

```python
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        n1, n2 = len(word1), len(word2)
        dp = [[0] * (n2 + 1) for _ in range(n1 + 1)]

        for i in range(n1 + 1):
            for j in range(n2 + 1):
                if i == 0 or j == 0:
                    dp[i][j] = i + j
                elif word1[i - 1] == word2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                elif word1[i - 1] != word2[j - 1]:
                    dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
        return dp[-1][-1]
```

### 其他题目

#### [1269. 停在原地的方案数](https://leetcode-cn.com/problems/number-of-ways-to-stay-in-the-same-place-after-some-steps/)

有一个长度为 arrLen 的数组，开始有一个指针在索引 0 处。每一步操作中，你可以将指针向左或向右移动 1 步，或者停在原地（指针不能被移动到数组范围外）。

给你两个整数 steps 和 arrLen ，请你计算并返回：在恰好执行 steps 次操作以后，指针仍然指向索引 0 处的方案数。由于答案可能会很大，请返回方案数 模 10^9 + 7 后的结果。

```
输入：steps = 3, arrLen = 2
输出：4
解释：3 步后，总共有 4 种不同的方法可以停在索引 0 处。
向右，向左，不动
不动，向右，向左
向右，不动，向左
不动，不动，不动
```

+ `dp[i][j]`：在 i步操作之后，指针位于下标 j的方案数。

  `0 <= i <= steps`, `0 <= j <= min(arrlen-1,steps/2 + 1 )`

+ 状态转移：`dp[i][j] = dp[i][j] + dp[i][j-1] + dp[i][j+1]`

+ 特例：`dp[0][0] == 1, dp[0][j] == 0`

  ​			`j=0`时，只能向右或不动；`j=min(arrlen-1,steps/2 + 1 )`时，只能向左或不动

```python
class Solution:
    def numWays(self, steps: int, arrLen: int) -> int:
        mod = 10 ** 9 + 7
        col = min(arrLen - 1, math.ceil(steps / 2))
        dp = [[0] * (col + 1) for _ in range(steps + 1)]
        dp[0][0] = 1
        for i in range(1, steps + 1):
            for j in range(0, col + 1):
                dp[i][j] = dp[i - 1][j]
                if j - 1 >= 0:
                    dp[i][j] = (dp[i][j] + dp[i - 1][j - 1]) % mod
                if j + 1 <= col:
                    dp[i][j] = (dp[i][j] + dp[i - 1][j + 1]) % mod
        return dp[steps][0]
```

