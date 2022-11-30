# leetcode-贪心思想



[TOC]



保证每次操作都是局部最优的，并且最后得到的结果是全局最优的。

能使用贪心算法解决的问题必须具备「无后效性」，即某个状态以前的过程不会影响以后的状态，只与当前状态有关。

### 1. 分发饼干

[力扣 455](https://leetcode-cn.com/problems/assign-cookies/)

题目：假设你是一位很棒的家长，想要给你的孩子们一些小饼干。但是，每个孩子最多只能给一块饼干。

对每个孩子 i，都有一个胃口值 g[i]，这是能让孩子们满足胃口的饼干的最小尺寸；并且每块饼干 j，都有一个尺寸 s[j] 。如果 s[j] >= g[i]，我们可以将这个饼干 j 分配给孩子 i ，这个孩子会得到满足。你的目标是尽可能满足越多数量的孩子，并输出这个最大数值。

```
输入: g = [1,2], s = [1,2,3]
输出: 2
解释: 
你有两个孩子和三块小饼干，2个孩子的胃口值分别是1,2。
你拥有的饼干数量和尺寸都足以让所有孩子满足。
所以你应该输出2.
```

解题：排序+贪心

```python
class Solution:
    def findContentChildren(self, g: List[int], s: List[int]) -> int:
        g.sort()
        s.sort()
        m, n = len(g), len(s)
        i = j = count = 0
        while i < m and j < n:
            while j < n and g[i] > s[j]:
                j += 1
            if j < n:
                count += 1
            i += 1
            j += 1
        return count
```

### 2. 无重叠区间

[力扣 435](https://leetcode-cn.com/problems/non-overlapping-intervals/)

题目：给定一个区间的集合，找到需要移除区间的最小数量，使剩余区间互不重叠。

```
输入: [ [1,2], [2,3], [3,4], [1,3] ]
输出: 1
解释: 移除 [1,3] 后，剩下的区间没有重叠。
```

解题：先计算最多能组成的不重叠区间个数，然后用区间总个数减去不重叠区间的个数。

在每次选择中，区间的结尾最为重要，选择的区间结尾越小，留给后面的区间的空间越大，那么后面能够选择的区间个数也就越大。

按区间的结尾进行排序，每次选择结尾最小，并且和前一个区间不重叠的区间。

```python
class Solution:
    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        if not intervals: return 0
        intervals.sort(key = lambda x: x[1])
        n = len(intervals)
        right = intervals[0][1]
        ans = 1
        for i in range(1, n):
            if intervals[i][0] >= right:
                ans += 1
                right = intervals[i][1]
        return n - ans
```

[力扣 452. 用最少数量的箭引爆气球](https://leetcode-cn.com/problems/minimum-number-of-arrows-to-burst-balloons/)

题目：气球在一个水平数轴上摆放，可以重叠，飞镖垂直投向坐标轴，使得路径上的气球都被刺破。求解最小的投飞镖次数使所有气球都被刺破。

```
输入：points = [[10,16],[2,8],[1,6],[7,12]]
输出：2
解释：对于该样例，x = 6 可以射爆 [2,8],[1,6] 两个气球，以及 x = 11 射爆另外两个气球
```

解题

```python
class Solution:
    def findMinArrowShots(self, points: List[List[int]]) -> int:
        if not points: return 0
        n = len(points)
        points.sort(key = lambda x : x[1])
        right = points[0][1]
        count = 1
        for i in range(1, n):
            if points[i][0] > right:
                count += 1
                right = points[i][1]
        return count
```

### 3. 根据身高重建队列

[力扣 406](https://leetcode-cn.com/problems/queue-reconstruction-by-height/)

题目描述：一个学生用两个分量 (h, k) 描述，h 表示身高，k 表示排在前面的有 k 个学生的身高比他高或者和他一样高。

```
输入：people = [[7,0],[4,4],[7,1],[5,0],[6,1],[5,2]]
输出：[[5,0],[7,0],[5,2],[6,1],[4,4],[7,1]]
解释：
编号为 0 的人身高为 5 ，没有身高更高或者相同的人排在他前面。
编号为 1 的人身高为 7 ，没有身高更高或者相同的人排在他前面。
编号为 2 的人身高为 5 ，有 2 个身高更高或者相同的人排在他前面，即编号为 0 和 1 的人。
编号为 3 的人身高为 6 ，有 1 个身高更高或者相同的人排在他前面，即编号为 1 的人。
编号为 4 的人身高为 4 ，有 4 个身高更高或者相同的人排在他前面，即编号为 0、1、2、3 的人。
编号为 5 的人身高为 7 ，有 1 个身高更高或者相同的人排在他前面，即编号为 1 的人。
```

解题：为了使插入操作不影响后续的操作，身高较高的学生应该先做插入操作，否则身高较小的学生原先正确插入的第 k 个位置可能会变成第 k+1 个位置。

身高 h 降序、个数 k 值升序，然后将某个学生插入队列的第 k 个位置中。

```python
class Solution:
    def reconstructQueue(self, people: List[List[int]]) -> List[List[int]]:
        res = []
        people.sort(key = lambda x : (-x[0], x[1]))
        for p in people:
            if len(res) <= p[1]:
                res.append(p)
            else:
                res.insert(p[1], p)
        return res
```

### 4. 买卖股票的最佳时机

[力扣 121](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock/)

题目描述：一次股票交易包含买入和卖出，只进行一次交易，求最大收益。

只要记录前面的最小价格，将这个最小价格作为买入价格，然后将当前的价格作为售出价格，查看当前收益是不是最大收益。

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        n = len(prices)
        if n == 0: return 0
        res = 0
        buy = prices[0]
        for i in range(1, n):
            if prices[i] < buy:
                buy = prices[i]
            else:
                res = max(res, prices[i] - buy)
        return res
```

[力扣 122. 买卖股票的最佳时机 II](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-ii/)

题目描述：可以进行多次交易，多次交易之间不能交叉进行，可以进行多次交易。

对于 [a, b, c, d]，如果有 a <= b <= c <= d ，那么最大收益为 d - a。而 d - a = (d - c) + (c - b) + (b - a) ，因此当访问到一个 prices[i] 且 prices[i] - prices[i-1] > 0，那么就把 prices[i] - prices[i-1] 添加到收益中。

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        n = len(prices)
        res = 0
        buy = prices[0]
        for i in range(1, n):
            if prices[i] > buy:
                res += prices[i] - buy
            buy = prices[i]
        return res
```

### 5. 种花问题

[力扣 605](https://leetcode-cn.com/problems/can-place-flowers/)

题目：flowerbed 数组中 1 表示已经种下了花朵。花朵之间至少需要一个单位的间隔，求解是否能种下 n 朵花。

```
输入：flowerbed = [1,0,0,0,1], n = 1
输出：true
```

解题：种花条件为 1. 自己为空 2. 左边为空 或 自己是最左边 3. 右边为空 或 自己是最右边。 为了减少自己为最左最右的特殊判断，可以在flowerbed前后加0

```python
class Solution:
    def canPlaceFlowers(self, flowerbed: List[int], n: int) -> bool:
        m, count = len(flowerbed), 0
        flowerbed = [0] + flowerbed + [0]
        for i in range(1, m + 1):
            if flowerbed[i] == 0 and flowerbed[i-1] == 0 and flowerbed[i+1] == 0:
                count += 1
                flowerbed[i] = 1
            if count >= n:
                return True
        return False
```

### 6. 非递减数列

[力扣 665](https://leetcode-cn.com/problems/non-decreasing-array/)

题目：判断一个数组是否能只修改一个数就成为非递减数组。

```
输入: nums = [4,2,3]
输出: true
解释: 你可以通过把第一个4变成1来使得它成为一个非递减数列。
```

解题：本题是要维持一个非递减的数列，所以遇到递减的情况时（nums[i] > nums[i + 1]），要么将前面的元素缩小，要么将后面的元素放大。如果放大nums[i+1]，会影响后续数列，因此，采用贪心策略，尽可能将nums[i]缩小，不破坏前面序列的非递减性。

```python
class Solution:
    def checkPossibility(self, nums: List[int]) -> bool:
        n = len(nums)
        cnt = 0
        for i in range(n-1):
            if nums[i] > nums[i + 1]:
                cnt += 1
                if i > 0 and nums[i + 1] < nums[i - 1]:
                    nums[i + 1] = nums[i]
        return cnt <= 1
```

### 7. 最大子序和

[力扣 53](https://leetcode-cn.com/problems/maximum-subarray/)

```
输入：nums = [-2,1,-3,4,-1,2,1,-5,4]
输出：6
解释：连续子数组 [4,-1,2,1] 的和最大，为 6 。
```

全局最优：选取最大“连续和”

**局部最优的情况下，并记录最大的“连续和”，可以推出全局最优**。

```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        res = -100001
        count = 0
        for i in range(len(nums)):
            count += nums[i]
            if count > res:
                res = count
            if count < 0:
                count = 0
        return res
```

动态规划解法

```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        res = t = nums[0]
        for i in range(1, len(nums)):
            t = max(nums[i], t + nums[i])
            res = max(res, t)
        return res
```

### 8. 划分字母区间

[力扣 763](https://leetcode-cn.com/problems/partition-labels/)

题目：字符串 `S` 由小写字母组成。我们要把这个字符串划分为尽可能多的片段，同一字母最多出现在一个片段中。返回一个表示每个字符串片段的长度的列表。

```
输入：S = "ababcbacadefegdehijhklij"
输出：[9,7,8]
解释：
划分结果为 "ababcbaca", "defegde", "hijhklij"。
每个字母最多出现在一个片段中。
像 "ababcbacadefegde", "hijhklij" 的划分是错误的，因为划分的片段数较少。
```

显然同一个字母的第一次出现的下标位置和最后一次出现的下标位置必须出现在同一个片段。因此需要遍历字符串，得到每个字母最后一次出现的下标位置，之后用贪心的方法将字符串划分为尽可能多的片段。

+ 遍历字符串，记录每个字母最后一次出现的下标
+ 初始化start = end = 0，对于每一个字母c，片段的结束位置不会小于endc，因此end = max(end, endc)
+ 当访问到下标end时，片段为[start, end]，将片段添加到结果中，令start = end + 1， 继续寻找下一个片段

```python
class Solution:
    def partitionLabels(self, S: str) -> List[int]:
        last = [0] * 26
        for i, ch in enumerate(S):
            last[ord(ch) - ord('a')] = i
        start = end = 0
        res = []
        for i, ch in enumerate(S):
            end = max(end, last[ord(ch) - ord('a')])
            if i == end:
                res.append(end - start + 1)
                start = end + 1
        return res
```

## 其他题目

#### [1833. 雪糕的最大数量](https://leetcode-cn.com/problems/maximum-ice-cream-bars/)

给你价格数组 `costs` 和现金量 `coins` ，请你计算并返回 Tony 用 `coins` 现金能够买到的雪糕的 **最大数量** 。

```
输入：costs = [1,3,2,4,1], coins = 7
输出：4
解释：Tony 可以买下标为 0、1、2、4 的雪糕，总价为 1 + 3 + 2 + 1 = 7
```

贪心而不是01背包

「01 背包」的复杂度为 O(N* C)，其中 N 为物品数量，C为背包容量。显然会 TLE。

优先选择价格小的物品会使得我们剩余金额尽可能的多，将来能够做的决策方案也就相应变多。

```python
class Solution:
    def maxIceCream(self, costs: List[int], coins: int) -> int:
        costs.sort()
        count = 0
        n = len(costs)
        for i in range(n):
            cost = costs[i]
            if coins >= cost:
                coins -= cost
                count += 1
            else:
                break
        return count
```

