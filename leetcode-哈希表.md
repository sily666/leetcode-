# leetcode-哈希表

[TOC]

哈希表使用 O(N) 空间复杂度存储数据，并且以 O(1) 时间复杂度求解问题。

+ **HashSet** 用于存储一个集合，可以查找元素是否在集合中。

如果元素有穷，并且范围不大，那么可以用一个布尔数组来存储一个元素是否存在。例如对于只有小写字符的元素，就可以用一个长度为 26 的布尔数组来存储一个字符集合，使得空间复杂度降低为 O(1)。

+ **HashMap** 主要用于映射关系，从而把两个元素联系起来。

HashMap 也可以用来对元素进行计数统计，此时键为元素，值为计数。和 HashSet 类似，如果元素有穷并且范围不大，可以用整型数组来进行统计。在对一个内容进行压缩或者其它转换时，利用 HashMap 可以把原始内容和转换后的内容联系起来。



#### [1. 两数之和](https://leetcode-cn.com/problems/two-sum/)

给定一个整数数组 `nums` 和一个整数目标值 `target`，请你在该数组中找出 **和为目标值** *`target`* 的那 **两个** 整数，并返回它们的数组下标。

```
输入：numbers = [2,7,11,15], target = 9
输出：[1,2]
```

```python
def twoSum(self, nums: List[int], target: int) -> List[int]:
    hashtable = dict()
    for i, num in enumerate(nums):
        if target - num in hashtable:
            return [hashtable[target - num], i]
        hashtable[nums[i]] = i
    return []
```

#### [217. 存在重复元素](https://leetcode-cn.com/problems/contains-duplicate/)

给定一个整数数组，判断是否存在重复元素。

```python
def containsDuplicate(self, nums: List[int]) -> bool:
    Set = set()
    for num in nums:
        if num in Set: 
            return True
        Set.add(num)
    return False
```

#### [594. 最长和谐子序列](https://leetcode-cn.com/problems/longest-harmonious-subsequence/)

和谐数组是指一个数组里元素的最大值和最小值之间的差别 **正好是 `1`** 。应该注意的是序列的元素不一定是数组的连续元素。

```
输入：nums = [1,3,2,2,5,2,3,7]
输出：5
解释：最长的和谐子序列是 [3,2,2,2,3]
```

```python
class Solution:
    def findLHS(self, nums: List[int]) -> int:
        dicts={}
        for i in nums: # 先计数 key为num值
            dicts[i]=dicts.get(i,0)+1  
            #dict.get(key, default=None)返回指定键的值，如果值不在字典中返回default值
        res=0
        for i in dicts: 
            if i+1 in dicts:
                res=max(res,dicts[i]+dicts[i+1])
        return res
```

#### [128. 最长连续序列](https://leetcode-cn.com/problems/longest-consecutive-sequence/)

给定一个未排序的整数数组 `nums` ，找出数字连续的最长序列（不要求序列元素在原数组中连续）。设计并实现时间复杂度为 `O(n)` 的解决方案

```
输入：nums = [100,4,200,1,3,2]
输出：4
解释：最长数字连续序列是 [1, 2, 3, 4]。它的长度为 4。
```

首先是去重。遍历每个数x，如果x-1存在于set中，说明可以直接遍历以x-1开头的序列，所以跳过对x的遍历。每个数只需遍历一次，因此复杂度为O(n)

```python
class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        res = 0
        set_num = set(nums)
        for i in set_num:
            if i-1 not in set_num:
                cur_num = i
                cur_res = 1
                while cur_num + 1 in set_num:
                    cur_num += 1
                    cur_res += 1
                res = max(cur_res, res)
        return res
```

### 其他题目

#### [888. 公平的糖果棒交换](https://leetcode-cn.com/problems/fair-candy-swap/)

爱丽丝和鲍勃有不同大小的糖果棒，他们想交换一根糖果棒，交换后他们都有相同的糖果总量。返回一个整数数组` ans`，其中 `ans[0] `是爱丽丝必须交换的糖果棒的大小，`ans[1] `是 Bob 必须交换的糖果棒的大小。

```
输入：A = [1,2], B = [2,3]
输出：[1,2]
```

```js
var fairCandySwap = function(A, B) {
    const Asum = A.reduce((prev, curr) => prev+curr)
    const Bsum = B.reduce((prev, curr) => prev+curr)
    const cha = (Asum - Bsum) / 2
    let rec = new Set(A)
    for(const y of B) {
        if(rec.has(y + cha)) {
            return [y + cha, y]
        }
    }
};
```

#### [1128. 等价多米诺骨牌对的数量](https://leetcode-cn.com/problems/number-of-equivalent-domino-pairs/)

给你一个由一些多米诺骨牌组成的列表` dominoes`。形式上，`dominoes[i] = [a, b]` 和` dominoes[j] = [c, d] `等价的前提是`a==c 且 b==d`，或是 `a==d 且 b==c`。找出满足 `dominoes[i] 和 dominoes[j] `等价的骨牌对 (i, j) 的数量。

```
输入：dominoes = [[1,2],[2,1],[3,4],[5,6]]
输出：1
```

**我们怎样将等价的二元组`(i,j)和(j,i)`映射到相同的key上？**

可以将key设置为`数字`或`字符串`，可以先比较`i和j的大小`，将小的放在前面，key就是`i*10+j`或者`‘ij’`

```js
var numEquivDominoPairs = function(dominoes) {
    const n = dominoes.length
    let res = 0
    let map = new Map()
    for(const dominoe of dominoes) {
        if(dominoe[0] > dominoe[1]) {
            [dominoe[0], dominoe[1]] = [dominoe[1], dominoe[0]]
        }
        const key = dominoe[0] * 10 + dominoe[1]
        
        map.has(key)?map.set(key, map.get(key)+1):map.set(key, 0)
        res += map.get(key)
    }

    return res
};
```

#### [1743. 从相邻元素对还原数组](https://leetcode-cn.com/problems/restore-the-array-from-adjacent-pairs/)

给你一个二维整数数组` adjacentPairs` ，大小为` n - 1 `，其中每个` adjacentPairs[i] = [ui, vi] `表示元素` ui` 和 `vi` 在` nums `中相邻。返回 原始数组` nums` 。

```
输入：adjacentPairs = [[2,1],[3,4],[3,2]]
输出：[1,2,3,4]
```

将数对记录到map中，每个数对应一个数组，找到数组长度为1的入口，逐渐遍历整个map

```js
var restoreArray = function(a) {
    let map = new Map()
    let set = new Set()
    for(const [x, y] of a) {
        map.has(x) ? map.set(x, [map.get(x)[0], y]) : map.set(x, [y])
        map.has(y) ? map.set(y, [map.get(y)[0], x]) : map.set(y, [x])
    }  // 值为数组，记录与键相邻的数，长度为1或2
       // Map(4) { 2 => [ 1, 3 ], 1 => [ 2 ], 3 => [ 4, 2 ], 4 => [ 3 ] }
    
    let res = []
    for(let i of map) {
        if(i[1].length == 1) {  
            res.push(i[0])
            set.add(i[0])
            break
        }
    }
    
    for(let i = 0; i < a.length; i++) {
        const v = res[res.length - 1] //数组最后一个数为键，找到下一个值
        const next = map.get(v)
        if(set.has(next[0])) { //区分中间的数和始末的数
            res.push(next[1])
            set.add(next[1])
        }else {
            res.push(next[0])
            set.add(next[0])
        }
    }
    return res
};
```

#### [5243. 同积元组](https://leetcode-cn.com/problems/tuple-with-same-product/)

给你一个由 不同 正整数组成的数组` nums `，请你返回满足` a * b = c * d `的元组` (a, b, c, d) `的数量。其中 a、b、c 和 d 都是` nums` 中的元素，且` a != b != c != d` 。

```
输入：nums = [2,3,4,6]
输出：8
解释：存在 8 个满足题意的元组：
(2,6,3,4) , (2,6,4,3) , (6,2,3,4) , (6,2,4,3)
(3,4,2,6) , (3,4,2,6) , (3,4,6,2) , (4,3,6,2) 乘积都为12
```

```
输入：nums = [1,2,4,5,10]
输出：16
解释：存在 16 个满足题意的元组：
(1,10,2,5) , (1,10,5,2) , (10,1,2,5) , (10,1,5,2)
(2,5,1,10) , (2,5,10,1) , (5,2,1,10) , (5,2,10,1) 乘积10
(2,10,4,5) , (2,10,5,4) , (10,2,4,5) , (10,2,4,5)
(4,5,2,10) , (4,5,10,2) , (5,4,2,10) , (5,4,10,2) 乘积20
```

一开始想到的是dfs，但是会超时。

只需要用一个map记录 两数相乘的结果的次数

```js
var tupleSameProduct = function(nums) {
    let map = {}; // 哈希表
    let n = nums.length;
    let res = 0; // 计数器
    for (let i = 0; i < n; ++i) {
        for (let j = i+1; j < n; ++j) {
            let x = nums[i]*nums[j];
            map[x]?(res += map[x], map[x]++):map[x] = 1;
        }
    }
    return res * 8;
};
```

