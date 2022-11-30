# leetcode-前缀和

#### [523. 连续的子数组和](https://leetcode-cn.com/problems/continuous-subarray-sum/)

给你一个整数数组 `nums` 和一个整数 `k` ，编写一个函数来判断该数组是否含有同时满足下述条件的连续子数组：

- 子数组大小 **至少为 2** ，且
- 子数组元素总和为 `k` 的倍数。

如果存在，返回 `true` ；否则，返回 `false` 。

```
输入：nums = [23,2,4,6,7], k = 6
输出：true
解释：[2,4] 是一个大小为 2 的子数组，并且和为 6 。
```

前缀和+哈希表

`两数相减为k的倍数` ==> `两数除以k的余数 相同` 

先记录下前缀和的余数，再遍历。哈希表key存余数，value存下标。若存在哈希表中，判断下标相减是否大于等于2；若不存在，余数添加到表中。

```python
class Solution:
    def checkSubarraySum(self, nums: List[int], k: int) -> bool:
        n = len(nums)
        if n < 2: return False

        arr = [0]
        for i in nums:
            arr.append((arr[-1] + i) % k)
        
        dic = {}
        for i, num in enumerate(arr):
            if num not in dic:
                dic[num] = i
            else:
                if i - dic[num] >= 2:
                    return True
        return False
```

#### [525. 连续数组](https://leetcode-cn.com/problems/contiguous-array/)

给定一个二进制数组 `nums` , 找到含有相同数量的 `0` 和 `1` 的最长连续子数组，并返回该子数组的长度。

```
输入: nums = [0,1,0]
输出: 2
说明: [0, 1] (或 [1, 0]) 是具有相同数量0和1的最长连续子数组。
```

用counter记录0和1的个数差，遍历数组，若为0，counter--；若为1，counter++；哈希表key记录counter值，value记录下标。

![1639555775623](D:\leetcode题目整理\leetcode-前缀和.assets\1639555775623.png)

```python
class Solution:
    def findMaxLength(self, nums: List[int]) -> int:
        hashmap = {0:-1}
        counter = ans = 0
        for i, num in enumerate(nums):
            if num == 0:
                counter -= 1
            else:
                counter += 1
            if counter in hashmap:
                ans = max(ans,i - hashmap[counter])
            else:
                hashmap[counter] = i
        return ans
```

#### [1744. 你能在你最喜欢的那天吃到你最喜欢的糖果吗？](https://leetcode-cn.com/problems/can-you-eat-your-favorite-candy-on-your-favorite-day/)