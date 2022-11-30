# Leetcode 题解 - 二分查找

[TOC]

- 当 `while (left < right)` 时，对应的更新式是 `left = middle + 1` ， `right = middle`
- 当 `while (left <= right)` 时，对应的更新式是 `left = middle + 1`，`right = middle - 1`

#### [69. x 的平方根](https://leetcode-cn.com/problems/sqrtx/)

实现 `int sqrt(int x)` 函数。结果只保留整数的部分，小数部分将被舍去。

```python
class Solution:
    def mySqrt(self, x: int) -> int:
        if x <= 1: return x
        l, r = 1, x
        while l <= r:
            mid = l + (r - l) // 2
            sqrtx = x // mid
            if sqrtx == mid:
                return mid
            elif sqrtx > mid:
                l = mid + 1
            else:
                r = mid - 1
        return r
```

#### [744. 寻找比目标字母大的最小字母](https://leetcode-cn.com/problems/find-smallest-letter-greater-than-target/)

给定一个有序的字符数组 letters 和一个字符 target，要求找出 letters 中大于 target 的最小字符，如果找不到就返回第 1 个字符。

```python
class Solution:
    def nextGreatestLetter(self, letters: List[str], target: str) -> str:
        n = len(letters)
        l, r = 0, n - 1
        while l <= r:
            mid = l + (r - l) // 2
            if letters[mid] > target:
                r = mid - 1
            else:
                l = mid + 1
        if l == n: return letters[0]
        else: return letters[l]
```

#### [540. 有序数组中的单一元素](https://leetcode-cn.com/problems/single-element-in-a-sorted-array/)

给定一个只包含整数的有序数组，每个元素都会出现两次，唯有一个数只会出现一次，找出这个数。

```python
class Solution:
    def singleNonDuplicate(self, nums: List[int]) -> int:
        n = len(nums)
        l, r = 0, n - 1
        while l < r:
            mid = l + (r - l) // 2 
            if mid % 2 == 1:
                mid -= 1   # 确保mid在偶数下标位置
            if nums[mid] == nums[mid + 1]: # 若偶下标与下一个下标的数相同，要找的数在后半段
                l =mid + 2
            else:
                r = mid
        return nums[l]
```

#### [278. 第一个错误的版本](https://leetcode-cn.com/problems/first-bad-version/)

题目描述：给定一个元素 n 代表有 [1, 2, ..., n] 版本，在第 x 位置开始出现错误版本，导致后面的版本都错误。可以调用` isBadVersion(int x) `知道某个版本是否错误，要求找到第一个错误的版本。

如果第 m 个版本出错，则表示第一个错误的版本在 [l, m] 之间，令 h = m；否则第一个错误的版本在 [m + 1, h] 之间，令 l = m + 1。

因为 h 的赋值表达式为 h = m，因此循环条件为 l < h。

```
class Solution:
    def firstBadVersion(self, n):
        l, r = 1, n
        while l < r:
            mid = l + (r - l) // 2
            if isBadVersion(mid):
                r = mid
            else:
                l = mid + 1
        return l
```

#### [153. 寻找旋转排序数组中的最小值](https://leetcode-cn.com/problems/find-minimum-in-rotated-sorted-array/)

```
Input: [3,4,5,1,2],
Output: 1
```

```python
class Solution:
    def findMin(self, nums: List[int]) -> int:    
        l, r = 0, len(nums) - 1
        while l < r:
            mid = l + (r - l) // 2
            if nums[mid] <= nums[r]:
                r = mid
            else:
                l = mid + 1
        return nums[l]
```

#### [34. 在排序数组中查找元素的第一个和最后一个位置](https://leetcode-cn.com/problems/find-first-and-last-position-of-element-in-sorted-array/)

题目描述：给定一个有序数组 nums 和一个目标 target，要求找到 target 在 nums 中的第一个位置和最后一个位置。

```
Input: nums = [5,7,7,8,8,10], target = 8
Output: [3,4]
```

可以用二分查找找出第一个位置和最后一个位置，但是寻找的方法有所不同，需要实现两个二分查找。我们将寻找 target 最后一个位置，转换成寻找 target+1 第一个位置，再往前移动一个位置。这样我们只需要实现一个二分查找代码即可。

```python
class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        def findnum(nums, target):
            l, r = 0, len(nums)
            while l < r :
                m = l + (r - l) // 2
                if nums[m] >= target:
                    r = m
                else:
                    l = m + 1
            return l
        first = findnum(nums, target)
        last = findnum(nums, target + 1) - 1
        if first == len(nums) or nums[first] != target:
            return [-1, -1]
        else:
            return [first, max(first, last)]
```

在寻找第一个位置的二分查找代码中，需要注意 h 的取值为` nums.length`，而不是`nums.length - 1`。先看以下示例：

```
nums = [2,2], target = 2
```

如果 h 的取值为` nums.length - 1`，那么 `last = findFirst(nums, target + 1) - 1 = 1 - 1 = 0`。这是因为` findLeft `只会返回` [0, nums.length - 1] `范围的值，对于` findFirst([2,2], 3) `，我们希望返回 3 插入 nums 中的位置，也就是数组最后一个位置再往后一个位置，即` nums.length`。所以我们需要将 h 取值为 `nums.length`，从而使得` findFirst`返回的区间更大，能够覆盖 target 大于 nums 最后一个元素的情况。

#### 其他题目

#### [852. 山脉数组的峰顶索引](https://leetcode-cn.com/problems/peak-index-in-a-mountain-array/)

```python
class Solution:
    def peakIndexInMountainArray(self, arr: List[int]) -> int:
        n = len(arr)
        left, right, ans = 1, n - 2, 0

        while left <= right:
            mid = (left + right) // 2
            if arr[mid] > arr[mid + 1]:
                ans = mid
                right = mid - 1
            else:
                left = mid + 1
        
        return ans
```

