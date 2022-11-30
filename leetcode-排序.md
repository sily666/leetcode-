# leetcode-排序

![排序](./img/排序.jpg)

#### [215. 数组中的第K个最大元素](https://leetcode-cn.com/problems/kth-largest-element-in-an-array/)

题目描述：找到倒数第 k 个的元素。

**排序** ：时间复杂度 O(NlogN)，空间复杂度 O(1)

```python
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        nums.sort()
        return nums[-1-k+1]
```

**堆** ：时间复杂度 O(NlogK)，空间复杂度 O(K)。

堆排序这种解法在总体数据规模 n 较大，而维护规模 k 较小时对时间复杂度的优化会比较明显

```python
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        heap = []
        for num in nums:
            heapq.heappush(heap, num)
            if len(heap) > k:
                heapq.heappop(heap) 
        return heap[0]
```

**快速选择** ：时间复杂度 O(N)，空间复杂度 O(1)

原版快排是O(nlogn)，而这里只需要在一个分支递归，因此降为O(n)

快排基本思想：

+ 选定Pivot中心轴

+ 将大于Pivot的数字放在Pivot的右边

+ 将小于Pivot的数字放在Pivot的左边

+ 分别对左右子序列重复以上步骤

  https://www.bilibili.com/video/BV1at411T75o?from=search&seid=1125440240352583843

```python
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        def partition(left, right):
            pivot = nums[left]
            l = left + 1
            r = right
            while l <= r:
                if nums[l] < pivot and nums[r] > pivot:
                    nums[l], nums[r] = nums[r], nums[l]
                if nums[l] >= pivot:
                    l += 1
                if nums[r] <= pivot:
                    r -= 1
            nums[r], nums[left] = nums[left], nums[r] # 注意：这里是nums[left]而不是pivot，要改变的数组的值
            if r == k - 1:
                return nums[r]
            elif r > k - 1:
                return partition(left, r - 1)
            else:
                return partition(r + 1, right)

        return partition(0, len(nums) - 1)
```

### 桶排序

设置若干个桶，每个桶存储出现频率相同的数。桶的下标表示数出现的频率，即第 i 个桶中存储的数出现的频率为 i。把数都放到桶之后，从后向前遍历桶，最先得到的 k 个数就是出现频率最多的的 k 个数。

#### [347. 前 K 个高频元素](https://leetcode-cn.com/problems/top-k-frequent-elements/)

给你一个整数数组 `nums` 和一个整数 `k` ，请你返回其中出现频率前 `k` 高的元素。你可以按 **任意顺序** 返回答案。

```python
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        from collections import Counter
        freq = Counter(nums)
        n = len(nums)
        bucket = [[] for _ in range(n + 1)]
        for i in freq:
            bucket[freq[i]].append(i)
        res = []
        for i in range(n, 0, -1):
            if k == 0: break
            for j in bucket[i]:
                res.append(j)
                k -= 1
        return res
```

#### [451. 根据字符出现频率排序](https://leetcode-cn.com/problems/sort-characters-by-frequency/)

```
输入:"tree"
输出:"eert"
```

```python
class Solution:
    def frequencySort(self, s: str) -> str: 
        from collections import Counter
        freq = Counter(s)
        n = len(s)
        res = ''
        bucket = [[] for _ in range(n + 1)]
        for i in freq:
            bucket[freq[i]].append(i)
        for i in range(n, 0, -1):
            if bucket[i] == []:
                continue
            for c in bucket[i]:
                res += c * i
        return res
```

### 荷兰国旗问题

有三种颜色的球，算法的目标是将这三种球按颜色顺序正确地排列。它其实是三向切分快速排序的一种变种，在三向切分快速排序中，每次切分都将数组分成三个区间：小于切分元素、等于切分元素、大于切分元素，而该算法是将数组分成三个区间：等于红色、等于白色、等于蓝色。

#### [75. 颜色分类](https://leetcode-cn.com/problems/sort-colors/)

```
Input: [2,0,2,1,1,0]
Output: [0,0,1,1,2,2]
```

快排中选择一个标定元素`pivot`，通过一次扫描，可以把数组分成三部分，正好符合当前问题的场景。

用三个指针表示0,1,2，一次遍历。

```python
class Solution:
    def sortColors(self, nums: List[int]) -> None:
        zero, one, two = -1, 0, len(nums)
        while one < two:
            if nums[one] == 2:
                two -= 1
                nums[one], nums[two] = nums[two], nums[one]
            elif nums[one] == 0:
                zero += 1
                nums[zero], nums[one] = nums[one], nums[zero]
                one += 1
            else:
                one += 1
```

