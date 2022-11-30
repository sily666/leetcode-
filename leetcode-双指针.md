# leetcode题解-双指针

双指针是一种思想，技巧或方法，并不是特别具体的算法。具体就是用两个变量动态存储两个或多个结点。通常在线性的数据结构中（链表和数组）。

### 1.有序数组的两数之和

[力扣 167](https://leetcode-cn.com/problems/two-sum-ii-input-array-is-sorted/)

**题目：**给定一个已按照 **升序排列** 的整数数组 `numbers` ，请你从数组中找出两个数满足相加之和等于目标数 `target` 。

```
输入：numbers = [2,7,11,15], target = 9
输出：[1,2]
```

**解题：**以上题目中数组是**有序的**，则可以用`双指针`解决。

使用双指针，一个指针指向值较小的元素，一个指针指向值较大的元素。指向较小元素的指针从头向尾遍历，指向较大元素的指针从尾向头遍历。

- 如果两个指针指向元素的和 sum == target，那么得到要求的结果；
- 如果 sum > target，移动较大的元素，使 sum 变小一些；
- 如果 sum < target，移动较小的元素，使 sum 变大一些。

```python
class Solution:
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        i, j = 0, len(numbers) - 1
        while i < j:
            if numbers[i] + numbers[j] == target:
                return [i+1, j+1]
            elif numbers[i] + numbers[j] < target:
                i += 1
            else:
                j -= 1
        return []
```

### 扩展1：两数之和

若题目中数组是**无序**的，用**哈希表**解决

[力扣 1](https://leetcode-cn.com/problems/two-sum/)

用 hashMap 存一下遍历过的元素和对应的索引。**将每一个遍历的值放入 map 中作为 key，下标作为value**
每访问一个元素，查看一下 hashMap 中是否存在满足要求的目标数字。
所有事情在一次遍历中完成，因为用了空间换取时间。

```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        hashtable = dict()
        for i, num in enumerate(nums):
            if target - num in hashtable:
                return [hashtable[target - num], i]
            hashtable[nums[i]] = i
        return []
```

### 扩展2：三数之和

**题目：**给你一个包含 n 个整数的数组 nums，判断 nums 中是否存在三个元素 a，b，c ，使得 a + b + c = 0 ？请你找出所有满足条件且不重复的三元组。

```
给定数组 nums = [-1, 0, 1, 2, -1, -4]，
满足要求的三元组集合为：
[
  [-1, 0, 1],
  [-1, -1, 2]
]
```

**思路**

- 先将数组进行排序
- 从左侧开始，选定一个值为 定值 ，右侧进行求解，获取与其相加为 0的两个值
- 类似于快排，定义首和尾（双指针）
- 首尾与 定值 相加
  1. 等于 0，记录这三个值
  2. 小于 0，首部右移
  3. 大于 0，尾部左移
- 定值右移，重复该步骤

```js
var threeSum = function(nums) {
    let ans = [];
    const len = nums.length;
    if(nums == null || len < 3) return ans;
    nums.sort((a, b) => a - b); // 排序
    for (let i = 0; i < len ; i++) {
        if(nums[i] > 0) break; // 如果当前数字大于0，则三数之和一定大于0，所以结束循环
        if(i > 0 && nums[i] == nums[i-1]) continue; // 去重
        let L = i+1;
        let R = len-1;
        while(L < R){
            const sum = nums[i] + nums[L] + nums[R];
            if(sum == 0){
                ans.push([nums[i],nums[L],nums[R]]);
                while (L<R && nums[L] == nums[L+1]) L++; // 去重
                while (L<R && nums[R] == nums[R-1]) R--; // 去重
                L++;
                R--;
            }
            else if (sum < 0) L++;
            else if (sum > 0) R--;
        }
    }        
    return ans;
};

```



### 2.反转字符串中的元音字母

[力扣 345](https://leetcode-cn.com/problems/reverse-vowels-of-a-string/)

**题目：**编写一个函数，以字符串作为输入，反转该字符串中的元音字母。

```
输入："hello"
输出："holle"
```

**解题：**利用字典查询速度快，左右双指针遍历，由于要交换元音，需要把str转换成list处理，最后再join转回str。

```python
class Solution:
    def reverseVowels(self, s: str) -> str:
        lib = {'a','e','i','o','u','A','E','I','O','U'}
        nums = list(s)
        i, j = 0, len(nums) - 1
        while i <= j:
            if nums[i] in lib and nums[j] in lib:
                nums[i], nums[j] = nums[j], nums[i]
                i += 1
                j -= 1
            elif nums[i] not in lib:
                i += 1
            elif nums[j] not in lib:
                j -= 1
        return ''.join(nums)
```

### 3. 验证回文字符串

[力扣 680](https://leetcode-cn.com/problems/valid-palindrome-ii/)

**题目：**给定一个非空字符串 `s`，**最多**删除一个字符。判断是否能成为回文字符串。

```
输入: "abca"
输出: True
```

**解题：**本题的关键是处理删除一个字符。在使用双指针遍历字符串时，如果出现两个指针指向的字符不相等的情况，我们就试着删除一个字符，再判断删除完之后的字符串是否是回文字符串。

在判断是否为回文字符串时，我们不需要判断整个字符串，因为左指针左边和右指针右边的字符之前已经判断过具有对称性质，所以只需要判断中间的子字符串即可。

在试着删除字符时，我们既可以删除左指针指向的字符，也可以删除右指针指向的字符。

```python
class Solution(object):
    def validPalindrome(self, s):
        isPalindrome = lambda x : x == x[::-1]
        left, right = 0, len(s) - 1
        while left <= right:
            if s[left] == s[right]:
                left += 1
                right -= 1
            else:
                return isPalindrome(s[left + 1 : right + 1]) or isPalindrome(s[left: right])
        return True
```

### 4. 合并两个有序数组

[力扣 88](https://leetcode-cn.com/problems/merge-sorted-array/)

**题目：**给你两个有序整数数组 `nums1` 和 `nums2`，请你将 `nums2` 合并到 `nums1` 中。

```
输入：nums1 = [1,2,3,0,0,0], m = 3, 
	 nums2 = [2,5,6], n = 3
输出：[1,2,2,3,5,6]
```

**解题：**需要从尾开始遍历，否则在` nums1 `上归并得到的值会覆盖还未进行归并比较的值。

```python
class Solution:
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        i, j = m - 1, n - 1
        index = m + n - 1
        while i >= 0 or j >= 0:
            if i < 0:
                nums1[index] = nums2[j]
                j -= 1
            elif j < 0:
                nums1[index] = nums1[i]
                i -= 1
            elif nums1[i] <= nums2[j]:
                nums1[index] = nums2[j]
                j -= 1
            else:
                nums1[index] = nums1[i]
                i -= 1
            index -= 1
```

### 5.判断链表是否存在环

[力扣 141](https://leetcode-cn.com/problems/linked-list-cycle/)

**题目：**给定一个链表，判断链表中是否有环。

**解题：**使用双指针，一个指针每次移动一个节点，一个指针每次移动两个节点，如果存在环，那么这两个指针一定会相遇。

```python
class Solution:
    def hasCycle(self, head: ListNode) -> bool:
        if not head or not head.next:
            return False
        
        slow = head
        fast = head.next

        while slow != fast:
            if not fast or not fast.next:
                return False
            slow = slow.next
            fast = fast.next.next
        
        return True
```

### 6. 删除有序数组中的重复项

[力扣 26](https://leetcode-cn.com/problems/remove-duplicates-from-sorted-array/)

**题目：**给你一个有序数组 `nums` ，请你**原地** 删除重复出现的元素，使每个元素 **只出现一次** ，返回删除后数组的新长度。

```
输入：nums = [1,1,2]
输出：2, nums = [1,2]
解释：函数应该返回新的长度 2 ，并且原数组 nums 的前两个元素被修改为 1, 2 。不需要考虑数组中超出新长度后面的元素。
```

**解题：**首先注意数组是有序的，那么重复的元素一定会相邻。要求删除重复元素，实际上就是将不重复的元素移到数组的左侧。

考虑用快慢指针，一个在前记作 p，一个在后记作 q，算法流程如下：

- 比较 p 和 q 位置的元素是否相等。
  - 如果相等，q 后移 1 位
  - 如果不相等，将 q 位置的元素复制到 p+1 位置上，p 后移一位，q 后移 1 位
    重复上述过程，直到 q 等于数组长度。

返回 p + 1，即为新数组长度。

```python
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        p = 0
        for q in range(1, len(nums)):
            if nums[q] != nums[p]:
                nums[p+1] = nums[q]
                p += 1
        return p+1
```

### 扩展1：删除有序数组中的重复项 II

若题目改为：给你一个有序数组 `nums` ，请你原地删除重复出现的元素，使每个元素 **最多出现两次** ，返回删除后数组的新长度。

[力扣 80](https://leetcode-cn.com/problems/remove-duplicates-from-sorted-array-ii/)（medium）

**解题：**我们需要检查上上个应该被保留的元素`nums[p-2]` 是否和当前待检查元素`nums[q]` 相同。

```python
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        n = len(nums)
        if n <= 2:
            return n
        
        p = 2
        for q in range(2, n):
            if nums[q] != nums[p-2]:
                nums[p] = nums[q]
                p += 1
        return p
```

### 扩展2：移除元素

**题目：**给你一个数组 `nums` 和一个值 `val`，你需要原地移除所有数值等于 `val` 的元素，并返回移除后数组的新长度。

[力扣 27](https://leetcode-cn.com/problems/remove-element/)

**解题：**快慢指针

```python
class Solution:
    def removeElement(self, nums: List[int], val: int) -> int:
        left = 0
        for right in nums:
            if right != val:
                nums[left] = right
                left += 1
        return left
```



### 7. 最长子序列

[力扣 524](https://leetcode-cn.com/problems/longest-word-in-dictionary-through-deleting/)（medium）

**题目：**删除 s 中的一些字符，使得它构成字符串列表 d 中的一个字符串，找出能构成的最长字符串。如果有多个相同长度的结果，返回字典序的最小字符串。

```
输入:s = "abpcplea", d = ["ale","apple","monkey","plea"]
输出: "apple"
```

**解题：**通过删除字符串 s 中的一个字符能得到字符串 t，可以认为 t 是 s 的子序列，我们可以使用双指针来判断一个字符串是否为另一个字符串的子序列。

先对字符串按长度排序，逐一判断是否为子序列。

```python
class Solution:
    def findLongestWord(self, s: str, d: List[str]) -> str:
        d.sort(key=lambda x: (-len(x), x))#对字典d进行排序，第一关键字是长度降序，第二关键字是字符串本身字典排序
        def f(c):                   #匹配函数
            i = 0
            for j in c:             #遍历单词里的字母
                k = s.find(j, i)    #查找函数，后一个参数是查找起点
                if k == -1:
                    return False    #查找失败就返回错误
                i = k + 1           #查找成功就更新查找起点
            return True
        for c in d:                 #遍历字符串列表
            if f(c):                #如果符合验证就输出
                return c
        return ''                   #否则输出空串
```

