# leetcode-链表

[TOC]



#### [160. 相交链表](https://leetcode-cn.com/problems/intersection-of-two-linked-lists/)

给你两个单链表的头节点 `headA` 和 `headB` ，请你找出并返回两个单链表相交的起始节点。如果两个链表没有交点，返回 `null` 。

https://leetcode-cn.com/problems/intersection-of-two-linked-lists/solution/intersection-of-two-linked-lists-shuang-zhi-zhen-l/

```python
class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        A, B = headA, headB
        while A != B:
            A = A.next if A else headB
            B = B.next if B else headA
        return A
```

#### [206. 反转链表](https://leetcode-cn.com/problems/reverse-linked-list/)

给你单链表的头节点 `head` ，请你反转链表，并返回反转后的链表。

```python
class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        prev, curr = None, head
        while curr is not None:
            next = curr.next
            curr.next = prev
            prev = curr
            curr = next
        return prev
```

#### [21. 合并两个有序链表](https://leetcode-cn.com/problems/merge-two-sorted-lists/)

将两个升序链表合并为一个新的 **升序** 链表并返回。新链表是通过拼接给定的两个链表的所有节点组成的。

迭代

```python
class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        p = prev = ListNode(-1)
        while l1 and l2:
            if l1.val < l2.val:
                prev.next = l1
                l1 = l1.next
            else:
                prev.next = l2
                l2 = l2.next
            prev = prev.next
        prev.next = l1 if l1 else l2
        return p.next
```

递归

```python
class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        if l1 == None: return l2
        if l2 == None: return l1
        if l1.val < l2.val:
            l1.next = self.mergeTwoLists(l1.next, l2)
            return l1
        else:
            l2.next = self.mergeTwoLists(l1, l2.next)
            return l2
```

#### [83. 删除排序链表中的重复元素](https://leetcode-cn.com/problems/remove-duplicates-from-sorted-list/)

存在一个按升序排列的链表，给你这个链表的头节点 `head` ，请你删除所有重复的元素，使每个元素 **只出现一次** 。

```python
class Solution:
    def deleteDuplicates(self, head: ListNode) -> ListNode:
        if not head:
            return head
        cur = head
        while cur.next:
            if cur.val == cur.next.val:
                cur.next = cur.next.next
            else:
                cur = cur.next
        return head
```

```python
class Solution:
    def deleteDuplicates(self, head: ListNode) -> ListNode:
        if head == None or head.next == None:
            return head
        head.next = self.deleteDuplicates(head.next)
        return head.next if head.next.val == head.val else head
```

#### [19. 删除链表的倒数第 N 个结点](https://leetcode-cn.com/problems/remove-nth-node-from-end-of-list/)

给你一个链表，删除链表的倒数第 `n` 个结点，并且返回链表的头结点。

```python
class Solution:
    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        pre = slow = fast = ListNode(-1)
        fast.next = head
        while n:
            fast = fast.next
            n -= 1
        while fast.next:
            slow = slow.next
            fast = fast.next
        slow.next = slow.next.next
        return pre.next
```

#### [24. 两两交换链表中的节点](https://leetcode-cn.com/problems/swap-nodes-in-pairs/)

```
输入：head = [1,2,3,4]
输出：[2,1,4,3]
```

```python
class Solution:
    def swapPairs(self, head: ListNode) -> ListNode:
        node = ListNode(-1, head)
        p = node
        q = p.next
        while q and q.next:
            p.next = q.next
            q.next = p.next.next
            p.next.next = q
            p = q
            q = p.next
        return node.next
```

```python
class Solution:
    def swapPairs(self, head: ListNode) -> ListNode:
        if head == None or head.next == None:
            return head
        newHead = head.next
        head.next = self.swapPairs(newHead.next)
        newHead.next = head
        return newHead
```

#### [445. 两数相加 II](https://leetcode-cn.com/problems/add-two-numbers-ii/)

给你两个 **非空** 链表来代表两个非负整数。数字最高位位于链表开始位置。它们的每个节点只存储一位数字。将这两数相加会返回一个新的链表。不能对列表中的节点进行翻转。

```
输入：l1 = [7,2,4,3], l2 = [5,6,4]
输出：[7,8,0,7]
```

解题思路：

本题的主要难点在于链表中数位的顺序与我们做加法的顺序是相反的，为了逆序处理所有数位，我们可以使用**栈**：把所有数字压入栈中，再依次取出相加。计算过程中需要注意进位的情况。

```python
class Solution:
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        s1, s2 = [], []
        while l1:
            s1.append(l1.val)
            l1 = l1.next
        while l2:
            s2.append(l2.val)
            l2 = l2.next
        
        ans = None
        carry = 0
        while s1 or s2 or carry:
            a = 0 if not s1 else s1.pop()
            b = 0 if not s2 else s2.pop()
            num = a + b + carry
            carry = num // 10
            num %= 10
            node = ListNode(num)
            node.next = ans
            ans = node
        return ans
```

#### [234. 回文链表](https://leetcode-cn.com/problems/palindrome-linked-list/)

请判断一个链表是否为回文链表。题目要求：以 O(1) 的空间复杂度来求解。

切成两半，把后半段反转，然后比较两半是否相等。

```python
class Solution:
    def isPalindrome(self, head: ListNode) -> bool:
        def cut(head, cutnode):
            while head.next != cutnode:
                head = head.next
            head.next = None
        
        def reverse(head):
            newhead = None
            while head:
                nextnode = head.next
                head.next = newhead
                newhead = head
                head = nextnode
            return newhead
        
        def isEqual(l1, l2):
            while l1 and l2:
                if l1.val != l2.val: return False
                l1 = l1.next
                l2 = l2.next
            return True

        if not head or not head.next: return True
        slow, fast = head, head.next
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        if fast: 
            slow = slow.next
        cut(head, slow)
        return isEqual(head, reverse(slow))
```

#### [725. 分隔链表](https://leetcode-cn.com/problems/split-linked-list-in-parts/)

把链表分隔成 k 部分，每部分的长度都应该尽可能相同，排在前面的长度应该大于等于后面的。

```python
class Solution:
    def splitListToParts(self, root: ListNode, k: int) -> List[ListNode]:
        n = 0
        p = root
        while p:
            p = p.next
            n += 1
        length = n // k
        mod = n % k

        res = []
        cur = root
        for i in range(k):
            res.append(cur)
            size = length + (1 if mod > 0 else 0)   # 算出每段的长度
            if cur:   # 这里判断cur是否存在，防止cur.next不存在报错
                for j in range(size-1):
                    cur = cur.next
                mod -= 1
                tmp = cur.next   # 把后面一段截掉，后面一段需在后面继续划分
                cur.next = None
                cur = tmp
        return res
```

#### [328. 奇偶链表](https://leetcode-cn.com/problems/odd-even-linked-list/)

给定一个单链表，把所有的奇数节点和偶数节点分别排在一起。

```python
class Solution:
    def oddEvenList(self, head: ListNode) -> ListNode:
        if not head:
            return head
        
        evenHead = head.next
        odd, even = head, evenHead
        while even and even.next:
            odd.next = even.next
            odd = odd.next
            even.next = odd.next
            even = even.next
        odd.next = evenHead
        return head
```

