from typing import List


class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution:

    # 二叉树深度
    def maxDepth(self, root):
        deque = [root]
        res = 0
        if not root:
            return 0
        while deque:
            tmp = []
            for node in deque:
                if node.left:
                    tmp.append(node.left)
                if node.right:
                    tmp.append(node.right)
            deque = tmp
            res += 1
        return res

    # 链表中倒数第K个节点
    def getKthFromEnd(self, head, k):
        fast, low = head, head
        for _ in range(k):
            fast = fast.next
        while fast:
            fast = fast.next
            low = low.next
        return low

    # 二叉树的镜像
    def mirrorTree(self, root: TreeNode):
        if not root:
            return
        deque = [root]
        while deque:
            tmp = []
            for node in deque:
                if node.left:
                    tmp.append(node.left)
                if node.right:
                    tmp.append(node.right)
                node.left, node.right = node.right, node.left
            deque = tmp
        return root

    # 两数之和
    def twoSum(self, nums, target):
        hashMap = {}
        for i in range(len(nums)):
            hashMap[nums[i]] = i
        for i in range(len(nums)):
            k = hashMap.get(target - nums[i])
            if k is not None and k != i:
                return [i, k]

    # 0901 两数相加
    def addTwoNumbers(self, l1, l2):
        p1, p2  = l1, l2
        cur = ListNode(0)
        dum = cur
        carry = 0

        while p1 or p2:
            x = p1.val if p1 else 0 # 如果p1为空则x为0，否则p1.val
            y = p2.val if p2 else 0
            sum = (x + y + carry) % 10
            carry = (x + y + carry) // 10
            if cur:
                # cur.val = sum
                # cur.next = ListNode(0) # 未初始化next节点，导致cur.next报错NoneType
                cur.next = ListNode(sum) # 初始化头结点为0，从头结点next节点才开始存入计算的值
                cur = cur.next
            if p1:
                p1 = p1.next
            if p2:
                p2 = p2.next
            if carry > 0:
                cur.next = ListNode(carry)
        return dum.next

    # 0901 无重复字符的最长子串
    def lengthOfLongestSubstring(self, s: str) -> int:
        hashMap = {}
        i, j = 0, 0
        length = len(s)
        count = 0

        if not s:
            return 0

        while i < length and j < length:
            if s[j] in hashMap:
                # count = max(count, j - i) // 写在这个位置，会导致无重复字符串如dwsre，不进入条件计算count
                # i = j
                '''
                abcabcbb 0 0  0 {'a': 0}
                abcabcbb 0 1 a 1 {'a': 0, 'b': 1}
                abcabcbb 0 2 ab 2 {'a': 0, 'b': 1, 'c': 2}
                abcabcbb 3 3  2 {'a': 3, 'b': 1, 'c': 2}
                abcabcbb 4 4  2 {'a': 3, 'b': 4, 'c': 2}
                abcabcbb 5 5  2 {'a': 3, 'b': 4, 'c': 5}
                abcabcbb 6 6  2 {'a': 3, 'b': 6, 'c': 5}
                abcabcbb 7 7  2 {'a': 3, 'b': 7, 'c': 5}
                '''
                i = max(hashMap.get(s[j]), i) //"abba"
                # i = hashMap.get(s[j])
            #     hashMap[s[j]] = j
            hashMap[s[j]] = j     #"au" " "
            # hashMap[s[j]] = j + 1
            count = max(count, j - i)
            count = max(count, j - i + 1)
            print(s, s[j], i, j, s[i:j+1], count, hashMap)
            j += 1
        return count

    def subsets(self, nums):
        """ 9/18 78.子集 """
        output = [[]]

        for num in nums:
            # output += [curr + [num] for curr in output]
            # tmp = [curr + [num] for curr in output]
            tmp = []
            print("#########", output)
            for curr in output:
                tmp_in = curr + [num]
                tmp += [tmp_in]
                print(curr, "+", [num], "=", tmp_in, "+=", tmp)
            output += tmp
            # print(tmp)
        print("#########", output)
        return output

    def singleNumber(self, nums):
        """9/23 数组中数字出现的次数"""
        hashMap = {}
        for num in nums:
            if num in hashMap:
                hashMap[num] += 1
            else:
                hashMap[num] = 1
        for k,v in hashMap.items():
            if v == 1:
                return k

    def merge(self, A: List[int], m: int, B: List[int], n: int) -> None:
        """9/23 合并两个有序数组"""
        res = []
        A[:] = A[:m] # [1,0],[2] -> [1,2,0]
        B[:] = B[:n]
        i, j = 0, 0

        while i < m and j < n:
            if A[i] < B[j]:
                res.append(A[i])
                i += 1
            else:
                res.append(B[j])
                j += 1
        if i < m:
            res[i + j:] = A[i:]
        if j < n:
            res[i + j:] = B[j:]
        A[:] = res

    def isSymmetric(self, root: TreeNode) -> bool:
        """9/28 对称的二叉树"""
        def dfs(L, R):
            if not L and not R:
                return True
            if not L or not R or L.val != R.val:
                return False
            return dfs(L.right, R.left) and dfs(L.left, R.right)
        return dfs(root.left, root.right) if root else True

    def removeDuplicates(self, nums: List[int]) -> int:
        """9/28 删除排序数组中的重复项"""
        p, q = 0, 1
        n = len(nums)

        while q < n:
            if nums[p] != nums[q]:
                p += 1
                nums[p] = nums[q]
            q += 1
        return p + 1

    def getNumberOfK(self, data, k):
        """9/29 排序数组中查找数字"""
        l, r =0, len(data)
        while l < r:

            mid = (l + r) // 2
            print(mid, l, r)
            if data[mid] < k:
                l = mid + 1
            else:
                r = mid

    def hasCycle(self, head: ListNode) -> bool:
        """9/30 环形链表"""
        hashMap = {}
        while head:
            if head not in hashMap:
                hashMap[head] = True
            else:
                return True
            head = head.next
        return False

if __name__ == "__main__":
    s = Solution()
    nums = [0,1,2,3,4,5,5,5,6,7,8,9]
    s.getNumberOfK(nums, 5)











