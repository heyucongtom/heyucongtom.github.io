"""Chapter 2"""

def insertionSort(lst, start, end):
	for i in range(start+1, end):
		key = lst[i]
		j = i-1
		while j >= start and key < lst[j]:
			lst[j+1] = lst[j]
			j -= 1
		lst[j+1] = key

def merge(lst, start, mid, end):
	"""
	Merge an lst, sorted from start to mid, and mid+1 to end
	"""
	left_lst = lst[start:mid]
	left_lst.append(float('inf'))

	right_lst = lst[mid:end]
	right_lst.append(float('inf'))
	i = j = 0
	k = start

	while k - start < end - start:
		if left_lst[i] < right_lst[j]:
			lst[k] = left_lst[i]
			i += 1
		else:
			lst[k] = right_lst[j]
			j += 1
		k += 1

def modifiedMergeSort(lst, start, end, k):
	"""
	if length of lst part is smaller than k, then do insertion
	else, do merge.
	"""
	if end - start <= k:
		insertionSort(lst, start, end)
	else:
		mid = (start + end) // 2
		modifiedMergeSort(lst, start, mid, k)
		modifiedMergeSort(lst, mid, end, k)
		merge(lst, start, mid, end)

def bubbleSort(lst, start, end):
	"""
	Swap elements.
	"""
	for i in range(start, end):
		for j in range(i+1, end):
			if lst[i] > lst[j]:
				temp = lst[i]
				lst[i] = lst[j]
				lst[j] = temp

def maximumSubArray(arr):
	"""
	Algo 4.1
	Linear algo for maximum Sub-array.
	"""
	n = len(arr)
	max_val = [-float('inf') for i in range(n)]
	max_val[0] = arr[0]
	for i in range(1, n):
		max_val[i] = max(max_val[i-1]+arr[i], arr[i])
	return max_val[n-1]

def findCrossMax(arr, low, mid, high):
	i = mid-1
	total = 0
	while i >= low and arr[i] > 0:
		total += arr[i]
		i -= 1	
	j = mid+1
	while j <= high and arr[j] > 0:
		total += arr[j]
		j += 1
	return i+1, j-1, total+arr[mid]

def maximumSubArray2(arr, low, high):

	if high == low:
		return low, high, arr[low]
	else:
		mid = (low + high) // 2
		left_low, left_high, left_val = maximumSubArray2(arr, low, mid)
		right_low, right_high, right_val = maximumSubArray2(arr, mid+1, high)
		cross_low, cross_high, cross_val = findCrossMax(arr, low, mid, high)
		if left_val > right_val and left_val > cross_val:
			return left_low, left_high, left_val
		if right_val > left_val and right_val > cross_val:
			return right_low, right_high, right_val
		else:
			return cross_low, cross_high, cross_val

		print(left_val, right_val, cross_val)

def heapify(A, i, heapsize):
	"""
	heap start at index 1
	[node, 1, 2, 3, 4, 5, 6, 7 ...]
	"""
	l = i * 2
	r = i * 2 + 1
	if l <= heapsize and A[i] < A[l]:
		largest = l
	else:
		largest = i
	if r <= heapsize and A[largest] < A[r]:
		largest = r
	if largest != i:
		A[i], A[largest] = A[largest], A[i]
		heapify(A, largest, heapsize)

def build_heap(A):
	A.insert(0, float('-inf'))
	heapsize = len(A) - 1
	i = heapsize // 2
	while i >= 1:
		heapify(A, i, heapsize)
		i -= 1

def heap_sort(A):
	build_heap(A)
	i = len(A) - 1
	while i >= 2:
		A[i], A[1] = A[1], A[i]
		i -= 1 # i indicates heapsize
		heapify(A, 1, i)

class PriorityQueue(object):

	def __init__(self):
		self._A = [float('-inf')]
		self.heapsize = 0

	def maximum(self):
		if self.heapsize == 0:
			raise Exception('Empty!')
		return self._A[self.heapsize]

	def extract_maximum(self):
		if self.heapsize == 0:
			raise Exception('Empty!')
		self._A[self.heapsize], self._A[1] = self._A[1], self._A[self.heapsize]
		self.heapsize -= 1
		heapify(self._A, 1, self.heapsize)
		
		return self._A.pop()

	def increase_key(self, idx, k):
		if k < self._A[idx]:
			raise Exception('k must be larger than A[idx]')
		self._A[idx] = k
		while idx > 1 and self._A[idx // 2] < self._A[idx]:
			self._A[idx // 2], self._A[idx] = self._A[idx], self._A[idx // 2]
			idx = idx // 2

	def insert(self, k):
		self._A.append(float('-inf'))
		self.heapsize += 1
		self.increase_key(self.heapsize, k)

pq = PriorityQueue()

lst = [3,1,2,-2,3,-6,3,1,2,5]
for i in lst:
	pq.insert(i)
for i in range(len(lst)):
	print(pq.extract_maximum())

# heap_sort(lst)
# print(lst)





