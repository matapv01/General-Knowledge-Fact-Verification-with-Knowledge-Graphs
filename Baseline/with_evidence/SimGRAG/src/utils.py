import re
import heapq


def extract_graph(llm_output):
	try:
		matched = re.search(r'\{.*?\}', llm_output, re.DOTALL)
		decoded = eval(matched.group(0))
		return decoded['graph']
	except:
		raise Exception("Fail to decode rewrited graph")


class kSmallest:
	def __init__(self, k):
		self.k = k
		self.heap = []

	def add(self, score, graph, reuse_nodes):
		heapq.heappush(self.heap, (-score, graph, reuse_nodes))
		if len(self.heap) > self.k:
			all_scores = sorted([-neg_score for neg_score, _, _ in self.heap])
			if all_scores[-1] > all_scores[self.k-1]:
				heapq.heappop(self.heap)
			
	def get(self):
		data = sorted([(-neg_score, graph, reuse_nodes) for neg_score, graph, reuse_nodes in self.heap])
  
		final_results = []
		for score, graph, reuse_nodes in data:
			if not reuse_nodes:
				final_results.append((score, graph, reuse_nodes))
				data.remove((score, graph, reuse_nodes))
				break
		for score, graph, reuse_nodes in data:
			if reuse_nodes:
				final_results.append((score, graph, reuse_nodes))
				data.remove((score, graph, reuse_nodes))
				break
   
		for score, graph, reuse_nodes in data:
			if len(final_results) >= self.k:
				break
			final_results.append((score, graph, reuse_nodes))
		return final_results

	def max_score(self):
		if len(self.heap) < self.k:
			return float('inf')
		return -min(self.heap)[0]


def check_answer(answer, groundtruths):
	for gt in groundtruths:
		if gt.lower() in answer.lower():
			return True
	return False